import ast
import copy
import importlib
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC, collate, scatter
from projects.mmdet3d_plugin.models.dynamics.ctrv_ekf import CTRVEKF, transition_and_jacobian, wrap_angle
from projects.mmdet3d_plugin.models.dynamics.agent_track_bank import AgentTrackBank, associate
from projects.mmdet3d_plugin.models.dynamics.dynamics_consistency import (
    compute_dynamics_loss, selected_predictions, to_world, dynamics_loss, HistoricalPerception)
from projects.mmdet3d_plugin.datasets.dynamics_history import scene_indices, history_indices, sensor_frame_info
from projects.mmdet3d_plugin.datasets.nuscenes_3d_dataset import NuScenes3DDataset
from projects.mmdet3d_plugin.models.sparsedrive import SparseDrive

ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path(__file__).parent / 'baseline'


def original_method(name):
    tree = ast.parse((BASELINE / 'sparsedrive.py.txt').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseDrive')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[method], type_ignores=[])
    namespace = vars(importlib.import_module('projects.mmdet3d_plugin.models.sparsedrive')).copy()
    exec(compile(module, 'original_baseline', 'exec'), namespace)
    return namespace[name]


def synthetic_outputs():
    torch.manual_seed(51)
    boxes = torch.zeros(2, 4, 11, requires_grad=True)
    offsets = (torch.randn(2, 4, 6, 12, 2)*.05).requires_grad_()
    classes = torch.zeros(2, 4, 10)
    classes[..., 0] = 3.
    det = dict(prediction=[boxes], classification=[classes])
    motion = dict(prediction=[offsets], classification=[torch.randn(2, 4, 6)])
    plan = dict(classification=[torch.zeros(2, 1, 18)])
    diagnostic = dict(attn_weights_gnn=torch.tensor([[[.3, .8, .1, .0]]]*2),
                      topk_indices_det=torch.tensor([[2, 0, 3]]*2))
    return (det, {}, motion, plan, diagnostic)


class FakeHead:
    def __init__(self, outputs):
        self.outputs = outputs
    def __call__(self, features, data):
        return self.outputs
    def eval(self):
        return self
    def loss(self, outputs, data):
        return {'planning_loss': outputs[2]['prediction'][0].square().mean(),
                'map_loss': outputs[0]['prediction'][0].square().mean()}


def harness(outputs):
    # Pseudo detections shifted .3m to ensure nonzero L1 with valid matches.
    nxt = list(synthetic_outputs())
    nxt[0] = {**nxt[0], 'prediction': [torch.ones_like(nxt[0]['prediction'][0]).detach()*.3]}
    return SimpleNamespace(evo_cfg=dict(TOP_K_CAUSAL_AGENTS=2, loss_weight=.7),
                           head=FakeHead(tuple(nxt)),
                           extract_feat=lambda *args, **kwargs: ([], None) if len(args) > 1 else [])


def input_data():
    metas = [dict(T_global=np.eye(4), T_global_inv=np.eye(4), timestamp=2.) for _ in range(2)]
    return dict(img_metas=metas, next_img_metas=metas, next_img=torch.zeros(2, 1),
                gt_ego_fut_cmd=torch.tensor([[1, 0, 0]]*2))


class MathTests(unittest.TestCase):
    def test_straight_and_intervals(self):
        for dt in (.01, .5, 1.3):
            ekf = CTRVEKF().initialize([1., 2.], 4., .3, 0.)
            np.testing.assert_allclose(ekf.predict(dt), [1+4*dt*np.cos(.3), 2+4*dt*np.sin(.3)])

    def test_turn(self):
        ekf = CTRVEKF().initialize([0., 0.], 8., .2, .4)
        dt = .5
        expected = [8/.4*(np.sin(.2+.4*dt)-np.sin(.2)),
                    8/.4*(np.cos(.2)-np.cos(.2+.4*dt))]
        np.testing.assert_allclose(ekf.predict(dt), expected)

    def test_jacobian_finite_difference_including_zero(self):
        for omega in (0., 1e-10, 1e-4, -.4, 1.):
            state = np.array([2., 4., 9., .4, omega])
            _, analytic = transition_and_jacobian(state, .7)
            numeric = np.zeros((5, 5))
            for j in range(5):
                h = np.eye(5)[j]*1e-6
                numeric[:, j] = (transition_and_jacobian(state+h, .7)[0]-transition_and_jacobian(state-h, .7)[0])/(2e-6)
            np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-7)

    def test_prior_preserved_and_joseph_covariance(self):
        ekf = CTRVEKF().initialize([0., 0.], 3., 0., .1)
        for i in range(500):
            ekf.predict(.1)
            prior, covariance = ekf.get_prior_position(), ekf.get_prior_covariance()
            if i % 3:
                ekf.update(prior + [.01, -.01])
            np.testing.assert_array_equal(prior, ekf.get_prior_position())
            np.testing.assert_array_equal(covariance, ekf.get_prior_covariance())
            self.assertTrue(np.isfinite(ekf.P).all())
            self.assertGreater(np.linalg.eigvalsh(ekf.P).min(), 0.)
            self.assertTrue(-np.pi <= ekf.x[3] < np.pi)

    def test_position_only_bootstrap_on_straight_and_turn(self):
        for omega in (0., .2):
            truth = CTRVEKF().initialize([0., 0.], 4., .3, omega)
            bank = AgentTrackBank({})
            bank.step(0., truth.x[None, :2].copy(), np.array([0]))
            for i in range(1, 5):
                truth.predict(.5)
                bank.step(i*.5, truth.x[None, :2].copy(), np.array([0]))
            tracks = bank.reliable_tracks()
            self.assertEqual(len(tracks), 1)
            expected = truth.predict(.5)
            prior = bank.predict_prior(tracks[0], 2.5)
            self.assertLess(np.linalg.norm(prior-expected), .5)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            CTRVEKF(measurement_noise=[-1, 1])
        with self.assertRaises(ValueError):
            CTRVEKF().initialize([np.nan, 0])
        with self.assertRaises(ValueError):
            transition_and_jacobian([0]*5, 0)

    def test_warmup_low_speed_missing_expiry(self):
        bank = AgentTrackBank({'ekf_max_track_age': 1.})
        bank.step(0., np.array([[0., 0.]]), np.array([0]))
        self.assertEqual(bank.reliable_tracks(), [])
        bank.step(.5, np.array([[.01, 0.]]), np.array([0]))
        self.assertEqual(bank.reliable_tracks(), [])
        bank.step(1., np.array([[.02, 0.]]), np.array([0]))
        self.assertEqual(len(bank.reliable_tracks()), 1)
        self.assertTrue(np.isfinite(bank.predict_prior(bank.reliable_tracks()[0], 1.5)).all())
        bank.step(1.5, np.empty((0, 2)), np.array([]))
        self.assertEqual(bank.reliable_tracks(), [])
        bank.step(3., np.empty((0, 2)), np.array([]))
        self.assertEqual(bank.tracks, [])

    def test_association_class_gate_one_to_one(self):
        tracks = [dict(ekf=CTRVEKF().initialize([0, 0]), label=0)]
        pairs = associate(np.array([[.1, 0], [.2, 0], [100, 0], [0, 0]]), np.array([0, 0, 0, 1]), tracks)
        self.assertEqual(pairs, [(0, 0)])

    def test_causality_scene_shuffle(self):
        def prior(origin):
            bank = AgentTrackBank({})
            for i in range(5):
                bank.step(i*.5, np.array([[origin+i, 0.]]), np.array([0]))
            return bank.predict_prior(bank.reliable_tracks()[0], 2.5)
        p = prior(0)
        prior(200)  # another scene/rank/order cannot change first sample
        future_detection = np.array([[9999., -9999.]])
        np.testing.assert_array_equal(prior(0), p)
        future_detection[:] = 0
        np.testing.assert_array_equal(prior(0), p)


class LossTests(unittest.TestCase):
    def test_l1_stop_gradient_and_empty(self):
        p = torch.tensor([[1., 3.], [5., 8.]], requires_grad=True)
        teacher = torch.zeros_like(p, requires_grad=True)
        loss = compute_dynamics_loss(p, teacher, torch.tensor([True, False]))
        self.assertEqual(float(loss), 2.)
        loss.backward()
        self.assertIsNone(teacher.grad)
        torch.testing.assert_close(p.grad, torch.tensor([[.5, .5], [0., 0.]]))
        p.grad = None
        z = compute_dynamics_loss(p, teacher, torch.zeros(2, dtype=torch.bool))
        z.backward()
        torch.testing.assert_close(p.grad, torch.zeros_like(p))

    def test_world_coordinate_transform_and_gradient(self):
        t = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
        p = torch.tensor([[2., 3.]], requires_grad=True)
        world = to_world(p, t)
        torch.testing.assert_close(world, torch.tensor([[7., 22.]]))
        world.sum().backward()
        torch.testing.assert_close(p.grad, torch.tensor([[1., -1.]]))
        back = to_world(torch.cat([world.detach(), torch.zeros(1, 1)], -1), np.linalg.inv(t))
        torch.testing.assert_close(back, p.detach())

    def test_decode_exact_legacy(self):
        outputs = synthetic_outputs()
        for b in range(2):
            indices, modes, prediction = selected_predictions(outputs, b, {'TOP_K_CAUSAL_AGENTS': 2})
            self.assertEqual(indices.tolist(), [0, 2])
            offsets = outputs[2]['prediction'][-1][b][indices]
            expected = torch.gather(offsets, 1, modes.reshape(2, 1, 1, 1).expand(2, 1, 12, 2)).squeeze(1).cumsum(1)
            torch.testing.assert_close(prediction, expected[:, 0])

    def test_evo_ast_unchanged(self):
        old = ast.parse((BASELINE/'sparsedrive.py.txt').read_text())
        new = ast.parse((ROOT/'projects/mmdet3d_plugin/models/sparsedrive.py').read_text())
        get = lambda tree: next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'get_evo_loss')
        self.assertEqual(ast.dump(get(old)), ast.dump(get(new)))

    def test_original_evo_values_and_gradients(self):
        old = original_method('get_evo_loss')
        results = []
        for method in (old, SparseDrive.get_evo_loss):
            outputs = synthetic_outputs()
            fake = harness(outputs)
            loss = method(fake, outputs, input_data(), [])['loss_evo']
            loss.backward()
            results.append((loss.detach(), outputs[0]['prediction'][0].grad, outputs[2]['prediction'][0].grad))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_disabled_forward_and_optimizer_equivalence(self):
        old_forward = original_method('forward_train')
        for flags in ({'enable_dynamics_loss': False}, {'enable_dynamics_loss': True, 'lambda_dyn': 0.}):
            results = []
            for forward in (old_forward, SparseDrive.forward_train):
                outputs = synthetic_outputs()
                fake = harness(outputs)
                fake.evo_cfg.update(flags)
                fake.head = FakeHead(outputs)
                fake.get_evo_loss = lambda *args: {'loss_evo': outputs[2]['prediction'][0].abs().mean()}
                params = [outputs[0]['prediction'][0], outputs[2]['prediction'][0]]
                optimizer = torch.optim.AdamW(params, lr=.001)
                with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_loss', side_effect=AssertionError('must not execute')):
                    losses = forward(fake, torch.zeros(2, 1), **input_data())
                sum(losses.values()).backward()
                grads = [p.grad.clone() for p in params]
                optimizer.step()
                results.append((losses, grads, [p.detach().clone() for p in params], optimizer.state_dict()))
            self.assertEqual(results[0][0].keys(), results[1][0].keys())
            for key in results[0][0]:
                torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
            for index in (1, 2):
                for a, b in zip(results[0][index], results[1][index]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            for key in results[0][3]['state']:
                for item in results[0][3]['state'][key]:
                    torch.testing.assert_close(results[0][3]['state'][key][item], results[1][3]['state'][key][item], rtol=0, atol=0)

    def test_historical_teacher_copies_online_weights_without_live_mutation(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(3)
            def forward(self, image):
                return [self.bn(image)]
            def train(self, mode=True):
                super().train(mode)  # mirrors MMCV ResNet.train returning None
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
                self.instance_bank = SimpleNamespace(reset=lambda: None)
                self.sampler = SimpleNamespace(dn_metas={'gt_marker': True})
            def forward(self, features, data):
                assert not any(k.startswith('gt_') for k in data)
                boxes = torch.zeros(1, 1, 11)
                boxes[..., 0] = self.weight
                classes = torch.zeros(1, 1, 10)
                classes[..., 0] = 4.
                return dict(prediction=[boxes], classification=[classes])
        class Model(torch.nn.Module):
            extract_feat = SparseDrive.extract_feat
            def __init__(self):
                super().__init__()
                self.img_backbone = Backbone()
                self.img_neck = None
                self.head = torch.nn.Module()
                self.head.det_head = Detector()
                self.use_deformable_func = False
        model = Model().train()
        frame = dict(img=torch.ones(1, 3, 2, 2), timestamp=torch.tensor(0.),
                     projection_mat=torch.eye(4)[None], image_wh=torch.tensor([[2, 2]]),
                     img_metas=dict(T_global=np.eye(4), timestamp=0.))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        teacher = HistoricalPerception(model)
        first = teacher([frame], {})[0][1]
        for k, v in model.state_dict().items():
            torch.testing.assert_close(before[k], v, rtol=0, atol=0)
        self.assertTrue(model.img_backbone.bn.training)
        self.assertEqual(model.head.det_head.sampler.dn_metas, {'gt_marker': True})
        with torch.no_grad():
            model.head.det_head.weight.fill_(2.)
        second = HistoricalPerception(model)([frame], {})[0][1]
        self.assertEqual(first[0, 0], 1.)
        self.assertEqual(second[0, 0], 2.)

    def test_dynamics_precedes_future_backbone_buffer_updates(self):
        # Original get_evo_loss extracts t+1 with the LIVE training backbone;
        # its BatchNorm buffers therefore cannot be copied by our history branch.
        outputs = synthetic_outputs()
        fake = harness(outputs)
        fake.head = FakeHead(outputs)
        fake.evo_cfg.update(enable_dynamics_loss=True, lambda_dyn=.1)
        fake.future_dependent_buffer = 0.
        def evo(*args):
            fake.future_dependent_buffer = 100.
            return {'loss_evo': outputs[2]['prediction'][0].sum()*0.}
        fake.get_evo_loss = evo
        def dynamic(model, *args):
            return outputs[2]['prediction'][0].sum()*0. + model.future_dependent_buffer
        with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_loss', side_effect=dynamic):
            losses = SparseDrive.forward_train(fake, torch.zeros(2, 1), **input_data())
        self.assertEqual(float(losses['loss_dyn']), 0.)
        self.assertEqual(fake.future_dependent_buffer, 100.)

    def test_synthetic_full_dynamics_causality_gradients(self):
        outputs = synthetic_outputs()
        model = harness(outputs)
        model.evo_cfg.update(ekf_min_observations=3)
        context = dict(scene_token='a', current_timestamp=1., next_timestamp=1.5, has_valid_next_frame=True,
                       history=[dict(scene_token='a', img_metas={'timestamp': i*.5}) for i in range(3)])
        data = input_data()
        data['dynamics_context'] = [context, {**context, 'scene_token': 'b', 'history': []}]
        history = [(i*.5, np.array([[0., 0.]]), np.array([0])) for i in range(3)]
        with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.HistoricalPerception') as teacher:
            teacher.return_value.return_value = history
            loss = dynamics_loss(model, outputs, data)
            changed = {**data, 'next_img': torch.randn(2, 1)*1000, 'next_img_metas': [{'T_global': np.eye(4)*999}]*2}
            loss2 = dynamics_loss(model, outputs, changed)
        torch.testing.assert_close(loss, loss2, rtol=0, atol=0)
        self.assertGreater(model.dynamics_diagnostics['valid_agents'], 0)
        loss.backward()
        self.assertGreater(float(outputs[2]['prediction'][0].grad.abs().sum()), 0)
        self.assertGreater(float(outputs[0]['prediction'][0].grad.abs().sum()), 0)


class DataTests(unittest.TestCase):
    @staticmethod
    def info(scene, time):
        return dict(scene_token=scene, timestamp=time*1e6,
                    ego2global_rotation=[1, 0, 0, 0], ego2global_translation=[0, 0, 0],
                    lidar2ego_rotation=[1, 0, 0, 0], lidar2ego_translation=[0, 0, 0],
                    cams={'camera': dict(data_path='sensor.jpg', sensor2lidar_rotation=np.eye(3),
                                         sensor2lidar_translation=np.zeros(3), cam_intrinsic=np.eye(3))})

    def test_scene_window_interleaved(self):
        infos = [self.info('a', 0), self.info('b', .2), self.info('a', .5), self.info('b', .7), self.info('a', 1.)]
        self.assertEqual(history_indices(infos, scene_indices(infos), 4, 5), [0, 2, 4])
        self.assertEqual(history_indices(infos, scene_indices(infos), 2, 5), [0, 2])
        fields = sensor_frame_info(infos[0])
        self.assertFalse(any(k.startswith('gt_') or 'instance' in k for k in fields))

    def test_configs_only_differ_by_dynamics(self):
        import mmcv
        e0 = mmcv.Config.fromfile(str(ROOT/'projects/configs/sparsedrive_small_stage2_ctrv_e0.py')).to_dict()
        e1 = mmcv.Config.fromfile(str(ROOT/'projects/configs/sparsedrive_small_stage2_ctrv_v1.py')).to_dict()
        self.assertEqual(e0['model']['evo_cfg']['loss_weight'], 1.)
        self.assertEqual(e1['model']['evo_cfg']['loss_weight'], 1.)
        evo0, evo1 = e0['model'].pop('evo_cfg'), e1['model'].pop('evo_cfg')
        for k in ('ENTROPY_THRESHOLD', 'TOP_K_CAUSAL_AGENTS', 'PSEUDO_LABEL_THRESHOLD', 'loss_weight'):
            self.assertEqual(evo0[k], evo1[k])
        e1.pop('dynamics_history_pipeline')
        for k in ('dynamics_history_length', 'dynamics_history_pipeline'):
            e1['data']['train'].pop(k)
        self.assertEqual(e0, e1)

    def test_dataset_container_boundary_and_rng(self):
        dataset = NuScenes3DDataset.__new__(NuScenes3DDataset)
        dataset.data_infos = [self.info('a', i*.5) for i in range(4)] + [self.info('b', 0.)]
        dataset.dynamics_scenes = scene_indices(dataset.data_infos)
        dataset.dynamics_history_length = 5
        dataset.get_augmentation = lambda: dict(rotate_3d=0.)
        dataset.get_data_info = lambda i: dict(timestamp=dataset.data_infos[i]['timestamp']/1e6)
        def pipeline(raw):
            random.random(); np.random.rand(); torch.rand(1)
            meta = dict(timestamp=raw['timestamp'], T_global=np.eye(4), T_global_inv=np.eye(4))
            return dict(img=DC(torch.zeros(1, 3, 2, 2), stack=True), img_metas=DC(meta, cpu_only=True),
                        timestamp=raw['timestamp'], projection_mat=np.eye(4)[None], image_wh=np.array([[2, 2]]))
        dataset.pipeline = pipeline
        dataset.dynamics_pipeline = pipeline
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        a = dataset[2]
        state = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
        dataset.dynamics_history_length = 0
        random.seed(1); np.random.seed(1); torch.manual_seed(1)
        b = dataset[2]
        self.assertEqual(state[0], random.getstate())
        np.testing.assert_array_equal(state[1], np.random.get_state()[1])
        torch.testing.assert_close(state[2], torch.get_rng_state(), rtol=0, atol=0)
        dataset.dynamics_history_length = 5
        last = dataset[3]
        self.assertFalse(last['has_valid_next_frame'].data)
        self.assertTrue(a['has_valid_next_frame'].data)
        self.assertEqual(len(a['dynamics_context'].data['history']), 3)
        self.assertEqual(a['dynamics_context'].data['history'][0]['timestamp'].dtype, torch.float64)
        batch = collate([a, last], samples_per_gpu=2)
        contexts = batch['dynamics_context'].data[0]
        self.assertEqual(len(contexts), 2)
        self.assertIsNot(contexts[0], contexts[1])
        torch.testing.assert_close(a['next_img'].data, b['next_img'].data)

if __name__ == '__main__':
    unittest.main(verbosity=2)
