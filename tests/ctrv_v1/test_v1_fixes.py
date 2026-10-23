"""Targeted V1 fixes: gate, initialization covariance, association and diagnostics."""
import copy
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from mmcv.runner import LogBuffer
from test_ctrv_v1 import synthetic_outputs, harness, input_data, original_method, ROOT
from projects.mmdet3d_plugin.models.dynamics.dynamics_consistency import (
    dynamics_loss, planning_entropy_trigger, selected_predictions, to_world)
from projects.mmdet3d_plugin.models.dynamics.agent_track_bank import AgentTrackBank, associate
from projects.mmdet3d_plugin.models.dynamics.ctrv_ekf import CTRVEKF
from projects.mmdet3d_plugin.models.dynamics.diagnostics import (
    diagnostics_totals, metrics_from_totals, reduce_dynamics_diagnostics, DynamicsDiagnosticsHook)
from projects.mmdet3d_plugin.models.dynamics.gradient_diagnostics import gradient_conflict


def history_data():
    data = input_data()
    data['dynamics_context'] = []
    for scene in ('a', 'b'):
        frames = [dict(scene_token=scene, img_metas=dict(timestamp=i*.5)) for i in range(3)]
        data['dynamics_context'].append(dict(scene_token=scene, history=frames,
            current_timestamp=1., next_timestamp=1.5, has_valid_next_frame=True))
    return data


def stationary_history(frames, cfg):
    return [(i*.5, np.array([[0., 0.]]), np.array([0])) for i in range(3)]


def mixed_outputs():
    outputs = synthetic_outputs()
    # Only batch 0's command-0 row has low entropy; other command rows stay high.
    outputs[3]['classification'][-1][0, 0, :6] = torch.tensor([100., -100., -100., -100., -100., -100.])
    return outputs


class EntropyFixTests(unittest.TestCase):
    def test_mixed_batch_matches_evo_and_zero_low_entropy_gradient(self):
        outputs, data = mixed_outputs(), history_data()
        model = harness(outputs)
        self.assertEqual([planning_entropy_trigger(outputs, data, b, model.evo_cfg) for b in range(2)], [False, True])
        with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.HistoricalPerception') as teacher:
            teacher.return_value.side_effect = stationary_history
            loss = dynamics_loss(model, outputs, data)
            self.assertEqual(teacher.call_count, 1)
            self.assertEqual(teacher.return_value.call_count, 1)
            self.assertEqual(teacher.return_value.call_args.args[0][0]['scene_token'], 'b')
        self.assertGreater(float(loss), 0.)
        loss.backward()
        grad = outputs[2]['prediction'][-1].grad
        self.assertEqual(float(grad[0].abs().sum()), 0.)
        self.assertGreater(float(grad[1].abs().sum()), 0.)
        box_grad = outputs[0]['prediction'][-1].grad
        self.assertEqual(float(box_grad[0].abs().sum()), 0.)
        self.assertGreater(float(box_grad[1].abs().sum()), 0.)
        # Compare high-sample gradient with a batch where low-sample history is invalid.
        alternate = mixed_outputs()
        other_model = harness(alternate)
        data['dynamics_context'][0]['history'] = []
        with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.HistoricalPerception') as teacher:
            teacher.return_value.side_effect = stationary_history
            dynamics_loss(other_model, alternate, data).backward()
        torch.testing.assert_close(grad[1], alternate[2]['prediction'][-1].grad[1], rtol=0, atol=0)
        # Original Evo independently gives exactly the same sample mask.
        original = mixed_outputs()
        evo = original_method('get_evo_loss')(harness(original), original, input_data(), [])['loss_evo']
        evo.backward()
        self.assertEqual(float(original[2]['prediction'][-1].grad[0].abs().sum()), 0.)
        self.assertGreater(float(original[2]['prediction'][-1].grad[1].abs().sum()), 0.)
        self.assertEqual(model.dynamics_diagnostics['entropy_triggered_samples'], 1)
        self.assertEqual(model.dynamics_diagnostics['active_samples'], 1)

    def test_command_missing_and_exact_boundary_match_original(self):
        for command in (0, 1, 2, None):
            for boundary in ('below', 'equal', 'above'):
                outputs, data = mixed_outputs(), input_data()
                if command is None:
                    data.pop('gt_ego_fut_cmd')
                else:
                    data['gt_ego_fut_cmd'] = torch.eye(3)[[command, command]]
                # Uniform entropy for sample 1, computed identically to original.
                row = outputs[3]['classification'][-1][1].reshape(3, 6)[0]
                probs = torch.softmax(row, -1)
                entropy = -(probs*torch.log(probs+1e-9)).sum()
                threshold = float(entropy) + {'below': -1e-3, 'equal': 0., 'above': 1e-3}[boundary]
                model = harness(outputs)
                model.evo_cfg['ENTROPY_THRESHOLD'] = threshold
                expected = [planning_entropy_trigger(outputs, data, b, model.evo_cfg) for b in range(2)]
                loss = original_method('get_evo_loss')(model, outputs, data, [])['loss_evo']
                loss.backward()
                grads = outputs[2]['prediction'][-1].grad
                actual = [grads is not None and bool(grads[b].abs().sum()) for b in range(2)]
                self.assertEqual(actual, expected, (command, boundary))
        self.assertFalse(planning_entropy_trigger(outputs, input_data(), 1, model.evo_cfg))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA FP16 parity requires a GPU')
    def test_fp16_gate_boundary_and_command_parity(self):
        for exact_boundary in (False, True):
            outputs, data = synthetic_outputs(), input_data()
            logits = outputs[3]['classification'][-1].cuda().half()
            # Finite low entropy: avoid inventing a replacement for original dtype.
            logits[0, 0, :6] = logits.new_tensor([8., 0., 0., 0., 0., 0.])
            outputs[3]['classification'][-1] = logits
            model = harness(outputs)
            if exact_boundary:
                probabilities = torch.softmax(logits[1].reshape(3, 6)[0], -1)
                model.evo_cfg['ENTROPY_THRESHOLD'] = float(-(probabilities*torch.log(probabilities+1e-9)).sum())
            expected = [planning_entropy_trigger(outputs, data, b, model.evo_cfg) for b in range(2)]
            self.assertFalse(expected[0])
            loss = original_method('get_evo_loss')(model, outputs, data, [])['loss_evo']
            loss.backward()
            gradients = outputs[2]['prediction'][-1].grad
            self.assertEqual(expected, [gradients is not None and bool(gradients[b].abs().sum()) for b in range(2)])

    def test_all_low_no_teacher_or_bank_and_graph_zero(self):
        outputs, data = mixed_outputs(), history_data()
        outputs[3]['classification'][-1][1] = outputs[3]['classification'][-1][0].clone()
        # Context is not needed at all for samples rejected by the trigger.
        data.pop('dynamics_context')
        with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.HistoricalPerception', side_effect=AssertionError('teacher')), \
             patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.AgentTrackBank', side_effect=AssertionError('EKF')):
            model = harness(outputs)
            loss = dynamics_loss(model, outputs, data)
        self.assertEqual(float(loss), 0.)
        loss.backward()
        self.assertEqual(float(outputs[2]['prediction'][-1].grad.abs().sum()), 0.)
        self.assertEqual(model.dynamics_diagnostics['matched_dyn_count'], 0)
        metrics = reduce_dynamics_diagnostics(model.dynamics_diagnostics, torch.device('cpu'))
        self.assertEqual(metrics['entropy_trigger_rate'], 0.)
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))


class BootstrapFixTests(unittest.TestCase):
    def bank(self, xy, cfg=None):
        bank = AgentTrackBank(cfg or {})
        for i, p in enumerate(xy):
            bank.step(i*.5, np.array([p], dtype=np.float64), np.array([0]))
        self.assertEqual(len(bank.tracks), 1)
        ekf = bank.tracks[0]['ekf']
        self.assertTrue(np.isfinite(ekf.x).all())
        self.assertTrue(np.isfinite(ekf.P).all())
        np.testing.assert_allclose(ekf.P, ekf.P.T, atol=1e-12)
        self.assertGreaterEqual(np.linalg.eigvalsh(ekf.P).min(), 0.)
        np.testing.assert_array_equal(ekf.P[:2, 2:], np.zeros((2, 3)))
        self.assertGreaterEqual(ekf.P[2, 2], 4.-1e-10)
        return bank

    def test_stationary_and_low_speed_remain_direction_uncertain(self):
        for xy in ([[0, 0], [0, 0], [0, 0]], [[0, 0], [.01, 0], [.02, 0]]):
            bank = self.bank(xy)
            ekf = bank.tracks[0]['ekf']
            np.testing.assert_array_equal(ekf.x[2:], np.zeros(3))
            self.assertGreaterEqual(ekf.P[3, 3], 10.-1e-10)
            self.assertGreaterEqual(ekf.P[4, 4], 10.-1e-10)
            position, covariance = bank.predict_prior(bank.tracks[0], 1.5, return_covariance=True)
            self.assertTrue(np.isfinite(position).all())
            self.assertGreater(np.linalg.eigvalsh(covariance).min(), 0.)

    def test_straight_and_turning_mean_and_covariance(self):
        for omega in (0., .2):
            truth = CTRVEKF().initialize([0, 0], 4., .3, omega)
            xy = [truth.x[:2].copy()]
            for _ in range(2):
                xy.append(truth.predict(.5))
            bank = self.bank(xy)
            ekf = bank.tracks[0]['ekf']
            self.assertAlmostEqual(ekf.x[4], omega, places=8)
            self.assertGreaterEqual(ekf.P[4, 4], 1.)
            position, covariance = bank.predict_prior(bank.tracks[0], 1.5, return_covariance=True)
            self.assertLess(np.linalg.norm(position-truth.predict(.5)), .5)
            self.assertGreater(np.trace(covariance), np.trace(ekf.P[:2, :2]))
            self.assertLess(np.linalg.eigvalsh(covariance).max(), 25.)

    def test_two_observations_yaw_unknown_and_configurable_floors(self):
        bank = self.bank([[0, 0], [2, 0]])
        self.assertGreaterEqual(bank.tracks[0]['ekf'].P[4, 4], 10.)
        custom = self.bank([[0, 0], [.01, 0], [.02, 0]], dict(
            ekf_bootstrap_variance_floor=[9., 2., 3.], ekf_bootstrap_unobserved_variance=[16., 20., 30.]))
        np.testing.assert_allclose(np.diag(custom.tracks[0]['ekf'].P)[2:], [9., 20., 30.])

    def test_noisy_observations_and_larger_measurement_noise(self):
        xy = [[0, 0], [2., .25], [4., -.25]]
        clean = self.bank(xy)
        noisy = self.bank(xy, dict(ekf_measurement_noise=[4., 4.], ekf_max_position_variance=1e5))
        ec, en = clean.tracks[0]['ekf'], noisy.tracks[0]['ekf']
        self.assertTrue((np.diag(en.P)[2:] >= np.diag(ec.P)[2:]).all())
        pc = clean.predict_prior(clean.tracks[0], 1.5, return_covariance=True)
        pn = noisy.predict_prior(noisy.tracks[0], 1.5, return_covariance=True)
        self.assertGreater(np.trace(pn[1]), np.trace(pc[1]))
        for _ in range(20):
            en.predict(.1)
            self.assertTrue(np.isfinite(en.P).all())
            self.assertGreater(np.linalg.eigvalsh(en.P).min(), 0.)


class AssociationFixTests(unittest.TestCase):
    def test_two_close_same_class_one_to_one_and_class_gate(self):
        tracks = [dict(ekf=CTRVEKF().initialize([x, 0]), label=label)
                  for x, label in ((0., 0), (.4, 0), (2., 1))]
        positions = np.array([[.35, 0], [.05, 0], [2., 0], [50., 0]])
        self.assertEqual(associate(positions, np.array([0, 0, 0, 0]), tracks), [(0, 1), (1, 0)])
        self.assertEqual(associate(positions, np.array([0, 0, 1, 0]), tracks), [(0, 1), (1, 0), (2, 2)])

    def test_current_position_mapping_independent_of_nn_future(self):
        outcomes = []
        for future in (0., 1000.):
            outputs, data = synthetic_outputs(), history_data()
            # Query 0 is at x=10; Query 2 at x=0. Their futures cross/swap.
            with torch.no_grad():
                outputs[0]['prediction'][-1][:, 0, 0] = 10.
                outputs[2]['prediction'][-1][:, 0, :, 0, 0] = -10. + future
                outputs[2]['prediction'][-1][:, 2, :, 0, 0] = 10. - future
            history = [(i*.5, np.array([[0., 0.], [10., 0.]]), np.array([0, 0])) for i in range(3)]
            real_associate = associate
            pairs, targets = [], []
            def record(xy, labels, tracks, *args):
                matched = real_associate(xy, labels, tracks, *args)
                pairs.append((xy.copy(), matched))
                return matched
            from projects.mmdet3d_plugin.models.dynamics.dynamics_consistency import compute_dynamics_loss
            def record_loss(pred, prior, mask):
                targets.append(prior.clone())
                return compute_dynamics_loss(pred, prior, mask)
            with patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.HistoricalPerception') as teacher, \
                 patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.associate', side_effect=record), \
                 patch('projects.mmdet3d_plugin.models.dynamics.dynamics_consistency.compute_dynamics_loss', side_effect=record_loss):
                teacher.return_value.return_value = history
                dynamics_loss(harness(outputs), outputs, data)
            self.assertEqual(pairs[0][1], [(0, 1), (1, 0)])
            outcomes.append((pairs, targets[0]))
        for a, b in zip(outcomes[0][0], outcomes[1][0]):
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1], b[1])
        torch.testing.assert_close(outcomes[0][1], outcomes[1][1], rtol=0, atol=0)

    def test_short_occlusion_reuses_track_but_current_missing_is_inactive(self):
        bank = AgentTrackBank(dict(ekf_max_track_age=1.5))
        for i in range(3):
            bank.step(i*.5, np.array([[0., 0.]]), np.array([0]))
        old = bank.tracks[0]
        bank.step(1.5, np.empty((0, 2)), np.array([]))
        self.assertEqual(bank.reliable_tracks(), [])
        bank.step(2., np.array([[.01, 0.]]), np.array([0]))
        self.assertIs(bank.tracks[0], old)
        self.assertEqual(len(bank.tracks[0]['observations']), 4)
        self.assertEqual(len(bank.reliable_tracks()), 1)

    def test_current_coordinate_transform_and_missing_reappearing_track(self):
        transform = np.eye(4); transform[0, 3] = 10.
        xy = to_world(torch.tensor([[.2, 0.]]), transform).numpy()
        track = dict(ekf=CTRVEKF().initialize([10., 0.]), label=0)
        self.assertEqual(associate(xy, np.array([0]), [track]), [(0, 0)])
        bank = AgentTrackBank(dict(ekf_max_track_age=.75))
        for i in range(3):
            bank.step(i*.5, np.array([[0., 0.]]), np.array([0]))
        old = bank.tracks[0]
        bank.step(1.5, np.empty((0, 2)), np.array([]))
        self.assertEqual(bank.reliable_tracks(), [])
        bank.step(2., np.array([[0., 0.]]), np.array([0]))
        self.assertIsNot(bank.tracks[0], old)
        self.assertEqual(len(bank.tracks[0]['observations']), 1)
        self.assertEqual(bank.reliable_tracks(), [])


class DiagnosticsFixTests(unittest.TestCase):
    def test_unequal_rank_batch_agent_denominators(self):
        a = dict(batch_size=2, entropy_triggered_samples=1, active_samples=1,
                 matched_dyn_count=1, raw_loss_dyn=2., weighted_loss_dyn=.2,
                 prior_variance_sum=3., track_observation_sum=3.)
        b = dict(batch_size=4, entropy_triggered_samples=3, active_samples=2,
                 matched_dyn_count=3, raw_loss_dyn=4., weighted_loss_dyn=.4,
                 prior_variance_sum=15., track_observation_sum=12.)
        totals = np.array(diagnostics_totals(a))+diagnostics_totals(b)
        metrics = metrics_from_totals(totals.tolist())
        self.assertAlmostEqual(metrics['entropy_trigger_rate'], 4/6)
        self.assertAlmostEqual(metrics['dyn_active_sample_ratio'], 3/6)
        self.assertEqual(metrics['raw_loss_dyn'], 3.5)
        self.assertEqual(metrics['raw_loss_dyn_rank_mean'], 3.)
        self.assertAlmostEqual(metrics['raw_loss_dyn_batch_mean'], 20/6)
        self.assertEqual(metrics['ekf_prior_position_variance'], 4.5)
        self.assertEqual(metrics['track_observation_count'], 3.75)

    def test_hook_detached_rng_loss_and_grad_unchanged(self):
        model = torch.nn.Linear(2, 1)
        model.evo_cfg = dict(enable_dynamics_loss=True, lambda_dyn=.1)
        model.dynamics_diagnostics = dict(batch_size=2, matched_dyn_count=0)
        model(torch.ones(1, 2)).sum().backward()
        grads = [p.grad.clone() for p in model.parameters()]
        rng = torch.get_rng_state().clone()
        runner = SimpleNamespace(model=model, log_buffer=LogBuffer(), outputs={'loss': torch.tensor(3.)})
        hook = DynamicsDiagnosticsHook()
        hook.after_train_iter(runner)
        for p, g in zip(model.parameters(), grads):
            torch.testing.assert_close(p.grad, g, rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(float(runner.outputs['loss']), 3.)
        self.assertTrue(all(np.isfinite(v) for v in hook.latest_metrics.values()))
        self.assertEqual(hook.latest_metrics['raw_loss_dyn'], 0.)
        model.evo_cfg['lambda_dyn'] = 0.
        with patch('projects.mmdet3d_plugin.models.dynamics.diagnostics.reduce_dynamics_diagnostics', side_effect=AssertionError('disabled')):
            hook.after_train_iter(runner)

    def test_only_early_stopping_removed_shared_budget(self):
        import mmcv
        old = mmcv.Config.fromfile(str(ROOT/'projects/configs/sparsedrive_small_stage2.py'))
        a = mmcv.Config.fromfile(str(ROOT/'projects/configs/sparsedrive_small_stage2_ctrv_e0.py'))
        b = mmcv.Config.fromfile(str(ROOT/'projects/configs/sparsedrive_small_stage2_ctrv_v1.py'))
        expected = [h for h in old.custom_hooks if h['type'] != 'WindowLossStopHook']
        self.assertEqual([h for h in a.custom_hooks if h['type'] != 'DynamicsDiagnosticsHook'], expected)
        self.assertEqual(a.custom_hooks, b.custom_hooks)
        for key in ('runner', 'seed', 'lr_config', 'optimizer', 'load_from', 'num_gpus', 'batch_size'):
            self.assertEqual(a[key], b[key])
        self.assertEqual(a.seed, 3407)
        self.assertEqual(a.model.evo_cfg.loss_weight, b.model.evo_cfg.loss_weight)
        self.assertEqual(a.model.head, old.model.head)
        self.assertEqual(b.model.head, old.model.head)

    def test_filter_preserves_unrelated_parent_hooks(self):
        import mmcv
        source = (ROOT/'projects/configs/sparsedrive_small_stage2_ctrv_e0.py').read_text()
        parent = mmcv.Config(dict(custom_hooks=[dict(type='WindowLossStopHook'),
                                               dict(type='OtherRequiredHook', priority='HIGH')]))
        namespace = {}
        with patch.object(mmcv.Config, 'fromfile', return_value=parent):
            exec(compile(source, 'comparison_config', 'exec'), namespace)
        self.assertEqual(namespace['custom_hooks'][0], dict(type='OtherRequiredHook', priority='HIGH'))
        self.assertEqual(namespace['custom_hooks'][1]['type'], 'DynamicsDiagnosticsHook')

    def test_sampler_sequence_same_seed_and_rank(self):
        import itertools
        from projects.mmdet3d_plugin.datasets.samplers.group_in_batch_sampler import GroupInBatchSampler
        class Dataset:
            flag = np.repeat(np.arange(8), 3)
            keep_consistent_seq_aug = True
            def __len__(self): return len(self.flag)
            def get_augmentation(self): return {'dummy_aug': float(np.random.uniform())}
        for rank in (0, 1):
            sampled = []
            for experiment in ('E0', 'E1'):
                np.random.seed(3407)
                sampler = GroupInBatchSampler(Dataset(), batch_size=2, world_size=2, rank=rank, seed=3407)
                sampled.append(list(itertools.islice(iter(sampler), 10)))
            self.assertEqual(sampled[0], sampled[1])

    def test_gradient_conflict_preserves_gradients_and_normal_step(self):
        def model():
            net = torch.nn.Module()
            net.img_backbone = torch.nn.Linear(2, 1)
            net.img_neck = torch.nn.Linear(1, 1)
            net.head = torch.nn.Module()
            net.head.det_head = torch.nn.Linear(1, 1)
            net.head.motion_plan_head = torch.nn.Linear(1, 1)
            return net
        base = model()
        models = [copy.deepcopy(base), copy.deepcopy(base)]
        results = []
        for i, net in enumerate(models):
            x = net.head.motion_plan_head(net.head.det_head(net.img_neck(net.img_backbone(torch.ones(1, 2)))))
            evo, dyn = x.sum(), -x.sum()
            optimizer = torch.optim.AdamW(net.parameters(), lr=.001)
            # Existing .grad buffers must survive the optional analysis unchanged.
            for p in net.parameters(): p.grad = torch.ones_like(p)
            if i:
                rng = torch.get_rng_state().clone()
                metrics = gradient_conflict(net, evo, dyn, 1., .1)
                torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                for group in metrics['groups'].values():
                    for kind in ('raw', 'weighted'):
                        product = group[kind]['evo_norm']*group[kind]['dyn_norm']
                        self.assertAlmostEqual(group[kind]['cosine'], -product/(product+1e-12), places=12)
                for p in net.parameters(): torch.testing.assert_close(p.grad, torch.ones_like(p))
            optimizer.zero_grad(set_to_none=True)
            (evo + .1*dyn).backward()
            optimizer.step()
            results.append([p.detach().clone() for p in net.parameters()])
        for a, b in zip(*results): torch.testing.assert_close(a, b, rtol=0, atol=0)
        metrics = gradient_conflict(base, torch.tensor(0.), None, 0., .1)
        self.assertTrue(all(g['raw']['cosine'] is None for g in metrics['groups'].values()))

if __name__ == '__main__':
    unittest.main(verbosity=2)
