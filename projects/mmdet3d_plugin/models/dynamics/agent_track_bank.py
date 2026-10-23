"""Ephemeral scene-local position/class association; query indices are never IDs."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment
from .ctrv_ekf import CTRVEKF, wrap_angle


def associate(positions, labels, tracks, gate=9.21, max_distance=6.):
    """Gated Mahalanobis Hungarian assignment, one observation per track."""
    if not len(positions) or not tracks:
        return []
    cost = np.full((len(positions), len(tracks)), 1e12)
    for j, track in enumerate(tracks):
        delta = positions - track['ekf'].x[:2]
        s = track['ekf'].P[:2, :2] + track['ekf'].R
        d2 = np.einsum('ni,in->n', delta, np.linalg.solve(s, delta.T))
        valid = (labels == track['label']) & (d2 <= gate)
        valid &= np.linalg.norm(delta, axis=1) <= max_distance
        cost[valid, j] = d2[valid]
    rows, cols = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1e12]


class AgentTrackBank:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = []
        self.time = None
        self.matched_count = self.observation_count = 0

    def _new_filter(self):
        return CTRVEKF(self.cfg.get('ekf_process_noise', (.1, .1, 1., .1, .1)),
                       self.cfg.get('ekf_measurement_noise', (.5, .5)),
                       self.cfg.get('ekf_initial_covariance', (1., 1., 25., 10., 10.)))

    def step(self, timestamp, positions, labels):
        timestamp = float(timestamp)
        if self.time is not None and timestamp <= self.time:
            raise ValueError('Historical observations must be strictly chronological')
        max_age = self.cfg.get('ekf_max_track_age', 1.5)  # seconds
        self.tracks = [t for t in self.tracks if timestamp - t['last_seen'] <= max_age]
        healthy = []
        for t in self.tracks:
            try:
                t['ekf'].predict(timestamp - self.time)
                healthy.append(t)
            except (FloatingPointError, np.linalg.LinAlgError):
                pass
        self.tracks = healthy
        pairs = associate(positions, labels, self.tracks,
                          self.cfg.get('ekf_association_gate', 9.21),
                          self.cfg.get('ekf_max_association_distance', 6.))
        matched = set()
        for i, j in pairs:
            t = self.tracks[j]
            t['ekf'].update(positions[i])
            t['observations'].append((timestamp, positions[i].copy()))
            t['last_seen'] = timestamp
            # Bootstrap from measured displacement, never from box yaw/GT velocity.
            history = t['observations']
            if len(history) <= 3:
                dt = history[-1][0] - history[-2][0]
                delta = history[-1][1] - history[-2][1]
                speed = np.linalg.norm(delta)/dt
                if speed >= self.cfg.get('ekf_min_heading_speed', .5):
                    heading = np.arctan2(delta[1], delta[0])
                    omega = 0.
                    if len(history) == 3:
                        delta0 = history[-2][1] - history[-3][1]
                        dt0 = history[-2][0] - history[-3][0]
                        if np.linalg.norm(delta0)/dt0 >= self.cfg.get('ekf_min_heading_speed', .5):
                            omega = wrap_angle(heading - np.arctan2(delta0[1], delta0[0])) / ((dt+dt0)/2)
                    t['ekf'].x[2:] = [speed, wrap_angle(heading + omega*dt/2), omega]
                else:
                    t['ekf'].x[2:] = [0., 0., 0.]
            matched.add(i)
        self.matched_count += len(pairs)
        self.observation_count += len(positions)
        for i, pos in enumerate(positions):
            if i not in matched:
                self.tracks.append(dict(ekf=self._new_filter().initialize(pos),
                                        label=int(labels[i]), last_seen=timestamp,
                                        observations=[(timestamp, pos.copy())]))
        self.time = timestamp

    def reliable_tracks(self):
        minimum = max(3, self.cfg.get('ekf_min_observations', 3))
        max_var = self.cfg.get('ekf_max_position_variance', 25.)
        return [t for t in self.tracks if len(t['observations']) >= minimum
                and t['last_seen'] == self.time
                and np.linalg.eigvalsh(t['ekf'].P[:2, :2]).max() <= max_var]

    def predict_prior(self, track, next_timestamp):
        # Predict a copy: no measurement at t+1 and no persistent bank across samples.
        ekf = copy.deepcopy(track['ekf'])
        dt = float(next_timestamp) - self.time
        if dt <= 0 or dt > self.cfg.get('ekf_max_prediction_dt', 1.5):
            return None
        try:
            position = ekf.predict(dt)
            if np.linalg.eigvalsh(ekf.get_prior_covariance()).max() > self.cfg.get('ekf_max_position_variance', 25.):
                return None
            return position
        except (FloatingPointError, np.linalg.LinAlgError):
            return None
