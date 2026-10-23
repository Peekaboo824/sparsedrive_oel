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
            if len(t['observations']) <= 3:
                self._bootstrap_motion(t)
            matched.add(i)
        self.matched_count += len(pairs)
        self.observation_count += len(positions)
        for i, pos in enumerate(positions):
            if i not in matched:
                self.tracks.append(dict(ekf=self._new_filter().initialize(pos),
                                        label=int(labels[i]), last_seen=timestamp,
                                        observations=[(timestamp, pos.copy())]))
        self.time = timestamp

    def _bootstrap_motion(self, track):
        """Reset mean AND uncertainty for finite-difference startup estimates.

        Retain the filtered position block; discard stale position/motion and
        motion/motion correlations when replacing the motion estimate. Diagonal
        bounds use fixed measurement R, finite-difference noise propagation,
        and configurable variance floors. No adaptive Q/R or learned noise.
        """
        ekf, history = track['ekf'], track['observations']
        floors = np.asarray(self.cfg.get('ekf_bootstrap_variance_floor', (4., .25, 1.)), dtype=np.float64)
        unknown = np.asarray(self.cfg.get('ekf_bootstrap_unobserved_variance',
                                        np.diag(ekf.initial_covariance)[2:]), dtype=np.float64)
        if (floors.shape != (3,) or unknown.shape != (3,)
                or not np.isfinite(floors).all() or not np.isfinite(unknown).all()
                or (floors <= 0).any() or (unknown <= 0).any()):
            raise ValueError('Bootstrap variances must be three finite positive values')
        dt = history[-1][0] - history[-2][0]
        delta = history[-1][1] - history[-2][1]
        distance = np.linalg.norm(delta)
        speed = distance/dt
        noise = np.linalg.eigvalsh(ekf.R).max()
        variance = np.maximum(floors, unknown)
        variance[0] = max(floors[0], 2*noise/dt**2)
        motion = np.zeros(3, dtype=np.float64)
        if speed >= self.cfg.get('ekf_min_heading_speed', .5):
            heading = np.arctan2(delta[1], delta[0])
            heading_variance = 2*noise/max(distance**2, 1e-12)
            omega, omega_variance = 0., max(floors[2], unknown[2])
            if len(history) == 3:
                delta0 = history[-2][1] - history[-3][1]
                dt0 = history[-2][0] - history[-3][0]
                distance0 = np.linalg.norm(delta0)
                if distance0/dt0 >= self.cfg.get('ekf_min_heading_speed', .5):
                    midpoint_dt = (dt+dt0)/2
                    omega = wrap_angle(heading-np.arctan2(delta0[1], delta0[0]))/midpoint_dt
                    previous_heading_variance = 2*noise/max(distance0**2, 1e-12)
                    # Conservative bound includes shared-middle-observation correlation.
                    omega_variance = max(floors[2],
                        (np.sqrt(heading_variance)+np.sqrt(previous_heading_variance))**2/midpoint_dt**2)
            variance[2] = omega_variance
            variance[1] = max(floors[1],
                (np.sqrt(heading_variance)+dt/2*np.sqrt(omega_variance))**2)
            motion[:] = [speed, wrap_angle(heading+omega*dt/2), omega]
        # At low speed heading/yaw rate remain unobserved, never confident zeros.
        ekf.x[2:] = motion
        p = np.zeros((5, 5), dtype=np.float64)
        p[:2, :2] = ekf.P[:2, :2]
        p[2:, 2:] = np.diag(variance)
        ekf.P = ekf._stable(p)

    def reliable_tracks(self):
        minimum = max(3, self.cfg.get('ekf_min_observations', 3))
        max_var = self.cfg.get('ekf_max_position_variance', 25.)
        return [t for t in self.tracks if len(t['observations']) >= minimum
                and t['last_seen'] == self.time
                and np.linalg.eigvalsh(t['ekf'].P[:2, :2]).max() <= max_var]

    def predict_prior(self, track, next_timestamp, return_covariance=False):
        # Predict a copy: no measurement at t+1 and no persistent bank across samples.
        ekf = copy.deepcopy(track['ekf'])
        dt = float(next_timestamp) - self.time
        if dt <= 0 or dt > self.cfg.get('ekf_max_prediction_dt', 1.5):
            return None
        try:
            position = ekf.predict(dt)
            prior_covariance = ekf.get_prior_covariance()
            if np.linalg.eigvalsh(prior_covariance).max() > self.cfg.get('ekf_max_position_variance', 25.):
                return None
            return (position, prior_covariance) if return_covariance else position
        except (FloatingPointError, np.linalg.LinAlgError):
            return None
