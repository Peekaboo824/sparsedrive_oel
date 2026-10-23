"""Position-only CTRV EKF. All filtering is detached CPU float64 arithmetic."""
import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def covariance(value, size):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim == 0:
        matrix = np.eye(size) * matrix
    elif matrix.ndim == 1:
        matrix = np.diag(matrix)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError('Invalid covariance shape or non-finite covariance')
    if not np.allclose(matrix, matrix.T) or np.linalg.eigvalsh(matrix).min() < 0:
        raise ValueError('Covariance must be symmetric positive semidefinite')
    return matrix.copy()


def transition_and_jacobian(state, dt):
    """Stable midpoint/sinc CTRV, including the nonzero d(position)/d(omega) at 0."""
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive (seconds)')
    x = np.asarray(state, dtype=np.float64).copy()
    a = x[4] * dt / 2
    if abs(a) < 1e-4:
        s = 1 - a*a/6 + a**4/120
        ds = -a/3 + a**3/30 - a**5/840
    else:
        s = np.sin(a)/a
        ds = (a*np.cos(a) - np.sin(a))/(a*a)
    phi = x[3] + a
    c, sn = np.cos(phi), np.sin(phi)
    distance = x[2] * dt * s
    f = np.eye(5, dtype=np.float64)
    f[0, 2], f[1, 2] = dt*s*c, dt*s*sn
    f[0, 3], f[1, 3] = -distance*sn, distance*c
    f[0, 4] = x[2]*dt*dt/2 * (ds*c - s*sn)
    f[1, 4] = x[2]*dt*dt/2 * (ds*sn + s*c)
    f[3, 4] = dt
    x[:2] += distance * np.array([c, sn])
    x[3] = wrap_angle(x[3] + x[4]*dt)
    return x, f


class CTRVEKF:
    """State [world x, world y, speed, motion heading, yaw rate], SI units.

    Q is a fixed covariance per prediction, R per position observation.
    get_prior_* retain the most recent predict result even after update().
    """
    def __init__(self, process_noise=(.1, .1, 1., .1, .1),
                 measurement_noise=(.5, .5),
                 initial_covariance=(1., 1., 25., 10., 10.)):
        self.Q = covariance(process_noise, 5)
        self.R = covariance(measurement_noise, 2)
        self.initial_covariance = covariance(initial_covariance, 5)
        self.x = self.P = self.prior_x = self.prior_P = None

    def initialize(self, observation, speed=0., heading=0., yaw_rate=0.):
        self.x = np.array([*observation, speed, heading, yaw_rate], dtype=np.float64)
        if self.x.shape != (5,) or not np.isfinite(self.x).all():
            raise ValueError('Expected finite 2D position and finite motion state')
        self.x[3] = wrap_angle(self.x[3])
        self.P = self.initial_covariance.copy()
        self.prior_x = self.prior_P = None
        return self

    @staticmethod
    def _stable(matrix):
        matrix = (matrix + matrix.T)/2
        values, vectors = np.linalg.eigh(matrix)
        if not np.isfinite(values).all() or values.max() > 1e12:
            raise FloatingPointError('EKF covariance diverged')
        return (vectors * np.maximum(values, 1e-9)) @ vectors.T

    def predict(self, dt):
        if self.x is None:
            raise RuntimeError('initialize() is required')
        self.x, f = transition_and_jacobian(self.x, dt)
        self.P = self._stable(f @ self.P @ f.T + self.Q)
        self.prior_x, self.prior_P = self.x.copy(), self.P.copy()
        return self.get_prior_position()

    def update(self, observation):
        z = np.asarray(observation, dtype=np.float64)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('Expected finite 2D position observation')
        c = np.eye(2, 5)
        innovation_cov = self.P[:2, :2] + self.R
        k = np.linalg.solve(innovation_cov, (self.P @ c.T).T).T
        self.x += k @ (z - self.x[:2])
        a = np.eye(5) - k @ c
        self.P = self._stable(a @ self.P @ a.T + k @ self.R @ k.T)
        # Equivalent positive-speed representation if the unconstrained Gaussian
        # update crossed zero. Transform the covariance cross terms as well.
        if self.x[2] < 0:
            self.x[2] *= -1
            self.x[3] += np.pi
            sign = np.diag([1., 1., -1., 1., 1.])
            self.P = sign @ self.P @ sign
        self.x[3] = wrap_angle(self.x[3])
        if not np.isfinite(self.x).all():
            raise FloatingPointError('EKF state diverged')

    def get_prior_position(self):
        if self.prior_x is None:
            raise RuntimeError('predict() is required before requesting a prior')
        return self.prior_x[:2].copy()

    def get_prior_covariance(self):
        if self.prior_P is None:
            raise RuntimeError('predict() is required before requesting a prior')
        return self.prior_P[:2, :2].copy()
