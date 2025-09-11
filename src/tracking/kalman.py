import numpy as np

class ConstantAccelerationKalman:
    def __init__(self, dt: float = 0.01, process_var: float = 30.0, meas_var: float = 25.0):
        # State: [x, y, vx, vy, ax, ay]
        self.dt = dt
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.Q = np.eye(6, dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * meas_var
        # Process noise scaled for acceleration uncertainty
        q = process_var
        self.Q[4,4] = q
        self.Q[5,5] = q
        self._update_F_H()

    def _update_F_H(self):
        dt = self.dt
        self.F = np.eye(6, dtype=np.float32)
        self.F[0,2] = dt
        self.F[1,3] = dt
        self.F[0,4] = 0.5 * dt * dt
        self.F[1,5] = 0.5 * dt * dt
        self.F[2,4] = dt
        self.F[3,5] = dt
        self.H = np.zeros((2,6), dtype=np.float32)
        self.H[0,0] = 1.0
        self.H[1,1] = 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray, r_scale: float = 1.0):
        # z: [x, y]
        R = self.R * float(r_scale)
        y = z.reshape(2,1).astype(np.float32) - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    def set_state(self, x: float, y: float):
        self.x[:] = 0
        self.x[0,0] = x
        self.x[1,0] = y
        self.P = np.eye(6, dtype=np.float32) * 50.0

    def get_position(self):
        return float(self.x[0,0]), float(self.x[1,0])

    def get_velocity(self):
        return float(self.x[2,0]), float(self.x[3,0])
