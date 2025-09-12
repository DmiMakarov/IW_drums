"""Constant acceleration Kalman filter."""
import numpy as np


class ConstantAccelerationKalman:
    """Constant acceleration Kalman filter."""

    def __init__(self, dt: float = 0.01, process_var: float = 30.0, meas_var: float = 25.0) -> None:
        """Initialize the Kalman filter."""
        #State is: [x, y, vx, vy, ax, ay]
        self.dt = dt
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.Q = np.eye(6, dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * meas_var
        # Process noise scaled for acceleration uncertainty
        q = process_var
        self.Q[4,4] = q
        self.Q[5,5] = q
        self.update_F_H()

    def update_F_H(self) -> None:  # noqa: N802
        """Update the F and H matrices."""
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

    def predict(self) -> np.ndarray:
        """Predict the state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray, r_scale: float = 1.0) -> np.ndarray:
        """Update the state."""
        # z is: [x, y]
        r = self.R * float(r_scale)
        y = z.reshape(2,1).astype(np.float32) - (self.H @ self.x)
        s = self.H @ self.P @ self.H.T + r
        k = self.P @ self.H.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        i = np.eye(6, dtype=np.float32)
        self.P = (i - k @ self.H) @ self.P
        return self.x.copy()

    def set_state(self, x: float, y: float) -> None:
        """Set the state."""
        self.x[:] = 0
        self.x[0,0] = x
        self.x[1,0] = y
        self.P = np.eye(6, dtype=np.float32) * 50.0

    def get_position(self) -> tuple[float, float]:
        """Get the position."""
        return float(self.x[0,0]), float(self.x[1,0])

    def get_velocity(self) -> tuple[float, float]:
        """Get the velocity."""
        return float(self.x[2,0]), float(self.x[3,0])
