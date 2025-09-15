"""Smoothing utilities for gesture tracking."""
from __future__ import annotations


class SimpleSmoother:
    """Simple position/velocity smoother using a small sliding window."""

    def __init__(self, window_size: int = 3) -> None:
        """Initialize the smoother."""
        self.positions: list[tuple[float, float]] = []
        self.velocities: list[tuple[float, float]] = []
        self.window_size = window_size

    def update(self, x: float, y: float, dt: float = 1 / 120.0) -> tuple[float, float]:
        """Update the smoother."""
        dimension = 2
        self.positions.append((x, y))
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        if len(self.positions) >= dimension:
            total_dt = max(1e-6, (len(self.positions) - 1) * dt)
            vx = (self.positions[-1][0] - self.positions[0][0]) / total_dt
            vy = (self.positions[-1][1] - self.positions[0][1]) / total_dt
            self.velocities.append((vx, vy))
            if len(self.velocities) > dimension:
                self.velocities.pop(0)
        return self.get_position()

    def get_position(self) -> tuple[float, float]:
        """Get the position."""
        if not self.positions:
            return 0.0, 0.0
        avg_x = sum(p[0] for p in self.positions) / len(self.positions)
        avg_y = sum(p[1] for p in self.positions) / len(self.positions)
        return float(avg_x), float(avg_y)

    def get_velocity(self) -> tuple[float, float]:
        """Get the velocity."""
        if not self.velocities:
            return 0.0, 0.0
        avg_vx = sum(v[0] for v in self.velocities) / len(self.velocities)
        avg_vy = sum(v[1] for v in self.velocities) / len(self.velocities)
        return float(avg_vx), float(avg_vy)



