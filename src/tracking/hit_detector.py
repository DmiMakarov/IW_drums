"""Hit detector."""
from dataclasses import dataclass


@dataclass
class HitConfig:
    """Hit configuration."""

    min_downward_speed: float = 8.0  # px/frame
    min_accel: float = 4.0           # px/frame^2
    refractory_frames: int = 10      # ~100 ms at 100 fps

class HitDetector:
    """Hit detector."""

    def __init__(self, cfg: HitConfig) -> None:
        """Initialize the hit detector."""
        self.cfg = cfg
        self.prev_vy = 0.0
        self.prev2_vy = 0.0
        self.refractory = 0

    def update(self, vy: float) -> bool:
        """Return True if a hit is detected this frame using velocity sign change with minimum downward speed and acceleration constraint.

        vy: positive is downward.
        """  # noqa: E501
        hit = False
        if self.refractory > 0:
            self.refractory -= 1
        else:
            # Approximate accel as finite difference
            ay = vy - self.prev_vy
            # Peak detection: previous velocity was fast downward, now non-positive (upward or stop)
            if self.prev_vy >= self.cfg.min_downward_speed and vy <= 0.0 and ay <= -self.cfg.min_accel:
                hit = True
                self.refractory = self.cfg.refractory_frames
        self.prev2_vy = self.prev_vy
        self.prev_vy = vy
        return hit
