from dataclasses import dataclass
from typing import Optional


@dataclass
class FrameState:
    frame_number: int
    timestamp: float

    ball_owner: Optional[int]

    ball_x: float
    ball_y: float