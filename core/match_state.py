from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchState:
    current_possession: Optional[int] = None
    previous_possession: Optional[int] = None

    current_frame: int = 0