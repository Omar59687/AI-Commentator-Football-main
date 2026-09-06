from dataclasses import dataclass
from typing import Optional


@dataclass
class Event:

    timestamp: float

    event_type: str

    player_id: Optional[int] = None

    target_player: Optional[int] = None

    team: Optional[str] = None

    description: str = ""