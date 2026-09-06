from abc import ABC, abstractmethod
from typing import List

from core.events import Event
from core.match_state import MatchState
from vision.models import FrameTracks


class BaseDetector(ABC):
    """
    Base class for all football event detectors.
    """

    @abstractmethod
    def detect(
        self,
        frame: FrameTracks,
        state: MatchState
    ) -> List[Event]:
        pass