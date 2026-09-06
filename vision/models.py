from dataclasses import dataclass
from typing import List


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class TrackObject:
    track_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass
class FrameTracks:
    frame_number: int
    players: List[TrackObject]
    referees: List[TrackObject]
    balls: List[TrackObject]


@dataclass
class VisionResult:
    fps: float
    frame_count: int
    tracks: List[FrameTracks]