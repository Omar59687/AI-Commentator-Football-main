from typing import Optional

from analyzer.geometry import distance
from vision.models import TrackObject

from analyzer.geometry import distance


class PossessionDetector:

    MAX_DISTANCE = 70

    def detect(self, frame_tracks):

        players = [
            obj for obj in frame_tracks
            if obj["class_name"] == "player"
        ]

        balls = [
            obj for obj in frame_tracks
            if obj["class_name"] == "ball"
        ]

        if not balls:
            return None

        ball = balls[0]

        closest = None
        best_distance = float("inf")

        for player in players:

            d = distance(
                player["bbox"],
                ball["bbox"]
            )

            if d < best_distance:

                best_distance = d
                closest = player

        if best_distance > self.MAX_DISTANCE:
            return None

        return closest