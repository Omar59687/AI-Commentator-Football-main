from analyzer.possession import PossessionDetector


class MatchObserver:

    def __init__(self):
        self.possession = PossessionDetector()

    def observe(self, tracks):

        observations = []

        for frame_number in range(0, len(tracks), 10):

            frame = tracks[frame_number]

            players = [
                obj for obj in frame
                if obj["class_name"] == "player"
            ]

            referees = [
                obj for obj in frame
                if obj["class_name"] == "referee"
            ]

            balls = [
                obj for obj in frame
                if obj["class_name"] == "ball"
            ]

            owner = self.possession.detect(frame)

            owner_id = owner["track_id"] if owner else None

            observations.append({

                "frame": frame_number,

                "players": len(players),

                "referees": len(referees),

                "ball_detected": len(balls) > 0,

                "closest_player": owner_id

            })

        return observations