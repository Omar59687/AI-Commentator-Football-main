from analyzer.possession import PossessionDetector


class EventEngine:

    def __init__(self):
        self.possession = PossessionDetector()

    def analyze(self, tracks):

        timeline = []

        events = []

        previous_owner = None

        for frame_number, frame in enumerate(tracks):

            owner = self.possession.detect(frame)

            owner_id = None

            if owner:
                owner_id = owner["track_id"]

            timeline.append({
                "frame": frame_number,
                "owner": owner_id
            })

            if owner_id != previous_owner:

                events.append({
                    "frame": frame_number,
                    "type": "possession",
                    "player": owner_id
                })

                if previous_owner is not None and owner_id is not None:

                    events.append({
                        "frame": frame_number,
                        "type": "pass",
                        "from": previous_owner,
                        "to": owner_id
                    })

            previous_owner = owner_id

        return timeline, events