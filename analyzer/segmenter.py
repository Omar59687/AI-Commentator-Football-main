class PossessionSegmenter:

    def build(self, timeline):

        if not timeline:
            return []

        segments = []

        current_owner = timeline[0]["owner"]

        start_frame = timeline[0]["frame"]

        for item in timeline[1:]:

            if item["owner"] != current_owner:

                segments.append(
                    {
                        "player": current_owner,
                        "start": start_frame,
                        "end": item["frame"] - 1
                    }
                )

                current_owner = item["owner"]

                start_frame = item["frame"]

        segments.append(
            {
                "player": current_owner,
                "start": start_frame,
                "end": timeline[-1]["frame"]
            }
        )

        return segments