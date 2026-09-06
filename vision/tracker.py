import supervision as sv


class Tracker:
    """
    Converts detections into tracked objects using ByteTrack.
    """

    def __init__(self):
        self.byte_tracker = sv.ByteTrack()

    def track(self, detections):
        """
        Track detections across video frames.

        Args:
            detections: List of YOLO Results.

        Returns:
            List of dictionaries containing tracked objects for each frame.
        """

        tracks = []

        for frame_number, detection in enumerate(detections):

            sv_detection = sv.Detections.from_ultralytics(
                detection
            )

            class_names = detection.names

            tracked = self.byte_tracker.update_with_detections(
                sv_detection
            )

            frame_tracks = []

            for item in tracked:

                bbox = item[0].tolist()

                confidence = float(item[2])

                class_id = int(item[3])

                track_id = int(item[4])

                frame_tracks.append(
                    {
                        "track_id": track_id,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "bbox": bbox,
                        "confidence": confidence,
                    }
                )

            tracks.append(frame_tracks)

        return tracks