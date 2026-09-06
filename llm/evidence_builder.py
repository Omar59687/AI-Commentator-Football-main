class EventEvidenceBuilder:

    OFFSETS = (-0.75, -0.25, 0.0, 0.5)

    def __init__(
        self,
        cell_width=960,
        jpeg_quality=82,
    ):

        self.cell_width = int(cell_width)
        self.jpeg_quality = int(jpeg_quality)

    def _sample_times(
        self,
        event,
        duration_seconds,
    ):

        start = max(
            0.0,
            float(event["start_seconds"]),
        )

        end = min(
            float(duration_seconds),
            float(event["end_seconds"]),
        )

        cue = max(
            start,
            min(end, float(event["cue_seconds"])),
        )

        latest = max(
            start,
            min(end, float(duration_seconds) - 0.04),
        )

        return [
            round(
                max(start, min(latest, cue + offset)),
                3,
            )
            for offset in self.OFFSETS
        ]

    def _read_frame(self, capture, timestamp):

        import cv2

        capture.set(
            cv2.CAP_PROP_POS_MSEC,
            float(timestamp) * 1000.0,
        )

        success, frame = capture.read()

        if not success or frame is None:
            raise RuntimeError(
                "Could not extract a visual evidence frame at "
                f"{timestamp:.3f} seconds."
            )

        return frame

    def _prepare_cell(
        self,
        frame,
        event_id,
        timestamp,
        is_cue,
    ):

        import cv2

        height, width = frame.shape[:2]

        cell_height = max(
            1,
            round(
                self.cell_width
                * height
                / max(width, 1)
            ),
        )

        interpolation = (
            cv2.INTER_AREA
            if width > self.cell_width
            else cv2.INTER_LINEAR
        )

        cell = cv2.resize(
            frame,
            (self.cell_width, cell_height),
            interpolation=interpolation,
        )

        label = (
            f"EVENT {event_id} | {timestamp:06.2f}s"
            + (" | CUE" if is_cue else "")
        )

        cv2.rectangle(
            cell,
            (0, 0),
            (min(self.cell_width, 430), 54),
            (0, 0, 0),
            thickness=-1,
        )

        cv2.putText(
            cell,
            label,
            (14, 37),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return cell

    def build(
        self,
        video_path,
        events,
        duration_seconds,
    ):

        import cv2
        import numpy as np

        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open video for evidence: {video_path}"
            )

        evidence = []

        try:
            for event in events:

                event_id = int(event["event_id"])
                cue = float(event["cue_seconds"])

                timestamps = self._sample_times(
                    event,
                    duration_seconds,
                )

                cells = []

                for timestamp in timestamps:

                    frame = self._read_frame(
                        capture,
                        timestamp,
                    )

                    cells.append(
                        self._prepare_cell(
                            frame,
                            event_id,
                            timestamp,
                            abs(timestamp - cue) < 0.02,
                        )
                    )

                sheet = np.vstack((
                    np.hstack((cells[0], cells[1])),
                    np.hstack((cells[2], cells[3])),
                ))

                success, encoded = cv2.imencode(
                    ".jpg",
                    sheet,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        self.jpeg_quality,
                    ],
                )

                if not success:
                    raise RuntimeError(
                        "Could not encode visual evidence for "
                        f"event {event_id}."
                    )

                evidence.append({
                    "event_id": event_id,
                    "cue_seconds": cue,
                    "timestamps": timestamps,
                    "frames": cells,
                    "jpeg_bytes": encoded.tobytes(),
                })

        finally:
            capture.release()

        return evidence
