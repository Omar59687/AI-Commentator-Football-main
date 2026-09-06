import math

from ultralytics import YOLO


class Detector:
    """
    Handles object detection using a YOLO model.
    """

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, frames, confidence: float = 0.25, batch_size: int = 16):
        """
        Run YOLO on a list of frames.

        Args:
            frames: List of OpenCV images.
            confidence: Minimum confidence score.
            batch_size: Number of frames processed together.

        Returns:
            List of YOLO Results objects.
        """

        detections = []

        for i in range(0, len(frames), batch_size):

            print(
                f"Processing batch {i // batch_size + 1}/"
                f"{math.ceil(len(frames) / batch_size)}"
            )
            
            batch = frames[i:i + batch_size]

            results = self.model.predict(
                batch,
                conf=confidence,
                verbose=False
            )

            detections.extend(results)

        return detections