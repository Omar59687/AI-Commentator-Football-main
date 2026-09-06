from pathlib import Path
import cv2

from vision.detector import Detector
from vision.tracker import Tracker


class VisionPipeline:

    def __init__(self, model_path: str):
        self.detector = Detector(model_path)
        self.tracker = Tracker()

    def read_video(self, video_path: str):

        cap = cv2.VideoCapture(video_path)

        frames = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            frames.append(frame)

        cap.release()

        return frames

    def run(self, video_path, max_frames=None):

        frames = self.read_video(video_path)

        print(f"Loaded {len(frames)} frames.")

        if max_frames is not None:
            frames = frames[:max_frames]

        detections = self.detector.detect(frames)

        tracks = self.tracker.track(detections)

        return {
            "frames": frames,
            "tracks": tracks
        }