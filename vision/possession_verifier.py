import math
import os
from pathlib import Path


CONFIDENCE_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def _normalise_team(value):

    team = str(value or "unclear").strip().lower()

    if team not in {"white", "green", "unclear"}:
        return "unclear"

    return team


def _normalise_confidence(value):

    confidence = str(value or "low").strip().lower()

    if confidence not in CONFIDENCE_RANK:
        return "low"

    return confidence


def reconcile_possession(
    verified_events,
    local_signals,
):

    signals_by_id = {
        int(signal["event_id"]): signal
        for signal in local_signals
    }

    reconciled = []

    for event in verified_events:

        event = dict(event)
        event_id = int(event["event_id"])

        gemini_team = _normalise_team(
            event.get("possession_team")
        )

        gemini_confidence = _normalise_confidence(
            event.get("possession_confidence")
        )

        local = signals_by_id.get(event_id, {})

        local_team = _normalise_team(
            local.get("possession_team")
        )

        local_confidence = _normalise_confidence(
            local.get("confidence")
        )

        teams_agree = (
            gemini_team != "unclear"
            and gemini_team == local_team
        )

        independent_agreement = (
            teams_agree
            and CONFIDENCE_RANK[gemini_confidence] >= 1
            and CONFIDENCE_RANK[local_confidence] >= 1
        )

        if independent_agreement:
            possession_team = gemini_team

            if (
                gemini_confidence == "high"
                and local_confidence == "high"
            ):
                possession_confidence = "high"
            else:
                possession_confidence = "medium"

            verification_source = "gemini+local-agreement"

        elif (
            local_team != "unclear"
            and local_confidence == "high"
            and (
                gemini_team == "unclear"
                or gemini_confidence == "low"
            )
        ):
            possession_team = local_team
            possession_confidence = "medium"
            verification_source = "local-only"

        elif (
            gemini_team != "unclear"
            and gemini_confidence == "high"
            and local_team == "unclear"
        ):
            possession_team = gemini_team
            possession_confidence = "medium"
            verification_source = "gemini-only"

        else:
            possession_team = "unclear"
            possession_confidence = "low"

            if (
                gemini_team != "unclear"
                and local_team != "unclear"
                and gemini_team != local_team
            ):
                verification_source = "conflict"
            else:
                verification_source = "insufficient-evidence"

        event["gemini_possession_team"] = gemini_team
        event["gemini_possession_confidence"] = (
            gemini_confidence
        )
        event["local_possession_team"] = local_team
        event["local_possession_confidence"] = (
            local_confidence
        )
        event["local_evidence"] = str(
            local.get(
                "evidence",
                "Local possession evidence unavailable.",
            )
        )
        event["possession_team"] = possession_team
        event["possession_confidence"] = (
            possession_confidence
        )
        event["verification_source"] = verification_source
        event["safe_to_name_team"] = (
            independent_agreement
            and possession_confidence in {"medium", "high"}
        )

        reconciled.append(event)

    return reconciled


def mark_team_mentions(
    events,
    max_mentions=4,
):

    marked = []
    last_verified_team = None
    mention_count = 0

    for event in events:

        event = dict(event)
        team = _normalise_team(
            event.get("possession_team")
        )

        safe = bool(
            event.get("safe_to_name_team", False)
        )

        changed = (
            team != "unclear"
            and team != last_verified_team
        )

        mention_team = (
            safe
            and changed
            and mention_count < int(max_mentions)
        )

        event["mention_team"] = mention_team

        if mention_team:
            mention_count += 1

        if safe and team != "unclear":
            last_verified_team = team

        marked.append(event)

    return marked


class LocalPossessionVerifier:

    def __init__(
        self,
        model_path=None,
        image_size=1280,
        confidence=0.12,
    ):

        default_model_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "best.pt"
        )

        self.model_path = model_path or os.getenv(
            "FOOTBALL_MODEL_PATH",
            str(default_model_path),
        )

        self.image_size = int(image_size)
        self.confidence = float(confidence)
        self.model = None

    def _load_model(self):

        if self.model is None:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

        return self.model

    def _box_values(self, box):

        coordinates = (
            box.xyxy[0]
            .detach()
            .cpu()
            .tolist()
        )

        return [float(value) for value in coordinates]

    def _shirt_color(self, frame, bbox):

        import cv2
        import numpy as np

        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)

        crop_x1 = max(0, round(x1 + width * 0.18))
        crop_x2 = min(
            frame_width,
            round(x2 - width * 0.18),
        )
        crop_y1 = max(0, round(y1 + height * 0.12))
        crop_y2 = min(
            frame_height,
            round(y1 + height * 0.55),
        )

        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return "unclear", 0.0

        shirt = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if shirt.size == 0:
            return "unclear", 0.0

        hsv = cv2.cvtColor(
            shirt,
            cv2.COLOR_BGR2HSV,
        )

        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        green_pixels = (
            (hue >= 30)
            & (hue <= 95)
            & (saturation >= 65)
            & (value >= 70)
        )

        white_pixels = (
            (saturation <= 65)
            & (value >= 145)
        )

        total = max(1, int(np.prod(hue.shape)))

        green_score = float(
            np.count_nonzero(green_pixels)
        ) / total

        white_score = float(
            np.count_nonzero(white_pixels)
        ) / total

        if (
            green_score >= 0.16
            and green_score >= white_score * 1.30
        ):
            return "green", green_score

        if (
            white_score >= 0.18
            and white_score >= green_score * 1.25
        ):
            return "white", white_score

        return "unclear", max(green_score, white_score)

    def _frame_vote(self, frame, result):

        players = []
        balls = []

        for box in result.boxes:

            class_id = int(box.cls[0].item())
            class_name = str(
                result.names[class_id]
            ).lower()
            confidence = float(box.conf[0].item())

            item = {
                "bbox": self._box_values(box),
                "confidence": confidence,
            }

            if class_name == "player":
                players.append(item)
            elif class_name == "ball":
                balls.append(item)

        if not players or not balls:
            return None

        ball = max(
            balls,
            key=lambda item: item["confidence"],
        )

        bx1, by1, bx2, by2 = ball["bbox"]
        ball_x = (bx1 + bx2) / 2.0
        ball_y = (by1 + by2) / 2.0

        closest = None
        closest_ratio = float("inf")

        for player in players:

            x1, y1, x2, y2 = player["bbox"]
            feet_x = (x1 + x2) / 2.0
            feet_y = y2
            player_height = max(1.0, y2 - y1)

            distance = math.hypot(
                feet_x - ball_x,
                feet_y - ball_y,
            )

            distance_ratio = distance / player_height

            if distance_ratio < closest_ratio:
                closest_ratio = distance_ratio
                closest = player

        if closest is None or closest_ratio > 1.25:
            return None

        team, shirt_score = self._shirt_color(
            frame,
            closest["bbox"],
        )

        if team == "unclear":
            return None

        return {
            "team": team,
            "shirt_score": shirt_score,
            "distance_ratio": closest_ratio,
            "ball_confidence": ball["confidence"],
            "player_confidence": closest["confidence"],
        }

    def verify(self, evidence):

        frames = []
        frame_event_ids = []

        for item in evidence:
            for frame in item["frames"]:
                frames.append(frame)
                frame_event_ids.append(
                    int(item["event_id"])
                )

        if not frames:
            return []

        model = self._load_model()

        results = model.predict(
            frames,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        votes_by_event = {}

        for frame, event_id, result in zip(
            frames,
            frame_event_ids,
            results,
        ):

            vote = self._frame_vote(frame, result)

            if vote is not None:
                votes_by_event.setdefault(
                    event_id,
                    [],
                ).append(vote)

        signals = []

        for item in evidence:

            event_id = int(item["event_id"])
            votes = votes_by_event.get(event_id, [])

            green_votes = [
                vote
                for vote in votes
                if vote["team"] == "green"
            ]
            white_votes = [
                vote
                for vote in votes
                if vote["team"] == "white"
            ]

            if len(green_votes) > len(white_votes):
                team = "green"
                winning_votes = green_votes
            elif len(white_votes) > len(green_votes):
                team = "white"
                winning_votes = white_votes
            else:
                team = "unclear"
                winning_votes = []

            total_votes = len(votes)
            winner_count = len(winning_votes)
            vote_share = (
                winner_count / total_votes
                if total_votes
                else 0.0
            )

            if (
                team != "unclear"
                and winner_count >= 3
                and vote_share >= 0.75
            ):
                confidence = "high"
            elif (
                team != "unclear"
                and winner_count >= 2
                and vote_share >= 0.60
            ):
                confidence = "medium"
            else:
                team = "unclear"
                confidence = "low"

            signals.append({
                "event_id": event_id,
                "possession_team": team,
                "confidence": confidence,
                "evidence": (
                    f"Local detector votes: green={len(green_votes)}, "
                    f"white={len(white_votes)}, usable={total_votes}."
                ),
            })

        return signals
