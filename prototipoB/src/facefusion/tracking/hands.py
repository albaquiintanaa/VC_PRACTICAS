"""Landmarks de hasta dos manos en la webcam original, sin reflejar."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@dataclass(frozen=True)
class HandObservation:
    points: np.ndarray
    label: str
    confidence: float


class HandTracker(AbstractContextManager):
    def __init__(self, model: Path):
        options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO, num_hands=2,
        )
        self._task = vision.HandLandmarker.create_from_options(options)
        self.closed = False
        self.last_timestamp = -1
        self.frames = 0

    def detect(self, frame, timestamp_ms):
        if timestamp_ms <= self.last_timestamp:
            raise ValueError("Timestamp de manos no creciente.")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._task.detect_for_video(image, timestamp_ms)
        self.last_timestamp = timestamp_ms
        self.frames += 1
        hands = []
        for landmarks, categories in zip(result.hand_landmarks, result.handedness):
            if not categories:
                continue
            points = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
            if points.shape == (21, 3) and np.isfinite(points).all():
                category = categories[0]
                hands.append(HandObservation(points, category.category_name, float(category.score)))
        return hands

    def close(self):
        if not self.closed:
            self._task.close()
            self.closed = True

    def __exit__(self, *args):
        self.close()
