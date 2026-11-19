"""Un estado de Face Landmarker en CPU por cada flujo de imágenes."""

from contextlib import AbstractContextManager
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class FaceTracker(AbstractContextManager):
    def __init__(self, model: Path, name: str):
        self.model = Path(model)
        self.name = name
        self.last_timestamp = -1
        self.observations = 0
        self.face_frames = 0
        self.invalid_results = 0
        self.timestamp_errors = 0
        self.resets = 0
        self.closed = False
        self._task = self._create()

    def _create(self):
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model), delegate=python.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def reset(self):
        self._task.close()
        self._task = None
        self._task = self._create()
        self.resets += 1

    def detect(self, frame, timestamp_ms: int):
        if timestamp_ms <= self.last_timestamp:
            self.timestamp_errors += 1
            raise ValueError(f"Timestamp no creciente en {self.name}.")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self._task.detect_for_video(image, timestamp_ms)
        self.last_timestamp = timestamp_ms
        self.observations += 1
        if not result.face_landmarks:
            return None
        points = np.array([(point.x, point.y, point.z) for point in result.face_landmarks[0]], dtype=np.float32)
        if points.shape != (478, 3) or not np.isfinite(points).all():
            self.invalid_results += 1
            return None
        self.face_frames += 1
        return points

    def summary(self):
        return {"source": self.name, "observations": self.observations, "face_frames": self.face_frames,
                "face_ratio": self.face_frames / self.observations if self.observations else 0,
                "invalid_results": self.invalid_results, "timestamp_errors": self.timestamp_errors,
                "last_timestamp_ms": self.last_timestamp, "resets": self.resets, "closed": self.closed}

    def close(self):
        if self._task is not None:
            self._task.close()
            self._task = None
        self.closed = True

    def __exit__(self, *args):
        self.close()
