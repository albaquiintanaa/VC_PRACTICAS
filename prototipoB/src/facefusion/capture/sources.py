"""Fuentes independientes con cierre explícito y reloj de reproducción."""

from contextlib import AbstractContextManager
from dataclasses import dataclass
import math
from pathlib import Path
import sys

import cv2
import numpy as np


class CameraSource(AbstractContextManager):
    def __init__(self, index=0):
        self.index = index
        self.frames_read = 0
        self.closed = False
        self.capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        if not self.capture.isOpened():
            self.close()
            raise RuntimeError(f"No se pudo abrir la webcam {index}.")
        self.backend = self.capture.getBackendName()
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def read(self):
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("La webcam no entrega fotogramas validos.")
        self.frames_read += 1
        return frame

    def close(self):
        self.capture.release()
        self.closed = not self.capture.isOpened()

    def __exit__(self, *args):
        self.close()


@dataclass(frozen=True)
class VideoFrame:
    image: np.ndarray
    index: int
    loop: int
    media_seconds: float

    @property
    def key(self):
        return self.loop, self.index


class TimedVideoSource(AbstractContextManager):
    """Selecciona por tiempo; descarta frames atrasados y repite el clip."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.closed = False
        self.rewinds = 0
        self.skipped_frames = 0
        self.frames_read = 0
        self._sample = None
        self._next_index = 0
        self._last_elapsed = -1.0
        if not self.path.is_file():
            raise ValueError(f"Video inexistente: {path}")
        self.capture = cv2.VideoCapture(str(self.path))
        try:
            if not self.capture.isOpened():
                raise RuntimeError("No se pudo abrir el video.")
            self.fps = self.capture.get(cv2.CAP_PROP_FPS)
            count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if not math.isfinite(self.fps) or self.fps <= 0 or not math.isfinite(count) or count < 1:
                raise ValueError("El video debe declarar FPS y numero de fotogramas validos.")
            self.frame_count = int(count)
            self.duration = self.frame_count / self.fps
        except Exception:
            self.close()
            raise

    def frame_at(self, elapsed: float) -> VideoFrame:
        if not math.isfinite(elapsed) or elapsed < 0 or elapsed < self._last_elapsed:
            raise ValueError("El reloj del video debe ser finito, positivo y monotono.")
        self._last_elapsed = elapsed
        global_index = math.floor(elapsed * self.fps)
        loop, index = divmod(global_index, self.frame_count)
        if self._sample is not None and self._sample.key == (loop, index):
            return self._sample
        if self._sample is not None:
            previous_global = self._sample.loop * self.frame_count + self._sample.index
            self.skipped_frames += max(0, global_index - previous_global - 1)
            if loop != self._sample.loop:
                if not self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0):
                    raise RuntimeError("No se pudo reiniciar el video.")
                self._next_index = 0
                self.rewinds += 1
        # No se interpreta un salto del reloj como una repetición lenta del archivo.
        while self._next_index < index:
            if not self.capture.grab():
                raise RuntimeError("El video termina antes de lo declarado.")
            self._next_index += 1
        ok, image = self.capture.read()
        if not ok or image is None:
            raise RuntimeError("No se pudo decodificar el fotograma del video.")
        self._next_index = index + 1
        self.frames_read += 1
        self._sample = VideoFrame(image, index, loop, global_index / self.fps)
        return self._sample

    def close(self):
        self.capture.release()
        self.closed = not self.capture.isOpened()

    def __exit__(self, *args):
        self.close()
