"""Control temporal independiente de FPS y selección de mano anatómica."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np


def anatomical_label(hand, swap_labels=False):
    if not swap_labels:
        return hand.label
    return {"Left": "Right", "Right": "Left"}.get(hand.label, "Unknown")


def pinch_distance(hand, width, height, min_palm_px=20):
    points = np.asarray(hand.points)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    pixels = points[:, :2] * [width - 1, height - 1]
    palm = np.linalg.norm(pixels[5] - pixels[17])
    if palm < min_palm_px:
        return None
    return float(np.linalg.norm(pixels[4] - pixels[8]) / palm)


def select_hand(hands, side, width, height, swap_labels=False, confidence=.75):
    candidates = [hand for hand in hands if hand.confidence >= confidence
                  and anatomical_label(hand, swap_labels) == side
                  and pinch_distance(hand, width, height) is not None]
    # Una etiqueta duplicada o ambigua no debe hacer saltar entre manos.
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True)
class PinchCalibration:
    closed: float
    opened: float
    swap_labels: bool
    camera_index: int = 0

    def __post_init__(self):
        if not math.isfinite(self.closed) or not math.isfinite(self.opened) or self.closed < 0 or self.opened - self.closed < .35:
            raise ValueError("La calibracion necesita una apertura mayor, separada al menos 0.35 anchos de palma.")
        if not isinstance(self.swap_labels, bool):
            raise ValueError("swap_labels debe ser booleano.")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema": 1, "calibration": asdict(self)}, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path, camera_index=0):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != 1:
            raise ValueError("Version de calibracion desconocida.")
        calibration = cls(**data["calibration"])
        if calibration.camera_index != camera_index:
            raise ValueError("Calibracion de otra camara; repetir con R.")
        return calibration


class PinchController:
    def __init__(self, calibration=None, tau=.18, hold_seconds=.4, return_tau=.3, endpoint_zone=.08):
        if not all(math.isfinite(value) for value in (tau, hold_seconds, return_tau, endpoint_zone)) or tau <= 0 or hold_seconds < 0 or return_tau <= 0 or not 0 <= endpoint_zone < .5:
            raise ValueError("Parametros temporales o zona de extremos invalidos.")
        self.calibration = calibration
        self.tau, self.hold_seconds, self.return_tau = tau, hold_seconds, return_tau
        self.endpoint_zone = endpoint_zone
        self.alpha = 0.0
        self.last_update = None
        self.last_seen = None
        self.candidate_count = 0
        self.last_wrist = None
        self.reacquisitions = 0
        self.was_missing = False

    def set_calibration(self, calibration):
        self.calibration = calibration
        self.alpha = 0.0
        self.last_seen = None
        self.candidate_count = 0
        self.last_wrist = None
        self.was_missing = False

    def update(self, hands, width, height, now):
        if not math.isfinite(now) or (self.last_update is not None and now < self.last_update):
            raise ValueError("El tiempo del controlador debe ser monotono y finito.")
        dt = 0 if self.last_update is None else now - self.last_update
        self.last_update = now
        if self.calibration is None:
            return {"alpha": 0.0, "target": 0.0, "distance": None, "state": "uncalibrated", "left_selected": False}
        hand = select_hand(hands, "Left", width, height, self.calibration.swap_labels)
        if hand is not None:
            wrist = hand.points[0, :2]
            if self.last_wrist is not None and np.linalg.norm(wrist - self.last_wrist) > .25:
                self.candidate_count = 0
            self.candidate_count += 1
            self.last_wrist = wrist.copy()
        else:
            self.candidate_count = 0
            self.last_wrist = None
        selected = hand is not None and self.candidate_count >= 3
        distance = pinch_distance(hand, width, height) if selected else None
        if selected:
            if self.was_missing:
                self.reacquisitions += 1
            self.was_missing = False
            self.last_seen = now
            unit = np.clip((distance - self.calibration.closed) / (self.calibration.opened - self.calibration.closed), 0, 1)
            target = float(np.clip((unit - self.endpoint_zone) / (1 - 2 * self.endpoint_zone), 0, 1))
            state, tau = "tracking", self.tau
        else:
            self.was_missing = self.last_seen is not None
            missing_for = now - self.last_seen if self.last_seen is not None else float("inf")
            if missing_for <= self.hold_seconds:
                target, state = self.alpha, "hold"
            else:
                target, state = 0.0, "returning" if self.alpha > 0 else "waiting"
            tau = self.return_tau
        self.alpha += (target - self.alpha) * (-math.expm1(-dt / tau))
        if abs(self.alpha - target) < .01:
            self.alpha = target
        self.alpha = float(np.clip(self.alpha, 0, 1))
        return {"alpha": self.alpha, "target": target, "distance": distance, "state": state, "left_selected": selected}
