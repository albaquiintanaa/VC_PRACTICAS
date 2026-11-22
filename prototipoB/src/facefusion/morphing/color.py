"""Igualación robusta de iluminación y cromaticidad de regiones faciales."""

import cv2
import numpy as np


def blend_boundary(source, target, mask):
    """Integra gradientes faciales con las condiciones de color del contorno.

    `source` contiene color sin premultiplicar; introducir aquí una imagen ya
    fundida conservaría el gradiente artificial del borde que queremos quitar.
    """
    binary = ((mask > .05) * 255).astype(np.uint8)
    x, y, width, height = cv2.boundingRect(binary)
    if width < 4 or height < 4 or x < 1 or y < 1 or x+width >= target.shape[1] or y+height >= target.shape[0]:
        fallback = np.rint(source * mask[...,None] + target * (1-mask[...,None])).astype(np.uint8)
        return fallback, {"status": "fallback_small_or_clipped_mask"}
    try:
        cloned = cv2.seamlessClone(source, target, binary, (x+width//2, y+height//2), cv2.NORMAL_CLONE)
    except cv2.error:
        fallback = np.rint(source * mask[...,None] + target * (1-mask[...,None])).astype(np.uint8)
        return fallback, {"status": "fallback_clone_error"}
    output = np.rint(cloned * mask[...,None] + target * (1-mask[...,None])).astype(np.uint8)
    band = (mask > .15) & (mask < .65)
    info = {"status": "poisson", "boundary_pixels": int(np.count_nonzero(band))}
    if band.any():
        baseline = source * mask[...,None] + target * (1-mask[...,None])
        info["boundary_difference_before"] = float(np.abs(baseline[band]-target[band]).mean())
        info["boundary_difference_after"] = float(np.abs(output[band].astype(np.float32)-target[band]).mean())
    return output, info


class SkinToneMatcher:
    """Desplaza medianas Lab, con límites y suavizado temporal opcional.

    Recibe una máscara geométrica de muestras de mejillas; no clasifica piel
    por color ni presupone un tono concreto. No cambia el contraste local.
    """

    def __init__(self, tau=.30):
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("Suavizado de color invalido.")
        self.tau = tau
        self.reset()

    def reset(self):
        self.offset = None
        self.last_time = None

    def correct(self, source, target, sample_mask, now=None):
        if now is not None and (not np.isfinite(now) or now < 0
                                or self.last_time is not None and now < self.last_time):
            raise ValueError("El reloj de color debe ser finito y no retroceder.")
        source_lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        target_lab = cv2.cvtColor(target.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
        valid = ((sample_mask > .5) & (source_lab[..., 0] > 5) & (source_lab[..., 0] < 95)
                 & (target_lab[..., 0] > 5) & (target_lab[..., 0] < 95))
        count = int(np.count_nonzero(valid))
        if count < 64:
            self.reset()
            return source.astype(np.float32), {"status": "insufficient_samples", "samples": count}
        source_median = np.median(source_lab[valid], axis=0)
        target_median = np.median(target_lab[valid], axis=0)
        desired = np.clip(target_median - source_median, [-30., -18., -18.], [30., 18., 18.]).astype(np.float32)
        if self.offset is None or now is None or self.last_time is None:
            self.offset = desired
        else:
            weight = 1 - np.exp(-(now - self.last_time) / self.tau)
            self.offset += weight * (desired - self.offset)
        self.last_time = now
        corrected_lab = source_lab + self.offset
        corrected_lab[..., 0] = np.clip(corrected_lab[..., 0], 0, 100)
        corrected_lab[..., 1:] = np.clip(corrected_lab[..., 1:], -127, 127)
        corrected = np.clip(cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2BGR) * 255, 0, 255)
        final_lab = cv2.cvtColor(corrected / 255, cv2.COLOR_BGR2LAB)
        info = {"status": "matched", "samples": count,
                "lab_offset": self.offset.tolist(),
                "median_error_before": float(np.linalg.norm(source_median - target_median)),
                "median_error_after": float(np.linalg.norm(np.median(final_lab[valid], axis=0) - target_median))}
        return corrected, info
