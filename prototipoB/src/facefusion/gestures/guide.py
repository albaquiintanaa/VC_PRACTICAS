"""Prueba guiada: lateralidad, calibración y casos físicos de aceptación."""

from collections import Counter
from dataclasses import asdict
import copy

import numpy as np

from .pinch import PinchCalibration, pinch_distance, select_hand

STAGES = ["verify_left", "verify_right", "calibrate_closed", "calibrate_open",
          "test_closed", "test_open", "test_middle", "test_right", "test_both",
          "loss_open", "loss_empty", "recovery", "mirror"]
HINTS = {
    "verify_left": "SOLO tu IZQUIERDA, palma visible. ESPACIO y mantenla.",
    "verify_right": "Retira izquierda. SOLO tu DERECHA. ESPACIO y mantenla.",
    "calibrate_closed": "IZQUIERDA: pulgar e indice juntos. ESPACIO y mantenlos juntos.",
    "calibrate_open": "IZQUIERDA: abre pulgar e indice al maximo comodo. ESPACIO y manten.",
    "test_closed": "IZQUIERDA: pinza cerrada para comprobar 0%. ESPACIO y manten.",
    "test_open": "IZQUIERDA: pinza abierta para comprobar 100%. ESPACIO y manten.",
    "test_middle": "IZQUIERDA: apertura intermedia (15-85%). ESPACIO y manten.",
    "test_right": "SOLO DERECHA: ESPACIO, abre y cierra su pinza durante la captura.",
    "test_both": "AMBAS: izquierda CERRADA; ESPACIO y mueve la pinza DERECHA.",
    "loss_open": "IZQUIERDA ABIERTA: ESPACIO y manten para preparar la perdida.",
    "loss_empty": "Retira AMBAS manos. ESPACIO y espera a que el control vuelva a 0%.",
    "recovery": "Recupera la IZQUIERDA ABIERTA. ESPACIO y manten hasta 100%.",
    "mirror": "IZQUIERDA ABIERTA: ESPACIO y manten; probaremos ambos espejos.",
}


class GestureGuide:
    def __init__(self, controller, calibration_path, camera_index=0, full_validation=False):
        self.controller, self.calibration_path = controller, calibration_path
        self.camera_index, self.full_validation = camera_index, full_validation
        self.stages = STAGES if full_validation else STAGES[:4]
        self.index = 0
        self.started = None
        self.rows = []
        self.trace = []
        self.results = {}
        self.events = []
        self.message = ""
        self.swap_labels = False
        self.left_label = None
        self.closed_samples = []
        self.mirror_first = None
        self.loss_trace_start = 0
        self.loss_reacquisitions = 0
        self.resumed_from = None
        self.previous_summary = None

    @classmethod
    def resume(cls, controller, calibration_path, report, source, camera_index=0):
        previous = report.get("gestures", {}).get("guide")
        if not previous or not previous.get("full_validation") or previous.get("completed"):
            raise ValueError("Se necesita un informe de guia completa pendiente de terminar.")
        if report.get("camera_index") != camera_index:
            raise ValueError("El informe pertenece a otra camara; repetir calibracion.")
        results = previous.get("results", {})
        count = len(results)
        if count < 4 or count >= len(STAGES) or set(results) != set(STAGES[:count]) or not all(item.get("passed") is True for item in results.values()):
            raise ValueError("El informe no contiene una secuencia parcial valida con calibracion terminada.")
        calibration = PinchCalibration(**previous["calibration"])
        if calibration.camera_index != camera_index or calibration.swap_labels != previous.get("swap_labels"):
            raise ValueError("Calibracion y lateralidad del informe no coinciden.")
        expected = {"tau": controller.tau, "hold_seconds": controller.hold_seconds,
                    "return_tau": controller.return_tau, "endpoint_zone": controller.endpoint_zone,
                    "confidence": .75, "min_palm_px": 20, "acquisition_frames": 3}
        if report["gestures"].get("parameters") != expected:
            raise ValueError("Los parametros han cambiado; repetir la validacion completa.")
        controller.set_calibration(calibration)
        guide = cls(controller, calibration_path, camera_index, full_validation=True)
        guide.swap_labels = calibration.swap_labels
        guide.left_label = previous["raw_label_for_anatomical_left"]
        # La pérdida debe medirse desde un estado abierto observado en esta sesión.
        # Los resultados originales permanecen completos en previous_summary.
        guide.index = min(count, STAGES.index("loss_open"))
        guide.results = {key: copy.deepcopy(results[key]) for key in STAGES[:guide.index]}
        guide.resumed_from = str(source)
        guide.previous_summary = copy.deepcopy(previous)
        return guide

    @property
    def done(self):
        return self.index >= len(self.stages)

    @property
    def stage(self):
        return "done" if self.done else self.stages[self.index]

    def start_capture(self, now):
        if not self.done and self.started is None:
            self.started, self.rows, self.message = now, [], ""

    def status(self):
        if self.done:
            return "VALIDACION COMPLETA" if self.full_validation else "CALIBRADO: controla con la IZQUIERDA", "R: recalibrar | Q/ESC: cerrar"
        hint = HINTS[self.stage]
        if self.stage == "mirror" and self.mirror_first is not None:
            hint = "Pulsa E para cambiar espejo. IZQUIERDA ABIERTA: ESPACIO y manten."
        state = f"Paso {self.index + 1}/{len(self.stages)} - {self.stage}"
        if self.started is not None:
            state += f" | CAPTURANDO: {len(self.rows)} muestras"
        if self.message:
            hint = self.message + " ESPACIO para repetir."
        return state, hint

    def update(self, hands, width, height, now, sample, mirror):
        left = select_hand(hands, "Left", width, height, self.swap_labels)
        right = select_hand(hands, "Right", width, height, self.swap_labels)
        single = hands[0] if len(hands) == 1 and hands[0].confidence >= .75 and pinch_distance(hands[0], width, height) is not None else None
        row = dict(sample, time=now, stage=self.stage, mirror=mirror,
                   raw_single=single.label if single else None,
                   left_ratio=pinch_distance(left, width, height) if left else None,
                   right_ratio=pinch_distance(right, width, height) if right else None,
                   hands_count=len(hands), reacquisitions=self.controller.reacquisitions)
        self.trace.append(row)
        if self.started is None or self.done:
            return None
        self.rows.append(row)
        minimum_seconds = 2.2 if self.stage in ("test_right", "test_both", "loss_empty") else 1.3
        if now - self.started < minimum_seconds:
            return None
        stage = self.stage
        if stage in ("verify_left", "verify_right"):
            valid = [r for r in self.rows if r["raw_single"] in ("Left", "Right")]
        elif stage in ("calibrate_closed", "calibrate_open"):
            valid = [r for r in self.rows if r["left_ratio"] is not None]
        elif stage == "test_right":
            valid = [r for r in self.rows if r["left_ratio"] is None and r["right_ratio"] is not None]
        elif stage == "test_both":
            valid = [r for r in self.rows if r["left_ratio"] is not None and r["right_ratio"] is not None and r["left_selected"]]
        elif stage == "loss_empty":
            valid = [r for r in self.rows if r["hands_count"] == 0]
        else:
            valid = [r for r in self.rows if r["left_selected"]]
        if len(valid) < 8:
            if now - self.started < 6:
                return None
            self.message = "Pocas muestras validas: revisa mano, luz y encuadre."
            self.started = None
            return None
        try:
            tail = valid[-3:]
            if stage in ("verify_left", "verify_right"):
                label, count = Counter(r["raw_single"] for r in valid).most_common(1)[0]
                if count / len(valid) < .85:
                    raise ValueError("Etiqueta inestable; muestra una sola palma de frente.")
                if stage == "verify_left":
                    self.left_label = label
                else:
                    if label == self.left_label:
                        raise ValueError("Misma etiqueta que la izquierda; verifica que muestras la DERECHA.")
                    self.swap_labels = self.left_label == "Right"
            elif stage == "calibrate_closed":
                self.closed_samples = [r["left_ratio"] for r in valid]
            elif stage == "calibrate_open":
                calibration = PinchCalibration(float(np.median(self.closed_samples)),
                                               float(np.median([r["left_ratio"] for r in valid])),
                                               self.swap_labels, self.camera_index)
                self.controller.set_calibration(calibration)
                calibration.save(self.calibration_path)
            elif stage in ("test_closed", "test_both"):
                if not all(r["alpha"] == 0 and r["target"] == 0 for r in tail):
                    raise ValueError("La izquierda no llega a 0%; cierra mas la pinza.")
                if stage == "test_both" and np.ptp([r["right_ratio"] for r in valid]) < .2:
                    raise ValueError("Mueve la pinza DERECHA mientras la izquierda permanece cerrada.")
            elif stage in ("test_open", "loss_open", "recovery", "mirror"):
                if not all(r["alpha"] == 1 and r["target"] == 1 for r in tail):
                    raise ValueError("La izquierda no llega a 100%; abre mas la pinza y manten.")
                if stage == "loss_open":
                    self.loss_trace_start = len(self.trace)
                    self.loss_reacquisitions = self.controller.reacquisitions
                if stage == "recovery" and self.controller.reacquisitions <= self.loss_reacquisitions:
                    raise ValueError("No se ha observado la recuperacion de la izquierda tras la perdida.")
                if stage == "mirror":
                    modes = {r["mirror"] for r in valid}
                    if len(modes) != 1:
                        raise ValueError("Cambia espejo antes de pulsar ESPACIO, no durante la captura.")
                    mode = modes.pop()
                    if self.mirror_first is None:
                        self.mirror_first = mode
                        self.started = None
                        self.events.append({"stage": "mirror_first", "mirror": mode, "samples": len(valid)})
                        return "mirror_first"
                    if mode == self.mirror_first:
                        raise ValueError("Pulsa E para probar el otro modo de espejo.")
            elif stage == "test_middle":
                if not all(.15 < r["alpha"] < .85 and .1 < r["target"] < .9 for r in tail):
                    raise ValueError("Busca una apertura intermedia, entre 15% y 85%.")
            elif stage == "test_right":
                if np.ptp([r["right_ratio"] for r in valid]) < .2:
                    raise ValueError("Abre y cierra la pinza DERECHA durante la captura.")
                if not all(r["alpha"] == 0 and not r["left_selected"] for r in tail):
                    raise ValueError("Retira la IZQUIERDA y espera al retorno a 0%.")
            elif stage == "loss_empty":
                states = {r["state"] for r in self.trace[self.loss_trace_start:]}
                if not all(r["alpha"] == 0 for r in tail) or not {"hold", "returning"}.issubset(states):
                    raise ValueError("Retira todas las manos y espera a completar retencion y retorno.")
        except ValueError as error:
            self.message, self.started = str(error), None
            return None
        self.results[stage] = {"passed": True, "samples": len(valid), "alpha_min": min(r["alpha"] for r in valid), "alpha_max": max(r["alpha"] for r in valid)}
        self.events.append({"stage": stage, "time": now, "samples": len(valid)})
        self.index += 1
        self.started = None
        return stage

    def summary(self):
        return {"full_validation": self.full_validation, "completed": self.done,
                "stage": self.stage, "results": self.results, "events": self.events,
                "raw_label_for_anatomical_left": self.left_label, "swap_labels": self.swap_labels,
                "calibration": asdict(self.controller.calibration) if self.controller.calibration else None,
                "resumed_from": self.resumed_from, "previous_summary": self.previous_summary}
