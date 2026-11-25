"""Muestra webcam y vídeo con seguidores independientes y repetición temporizada."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from facefusion.capture.sources import CameraSource, TimedVideoSource
from facefusion.rendering.landmarks import dual_panel, gesture_hud
from facefusion.tracking.face import FaceTracker
from facefusion.tracking.hands import HandTracker
from facefusion.gestures.pinch import PinchCalibration, PinchController
from facefusion.gestures.guide import GestureGuide
from facefusion.morphing import FaceMorpher
from facefusion.morphing.validation import MorphGuide
from facefusion.rendering.morph import morph_panel

WINDOW = "FaceFusion - webcam y video"


def window_visible():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def positive(value):
    number = float(value)
    if not 0 < number < 86400:
        raise argparse.ArgumentTypeError("Duracion fuera del rango (0, 86400).")
    return number


def blend(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("Mezcla fuera del rango [0, 1].")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--duration", type=positive)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Exigir dos reinicios del clip y >=90%% de detecciones en cada fuente.")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--gestures", action="store_true", help="Activar control con la mano izquierda.")
    parser.add_argument("--morph", action="store_true", help="Morphing geometrico; activa el control gestual.")
    parser.add_argument("--validate-morph", action="store_true", help="Recoger extremos y mezcla intermedia con la mano; activa --morph.")
    parser.add_argument("--morph-alpha", type=blend, help="Intensidad fija [0,1], escalada a mezcla [0,0.70]; activa --morph. G vuelve al gesto.")
    parser.add_argument("--guide", action="store_true", help="Calibracion y validacion fisica guiadas de fase 3; activa --gestures.")
    parser.add_argument("--resume-gesture-report", type=Path, help="Retomar una guia parcial con la misma camara y calibracion.")
    parser.add_argument("--calibration", type=Path, default=ROOT / "outputs/metrics/hand_calibration.json")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    args.morph = args.morph or args.morph_alpha is not None or args.validate_morph
    args.guide = args.guide or args.resume_gesture_report is not None
    args.gestures = args.gestures or args.guide or args.morph
    args.report = args.report or ROOT / "outputs/metrics" / ("phase4_preview.json" if args.morph else "phase3_resumed.json" if args.resume_gesture_report else "phase3_preview.json" if args.gestures else "phase2_preview.json")
    if args.morph and args.guide:
        parser.error("Ejecutar la guia completa de fase 3 por separado, sin --morph.")
    if args.validate_morph and (args.headless or args.morph_alpha is not None):
        parser.error("La prueba gestual del morphing requiere ventana y coeficiente controlado con la mano.")
    resume_data = None
    resume_bytes = None
    if args.resume_gesture_report:
        if args.report.resolve() == args.resume_gesture_report.resolve():
            parser.error("La salida debe ser distinta del informe original para conservarlo.")
        try:
            resume_bytes = args.resume_gesture_report.read_bytes()
            resume_data = json.loads(resume_bytes)
            # Comprobar el historial antes de abrir la cámara.
            GestureGuide.resume(PinchController(), args.calibration, resume_data, args.resume_gesture_report, args.camera)
        except (OSError, ValueError, TypeError, KeyError) as error:
            parser.error(f"No se puede reanudar: {error}")
    if args.headless and args.duration is None:
        parser.error("--headless requiere --duration.")
    if not args.video.is_file():
        parser.error(f"Video inexistente: {args.video}")
    if args.headless and args.guide:
        parser.error("La guia requiere ventana y participacion de una persona.")
    camera = video = camera_tracker = video_tracker = None
    hand_tracker = controller = guide = None
    gesture_trace = []
    morpher = FaceMorpher() if args.morph else None
    morph_guide = MorphGuide() if args.validate_morph else None
    morph_trace, morph_captures = [], []
    fixed_alpha = args.morph_alpha
    started = None
    loops = []
    snapshot_saved = False
    max_clock_error = 0.0
    mirror, mesh, numbers, overlay = not args.no_mirror, True, False, True
    if args.morph:
        mesh = overlay = False
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "status": "error", "delegate": "CPU", "video_path": str(args.video.resolve()), "camera_index": args.camera, "snapshot_saved": False, "mirror_modes_seen": [], "stop_reason": "error"}
    if resume_data is not None:
        report["resumed_from_sha256"] = sha256(resume_bytes).hexdigest()
    mirror_modes = set()
    try:
        with ExitStack() as stack:
            # Abrir vídeo primero: un archivo inválido no ocupa la webcam.
            video = stack.enter_context(TimedVideoSource(args.video))
            camera = stack.enter_context(CameraSource(args.camera))
            model = ROOT / "models/pretrained/face_landmarker.task"
            camera_tracker = stack.enter_context(FaceTracker(model, "webcam"))
            video_tracker = stack.enter_context(FaceTracker(model, "video"))
            report["independent_trackers"] = camera_tracker._task is not video_tracker._task
            if args.gestures:
                hand_tracker = stack.enter_context(HandTracker(ROOT / "models/pretrained/hand_landmarker.task"))
                calibration = None
                if not args.guide and args.calibration.exists():
                    try:
                        calibration = PinchCalibration.load(args.calibration, args.camera)
                    except (ValueError, TypeError, KeyError) as error:
                        report["calibration_load_error"] = str(error)
                controller = PinchController(calibration)
                if resume_data is not None:
                    guide = GestureGuide.resume(controller, args.calibration, resume_data, args.resume_gesture_report, args.camera)
                elif calibration is None:
                    guide = GestureGuide(controller, args.calibration, args.camera, args.guide)
            if not args.headless:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, 960 if args.morph else 800 if args.gestures else 1280, 740 if args.morph else 500 if args.gestures else 644)
                if args.gestures:
                    cv2.moveWindow(WINDOW, 20, 20)
                stack.callback(cv2.destroyAllWindows)
            started = time.monotonic()
            last_sample = None
            video_points = None
            print("Dos fuentes abiertas. Mantente frente a la webcam. Q/ESC para cerrar.", flush=True)
            while True:
                if not args.headless and not window_visible():
                    report["stop_reason"] = "window"
                    break
                if args.duration is not None and time.monotonic() - started >= args.duration:
                    report["stop_reason"] = "duration"
                    break
                camera_frame = camera.read()
                timestamp = max(camera_tracker.last_timestamp + 1, round((time.monotonic() - started) * 1000))
                camera_points = camera_tracker.detect(camera_frame, timestamp)
                hands = hand_tracker.detect(camera_frame, timestamp) if hand_tracker else None
                elapsed = time.monotonic() - started
                gesture_sample = controller.update(hands, camera_frame.shape[1], camera_frame.shape[0], elapsed) if controller else None
                completed_stage = guide.update(hands, camera_frame.shape[1], camera_frame.shape[0], elapsed, gesture_sample, mirror) if guide else None
                if controller:
                    gesture_trace.append(dict(gesture_sample, time=elapsed, mirror=mirror,
                                              hands=[{"label": h.label, "confidence": h.confidence} for h in hands]))
                sample = video.frame_at(elapsed)
                max_clock_error = max(max_clock_error, (elapsed - sample.media_seconds) * 1000)
                if last_sample != sample.key:
                    if last_sample is None or sample.loop != last_sample[0]:
                        if last_sample is not None:
                            video_tracker.reset()
                        loops.append({"loop": sample.loop, "elapsed_seconds": round(elapsed, 4), "first_frame": sample.index})
                    video_timestamp = max(video_tracker.last_timestamp + 1, round(sample.media_seconds * 1000))
                    video_points = video_tracker.detect(sample.image, video_timestamp)
                    last_sample = sample.key
                swap = guide.swap_labels if guide else controller.calibration.swap_labels if controller and controller.calibration else False
                guidance = guide.status() if guide else ("CALIBRADO: controla con la IZQUIERDA", "Juntos: intensidad 0% | Separados: intensidad 100% | S: captura")
                morph_stage = None
                if morpher:
                    alpha = gesture_sample["alpha"] if fixed_alpha is None else fixed_alpha
                    morph_started = time.perf_counter()
                    result = morpher.render(camera_frame, camera_points, sample.image, video_points, alpha, time_seconds=elapsed)
                    morph_trace.append({"time": elapsed, "alpha": result.alpha, "control_alpha": alpha, "status": result.status,
                                        "tone": result.tone,
                                        "fixed": fixed_alpha is not None, "mirror": mirror,
                                        "render_ms": (time.perf_counter() - morph_started) * 1000})
                    if morph_guide and not (guide and not guide.done):
                        morph_stage = morph_guide.update(result, gesture_sample, elapsed, fixed_alpha is not None)
                        guidance = morph_guide.status()
                    panel = morph_panel(camera_frame, camera_points, sample.image, video_points, result,
                                        elapsed, gesture_sample, guidance, mirror, mesh, numbers, overlay,
                                        hands, swap, fixed_alpha is not None)
                    if morph_stage:
                        directory = args.snapshot.parent / (args.snapshot.stem + "_evidence") if args.snapshot else ROOT / "outputs/recordings/phase4_evidence"
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / (morph_stage + ".png")
                        if not cv2.imwrite(str(path), panel):
                            raise RuntimeError("No se pudo guardar la mezcla guiada.")
                        bundle = directory / (morph_stage + ".npz")
                        np.savez_compressed(bundle, camera=camera_frame, video=sample.image,
                                            camera_points=camera_points, video_points=video_points,
                                            result=result.image, mask=result.mask, alpha=result.alpha,
                                            control_alpha=alpha, max_blend=morpher.max_blend,
                                            geometry=result.geometry, triangles=result.triangles)
                        morph_captures.append(dict(morph_trace[-1], stage=morph_stage, path=str(path), bundle=str(bundle)))
                        print(f"Mezcla capturada: {morph_stage} ({result.alpha:.3f}). {path}", flush=True)
                else:
                    panel = dual_panel(camera_frame, camera_points, sample.image, video_points, sample, video.fps, elapsed, mirror, mesh, numbers, overlay, hands, swap)
                    if controller:
                        panel = gesture_hud(panel, gesture_sample, guidance)
                if completed_stage:
                    print(f"Paso completado: {completed_stage}", flush=True)
                    if args.snapshot:
                        evidence = args.snapshot.parent / (args.snapshot.stem + '_evidence')
                        evidence.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(evidence / f'{completed_stage}.png'), panel):
                            raise RuntimeError("No se pudo guardar evidencia del gesto.")
                mirror_modes.add(mirror)
                if args.snapshot and not snapshot_saved and elapsed >= 1 and camera_points is not None and video_points is not None:
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(args.snapshot), panel):
                        raise RuntimeError("No se pudo guardar la captura del visor.")
                    snapshot_saved = True
                    print(f"Captura guardada: {args.snapshot}", flush=True)
                if not args.headless:
                    cv2.imshow(WINDOW, panel)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        report["stop_reason"] = "key"
                        break
                    if not window_visible():
                        report["stop_reason"] = "window"
                        break
                    if key in (ord("e"), ord("E")):
                        mirror = not mirror
                    elif key in (ord("m"), ord("M")):
                        mesh = not mesh
                    elif key in (ord("n"), ord("N")):
                        numbers = not numbers
                    elif key in (ord("a"), ord("A")):
                        overlay = not overlay
                    elif morpher and key in (ord("0"), ord("5"), ord("1")):
                        fixed_alpha = {ord("0"): 0., ord("5"): .5, ord("1"): 1.}[key]
                    elif morpher and key in (ord("g"), ord("G")):
                        fixed_alpha = None
                    elif morpher and key in (ord("s"), ord("S")):
                        directory = args.snapshot.parent / (args.snapshot.stem + "_evidence") if args.snapshot else ROOT / "outputs/recordings/phase4_evidence"
                        directory.mkdir(parents=True, exist_ok=True)
                        path = directory / f"capture_{len(morph_captures)+1:02d}_{result.alpha:.3f}.png"
                        if not cv2.imwrite(str(path), panel):
                            raise RuntimeError("No se pudo guardar evidencia de morphing.")
                        raw_path = path.with_name(path.stem + "_result.png")
                        if not cv2.imwrite(str(raw_path), result.image):
                            raise RuntimeError("No se pudo guardar el resultado sin interfaz.")
                        morph_captures.append(dict(morph_trace[-1], path=str(path), result_path=str(raw_path)))
                        print(f"Morphing guardado: {path}", flush=True)
                    elif key == 32 and guide:
                        guide.start_capture(time.monotonic() - started)
                    elif key in (ord("r"), ord("R")) and controller:
                        controller.set_calibration(None)
                        guide = GestureGuide(controller, args.calibration, args.camera, args.guide)
                if args.guide and guide and guide.done:
                    report["stop_reason"] = "gesture_guide_completed"
                    break
            report["status"] = "ok"
    except (RuntimeError, ValueError, OSError, cv2.error) as error:
        report["error"] = f"{type(error).__name__}: {error}"
    except KeyboardInterrupt:
        report.update(status="ok", stop_reason="interrupt")
    finally:
        report["snapshot_saved"] = snapshot_saved
        report["mirror_modes_seen"] = sorted(mirror_modes)
        report["elapsed_seconds"] = round(time.monotonic() - started, 4) if started else 0
        report["camera"] = {"frames_read": camera.frames_read, "backend": camera.backend, "closed": camera.closed} if camera else None
        report["video"] = {"fps": video.fps, "frame_count": video.frame_count, "duration_seconds": video.duration, "frames_read": video.frames_read, "skipped_frames": video.skipped_frames, "rewinds": video.rewinds, "closed": video.closed, "loops": loops, "max_selection_error_ms": round(max_clock_error, 4)} if video else None
        report["trackers"] = [tracker.summary() for tracker in (camera_tracker, video_tracker) if tracker]
        report["resources_closed"] = all(resource.closed for resource in (camera, video, camera_tracker, video_tracker, hand_tracker) if resource)
        if morpher:
            report["morphing"] = dict(morpher.summary(), trace=morph_trace, captures=morph_captures)
            if morph_guide:
                report["morphing"]["guide"] = morph_guide.summary()
                report["morph_numeric_validation_passed"] = morph_guide.done and report["resources_closed"] and report["status"] == "ok"
                if not report["morph_numeric_validation_passed"] and report["status"] == "ok":
                    report["status"] = "review"
        if controller:
            report["gestures"] = {"hand_tracker_closed": hand_tracker.closed if hand_tracker else None,
                                  "calibration_file": str(args.calibration), "trace": gesture_trace,
                                  "parameters": {"tau": controller.tau, "hold_seconds": controller.hold_seconds,
                                                 "return_tau": controller.return_tau, "endpoint_zone": controller.endpoint_zone,
                                                 "confidence": .75, "min_palm_px": 20, "acquisition_frames": 3},
                                  "guide": guide.summary() if guide else None}
            if guide:
                report["gestures"]["guided_trace"] = guide.trace
            if args.guide:
                report["gesture_validation_passed"] = bool(guide and guide.done and report["resources_closed"] and report["status"] == "ok")
                if not report["gesture_validation_passed"] and report["status"] == "ok":
                    report["status"] = "review"
        if args.validate:
            checks = {
                "execution_ok": report["status"] == "ok",
                "independent_trackers": report.get("independent_trackers", False),
                "camera_and_video_read": bool(camera and video and camera.frames_read and video.frames_read),
                "two_video_restarts": bool(video and video.rewinds >= 2),
                "tracking_stable_90_percent": len(report["trackers"]) == 2 and all(t["face_ratio"] >= .9 for t in report["trackers"]),
                "monotonic_valid_results": len(report["trackers"]) == 2 and all(t["timestamp_errors"] == 0 and t["invalid_results"] == 0 for t in report["trackers"]),
                "video_clock_quantization": bool(video and max_clock_error <= 1000 / video.fps + .001),
                "resources_closed": report["resources_closed"],
                "snapshot_saved_if_requested": not args.snapshot or snapshot_saved,
            }
            report["validation_checks"] = checks
            report["validation_passed"] = all(checks.values())
            if not report["validation_passed"] and report["status"] == "ok":
                report["status"] = "review"
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        printable = dict(report)
        if "gestures" in printable:
            printable["gestures"] = {key: value for key, value in report["gestures"].items() if key not in ("trace", "guided_trace")}
            printable["gestures"]["trace_rows"] = len(gesture_trace)
        if "morphing" in printable:
            printable["morphing"] = {key: value for key,value in report["morphing"].items() if key != "trace"}
            printable["morphing"]["trace_rows"] = len(morph_trace)
        print(json.dumps(printable, indent=2, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
