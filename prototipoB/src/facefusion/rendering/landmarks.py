"""Coordenadas coherentes con el espejo y visor de dos fuentes."""

import cv2
import numpy as np
from mediapipe.tasks.python import vision


def landmark_pixels(points, width, height, mirrored=False):
    pixels = np.rint(points[:, :2] * [width - 1, height - 1]).astype(int)
    if mirrored:
        pixels[:, 0] = width - 1 - pixels[:, 0]
    return pixels


def annotate(frame, points, mirrored=False, mesh=True, numbers=False, overlay=True):
    output = cv2.flip(frame, 1) if mirrored else frame.copy()
    if points is None or not overlay:
        return output
    height, width = output.shape[:2]
    pixels = landmark_pixels(points, width, height, mirrored)

    def valid(i):
        return 0 <= pixels[i, 0] < width and 0 <= pixels[i, 1] < height

    connections = vision.FaceLandmarksConnections
    groups = [(connections.FACE_LANDMARKS_FACE_OVAL, (255, 220, 50)),
              (connections.FACE_LANDMARKS_LIPS, (220, 90, 255)),
              (connections.FACE_LANDMARKS_LEFT_EYE, (40, 220, 255)),
              (connections.FACE_LANDMARKS_RIGHT_EYE, (40, 220, 255))]
    if mesh:
        groups.insert(0, (connections.FACE_LANDMARKS_TESSELATION, (90, 100, 90)))
    for edges, color in groups:
        for edge in edges:
            if valid(edge.start) and valid(edge.end):
                cv2.line(output, tuple(pixels[edge.start]), tuple(pixels[edge.end]), color, 1, cv2.LINE_AA)
    for i, pixel in enumerate(pixels):
        if valid(i):
            cv2.circle(output, tuple(pixel), 1, (70, 255, 100), -1, cv2.LINE_AA)
    if numbers:
        for i in (1, 33, 61, 152, 263, 291):
            if valid(i):
                position = tuple(pixels[i] + [3, -4])
                cv2.putText(output, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(output, str(i), position, cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def fit(frame, width=640, height=480):
    scale = min(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
    output = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    output[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return output


def annotate_hands(image, hands, mirrored, swap_labels=False):
    from facefusion.gestures.pinch import anatomical_label
    height, width = image.shape[:2]
    for hand in hands:
        points = landmark_pixels(hand.points, width, height, mirrored)
        color = (255, 220, 50) if anatomical_label(hand, swap_labels) == "Left" else (160, 160, 160)
        for edge in vision.HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(image, tuple(points[edge.start]), tuple(points[edge.end]), color, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(image, tuple(point), 4 if index in (4, 8) else 2, (220, 90, 255) if index in (4, 8) else color, -1, cv2.LINE_AA)
        cv2.line(image, tuple(points[4]), tuple(points[8]), (220, 90, 255), 2, cv2.LINE_AA)
        label = f"{'IZQ' if anatomical_label(hand, swap_labels) == 'Left' else 'DER'} raw:{hand.label} {hand.confidence:.2f}"
        position = (max(3, min(width - 220, int(points[0, 0]))), max(18, min(height - 6, int(points[0, 1]) + 24)))
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, label, position, cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    return image


def gesture_hud(panel, sample, guide_status):
    output = np.vstack((panel, np.zeros((156, 1280, 3), dtype=np.uint8)))
    output[:36] = 0
    cv2.putText(output, "FACEFUSION - FASE 3: CONTROL DE PINZA IZQUIERDA", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(output, f"MEZCLA PREVISTA: {sample['alpha'] * 100:5.1f}%", (15, 676), cv2.FONT_HERSHEY_SIMPLEX, .8, (100, 245, 140), 2, cv2.LINE_AA)
    distance = '--' if sample['distance'] is None else f"{sample['distance']:.3f}"
    cv2.putText(output, f"Estado: {sample['state']} | pinza/palma: {distance} | R: recalibrar", (630, 676), cv2.FONT_HERSHEY_SIMPLEX, .45, (190, 210, 220), 1, cv2.LINE_AA)
    cv2.rectangle(output, (15, 690), (1265, 702), (55, 65, 70), -1)
    if sample['alpha'] > 0:
        cv2.rectangle(output, (15, 690), (15 + round(1250 * sample['alpha']), 702), (255, 220, 50), -1)
    cv2.putText(output, guide_status[0], (15, 728), cv2.FONT_HERSHEY_SIMPLEX, .55, (70, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(output, guide_status[1], (15, 757), cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(output, "El morphing facial se implementa en la fase 4. Aqui validamos su coeficiente de control.", (15, 786), cv2.FONT_HERSHEY_SIMPLEX, .46, (170, 180, 190), 1, cv2.LINE_AA)
    return output


def dual_panel(camera, camera_points, video, video_points, sample, fps, elapsed, mirror=True, mesh=True, numbers=False, overlay=True, hands=None, swap_labels=False):
    left_image = annotate(camera, camera_points, mirror, mesh, numbers, overlay)
    if hands is not None:
        annotate_hands(left_image, hands, mirror, swap_labels)
    left = fit(left_image)
    right = fit(annotate(video, video_points, False, mesh, numbers, overlay))
    panel = np.zeros((644, 1280, 3), dtype=np.uint8)
    panel[116:596, :640] = left
    panel[116:596, 640:] = right
    cv2.putText(panel, "FACEFUSION - FASE 2: CAPTURA Y SEGUIMIENTO", (15, 27), cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(panel, "M: malla | N: indices | E: espejo webcam | A: puntos | Q/ESC: cerrar", (15, 57), cv2.FONT_HERSHEY_SIMPLEX, .5, (190, 200, 210), 1, cv2.LINE_AA)
    for x, title, points in [(15, "WEBCAM EN DIRECTO" + (" (espejo)" if mirror else ""), camera_points), (655, "VIDEO GRABADO", video_points)]:
        count = len(points) if points is not None else 0
        color = (100, 245, 140) if count else (70, 190, 255)
        cv2.putText(panel, f"{title} - {count} puntos", (x, 101), cv2.FONT_HERSHEY_SIMPLEX, .53, color, 1, cv2.LINE_AA)
    cv2.putText(panel, f"Tiempo: {elapsed:.1f} s | CPU | Seguidores independientes", (15, 624), cv2.FONT_HERSHEY_SIMPLEX, .52, (190, 200, 210), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Vuelta {sample.loop + 1} | Frame {sample.index} | t={sample.index / fps:.2f} s", (655, 624), cv2.FONT_HERSHEY_SIMPLEX, .52, (190, 200, 210), 1, cv2.LINE_AA)
    return panel
