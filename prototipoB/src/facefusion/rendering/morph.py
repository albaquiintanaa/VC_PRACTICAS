"""Visor de fase 4: resultado grande y dos fuentes de comparación."""

import cv2
import numpy as np

from .landmarks import annotate, annotate_hands, fit


def morph_panel(camera, camera_points, video, video_points, result, elapsed,
                gesture, guide_status, mirror=True, mesh=False, numbers=False,
                overlay=False, hands=None, swap_labels=False, fixed=False):
    output = np.zeros((740, 960, 3), np.uint8)
    rendered = cv2.flip(result.image, 1) if mirror else result.image.copy()
    if mesh and result.geometry is not None:
        points = result.geometry.copy()
        if mirror:
            points[:, 0] = camera.shape[1] - 1 - points[:, 0]
        for triangle in result.triangles:
            cv2.polylines(rendered, [np.rint(points[triangle]).astype(np.int32)], True, (110,210,110), 1, cv2.LINE_AA)
    output[90:570, :640] = fit(rendered)
    webcam = annotate(camera, camera_points, mirror, mesh, numbers, overlay)
    if hands:
        annotate_hands(webcam, hands, mirror, swap_labels)
    output[90:330, 640:] = fit(webcam, 320, 240)
    output[330:570, 640:] = fit(annotate(video, video_points, False, mesh, numbers, overlay), 320, 240)
    lines = [
        ("FACEFUSION - FASE 4: MORPHING GEOMETRICO", 15, 27, .68, (240,240,240)),
        ("M: malla | A: puntos fuentes | E: espejo | S: captura | R: calibrar | Q/Esc: salir",15,53,.43,(190,200,210)),
        (f"RESULTADO {'(espejo)' if mirror else ''}",15,80,.48,(100,245,140)),
        ("FUENTES: webcam / video",650,80,.47,(190,200,210)),
        (f"INTENSIDAD: {result.control_alpha * 100:5.1f}% | {'FIJA (diagnostico)' if fixed else 'MANO IZQUIERDA'}",15,605,.68,(100,245,140)),
        (f"Estado: {result.status} | gesto: {gesture['state']} | CPU | {elapsed:.1f} s",15,634,.48,(190,210,220)),
        (guide_status[0],15,672,.48,(70,220,255)),
        (guide_status[1],15,698,.45,(230,230,230)),
        ("Diagnostico: 0 / 5 / 1 = intensidad 0 / 50 / 100% | G = gesto | mezcla maxima 70%",15,729,.43,(170,180,190)),
    ]
    for text,x,y,scale,color in lines:
        cv2.putText(output,text,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)
    cv2.rectangle(output,(15,645),(945,653),(55,65,70),-1)
    if result.control_alpha > 0:
        cv2.rectangle(output,(15,645),(15+round(930*result.control_alpha),653),(255,220,50),-1)
    return output
