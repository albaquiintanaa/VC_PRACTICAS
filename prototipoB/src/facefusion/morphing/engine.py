"""Alineación por similitud, deformación triangular y composición en CPU.

Se remuestrean las texturas directamente desde las fuentes originales: el
lienzo canónico solo define la geometría y los mapas de muestreo.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

from .color import SkinToneMatcher, blend_boundary


# Orden del óvalo de MediaPipe. Se excluyen iris y bordes móviles de párpados:
# cerrar los ojos no debe colapsar un triángulo de la malla de deformación.
OVAL = (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109)
LANDMARK_IDS = OVAL + (
    33, 133, 362, 263,                         # esquinas de los ojos
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,  # cejas
    168, 1, 98, 327, 2,                       # nariz
    61, 0, 291, 17,                           # extremos exteriores de labios
    50, 101, 205, 280, 330, 425, 117, 346, 151,  # mejillas y frente
)


class InvalidGeometry(ValueError):
    """La observación no admite una deformación segura."""


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:, :2].T + matrix[:, 2]


def signed_areas(points, triangles):
    p = points[triangles]
    a, b = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]
    return (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * .5


@dataclass
class AlignedFace:
    pixels: np.ndarray
    canonical: np.ndarray
    to_canonical: np.ndarray
    to_image: np.ndarray


@dataclass
class MorphResult:
    image: np.ndarray
    mask: np.ndarray
    status: str
    alpha: float
    geometry: np.ndarray | None = None
    triangles: np.ndarray | None = None
    control_alpha: float = 0.
    tone: dict | None = None

    @property
    def applied(self):
        return self.status == "morphing"


def align_face(points, shape, size=320):
    """Normaliza escala y roll con centros oculares; la nariz valida la pose.

    La transformación es una similitud (no aplasta diferencias faciales).
    Los umbrales 2D son filtros conservadores, no una estimación de pose 3D.
    """
    points = np.asarray(points)
    if points.shape != (478, 3) or not np.isfinite(points).all():
        raise InvalidGeometry("coordenadas_invalidas")
    height, width = shape[:2]
    pixels = points[:, :2].astype(np.float64) * [width - 1, height - 1]
    selected = pixels[list(LANDMARK_IDS)]
    if (selected < 1).any() or (selected > [width - 2, height - 2]).any():
        raise InvalidGeometry("cara_fuera_de_imagen")
    left = pixels[[33, 133]].mean(axis=0)
    right = pixels[[362, 263]].mean(axis=0)
    delta = right - left
    distance = np.linalg.norm(delta)
    if distance < 24:
        raise InvalidGeometry("cara_demasiado_pequena")
    roll = np.degrees(np.arctan2(delta[1], delta[0]))
    if abs(roll) > 35:
        raise InvalidGeometry("pose_no_admitida")
    axis = delta / distance
    down = np.array([-axis[1], axis[0]])
    nose = pixels[1] - (left + right) / 2
    yaw_proxy, pitch_proxy = nose @ axis / distance, nose @ down / distance
    eye_widths = [np.linalg.norm(pixels[33] - pixels[133]),
                  np.linalg.norm(pixels[362] - pixels[263])]
    if (abs(yaw_proxy) > .38 or not .08 <= pitch_proxy <= .95
            or min(eye_widths) / max(max(eye_widths), 1e-6) < .45):
        raise InvalidGeometry("pose_no_admitida")
    scale = size * .26 / distance
    rotation = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
    linear = rotation * scale
    offset = np.array([size * .37, size * .30]) - linear @ left
    matrix = np.column_stack((linear, offset))
    canonical = transform_points(selected, matrix).astype(np.float32)
    if (canonical < 2).any() or (canonical > size - 3).any():
        raise InvalidGeometry("cara_fuera_de_lienzo")
    return AlignedFace(selected.astype(np.float32), canonical, matrix,
                       cv2.invertAffineTransform(matrix))


class FaceMorpher:
    """Una triangulación para toda la sesión; los fallos no conservan caras viejas."""

    def __init__(self, size=320, feather=18.0, min_area=.08, max_blend=.70, color_match=True, seamless=True):
        if not isinstance(size, int) or not 128 <= size <= 640:
            raise ValueError("El lienzo debe medir entre 128 y 640 pixeles.")
        if not np.isfinite(feather) or not 1 <= feather <= size / 8:
            raise ValueError("Borde suave fuera de rango.")
        if not np.isfinite(min_area) or min_area <= 0:
            raise ValueError("Area minima invalida.")
        if not np.isfinite(max_blend) or not 0 < max_blend <= .70:
            raise ValueError("La mezcla maxima debe estar entre cero y 0.70.")
        self.size, self.feather, self.min_area = size, feather, min_area
        self.max_blend, self.color_match = float(max_blend), bool(color_match)
        self.seamless = bool(seamless)
        self.tone_matcher = SkinToneMatcher()
        self.triangles = None
        self.reference = None
        self.counts = {}

    def _topology(self, camera, video):
        if self.triangles is not None:
            return self.triangles
        reference = (camera + video) * .5
        try:
            triangulation = Delaunay(reference)
        except QhullError as error:
            raise InvalidGeometry("malla_degenerada") from error
        triangles = triangulation.simplices.copy()
        # No ignorar puntos omitidos por Qhull ni introducir ruido aleatorio.
        if len(np.unique(triangles)) != len(LANDMARK_IDS):
            raise InvalidGeometry("malla_degenerada")
        self._validate_mesh(camera, video, reference, triangles, reference)
        self.reference = reference.copy()
        self.triangles = triangles
        return triangles

    def _validate_mesh(self, camera, video, destination, triangles, reference):
        expected = np.sign(signed_areas(reference, triangles))
        for points in (camera, video, destination):
            areas = signed_areas(points, triangles)
            if (np.abs(areas) < self.min_area).any():
                raise InvalidGeometry("malla_degenerada")
            if (np.sign(areas) != expected).any():
                raise InvalidGeometry("malla_invertida")

    def _maps(self, destination, camera_pixels, video_pixels, triangles):
        # La misma rasterización sirve para ambas texturas. Cada píxel se
        # asigna una sola vez; las aristas compartidas no se suman ni oscurecen.
        maps = np.full((4, self.size, self.size), -1, dtype=np.float32)
        covered = np.zeros((self.size, self.size), np.uint8)
        for indices in triangles:
            target = destination[indices].astype(np.float32)
            x0, y0 = np.floor(target.min(axis=0)).astype(int)
            x1, y1 = np.ceil(target.max(axis=0)).astype(int) + 1
            if x0 < 0 or y0 < 0 or x1 > self.size or y1 > self.size:
                raise InvalidGeometry("triangulo_fuera_de_lienzo")
            raster = np.zeros((y1 - y0, x1 - x0), np.uint8)
            cv2.fillConvexPoly(raster, np.rint(target - [x0, y0]).astype(np.int32), 1)
            rows, cols = np.nonzero(raster)
            ys, xs = rows + y0, cols + x0
            for offset, source in ((0, camera_pixels), (2, video_pixels)):
                matrix = cv2.getAffineTransform(target, source[indices].astype(np.float32))
                maps[offset, ys, xs] = matrix[0, 0] * xs + matrix[0, 1] * ys + matrix[0, 2]
                maps[offset + 1, ys, xs] = matrix[1, 0] * xs + matrix[1, 1] * ys + matrix[1, 2]
            covered[ys, xs] = 1
        return maps, covered

    def _skin_samples(self, destination, smooth):
        samples = np.zeros((self.size, self.size), np.uint8)
        # Mejillas altas, apartadas de labios, dientes, ojos y borde facial.
        # La selección es geométrica y no depende de un umbral de tono de piel.
        for ids in ((50, 101, 117), (280, 330, 346)):
            points = destination[[LANDMARK_IDS.index(i) for i in ids]]
            cv2.fillConvexPoly(samples, np.rint(points).astype(np.int32), 1)
        return samples * (smooth >= .99)

    def render(self, camera, camera_points, video, video_points, alpha, *, time_seconds=None):
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("Alpha debe ser finito y estar entre cero y uno.")
        for image in (camera, video):
            if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 3:
                raise ValueError("Se requieren imagenes BGR uint8 validas.")
        control_alpha = float(alpha)
        alpha = control_alpha * self.max_blend
        mask = np.zeros(camera.shape[:2], np.float32)
        geometry = None
        tone_info = {"status": "not_applied"}
        try:
            if camera_points is None or video_points is None:
                raise InvalidGeometry("sin_cara_webcam" if camera_points is None else "sin_cara_video")
            cam = align_face(camera_points, camera.shape, self.size)
            vid = align_face(video_points, video.shape, self.size)
            triangles = self._topology(cam.canonical, vid.canonical)
            destination = (1 - alpha) * cam.canonical + alpha * vid.canonical
            self._validate_mesh(cam.canonical, vid.canonical, destination, triangles, self.reference)
            geometry = transform_points(destination, cam.to_image)
            if (geometry < 1).any() or (geometry > [camera.shape[1] - 2, camera.shape[0] - 2]).any():
                raise InvalidGeometry("destino_fuera_de_imagen")
            if alpha == 0:
                self.tone_matcher.reset()
                result = MorphResult(camera.copy(), mask, "webcam", alpha, geometry, triangles,
                                     control_alpha=control_alpha, tone=tone_info)
            else:
                maps, covered = self._maps(destination, cam.pixels, vid.pixels, triangles)
                face_mask = np.zeros((self.size, self.size), np.uint8)
                cv2.fillPoly(face_mask, [np.rint(destination[:len(OVAL)]).astype(np.int32)], 1)
                # La distancia solo suaviza hacia dentro: nunca incorpora el
                # fondo del vídeo ni extiende la máscara fuera del óvalo.
                smooth = np.clip((cv2.distanceTransform(face_mask, cv2.DIST_L2, 5) - 1) / self.feather, 0, 1)
                if np.any((smooth > 0) & (covered == 0)):
                    raise InvalidGeometry("malla_incompleta")
                for offset, image in ((0, camera), (2, video)):
                    active = smooth > 0
                    if (np.any(maps[offset][active] < 0) or np.any(maps[offset + 1][active] < 0)
                            or np.any(maps[offset][active] > image.shape[1] - 1)
                            or np.any(maps[offset + 1][active] > image.shape[0] - 1)):
                        raise InvalidGeometry("muestreo_fuera_de_imagen")
                webcam_texture = cv2.remap(camera, maps[0], maps[1], cv2.INTER_LINEAR)
                video_texture = cv2.remap(video, maps[2], maps[3], cv2.INTER_LINEAR)
                if self.color_match:
                    video_texture, tone_info = self.tone_matcher.correct(
                        video_texture, webcam_texture, self._skin_samples(destination, smooth), time_seconds)
                texture = (1 - alpha) * webcam_texture.astype(np.float32) + alpha * video_texture.astype(np.float32)
                dimensions = (camera.shape[1], camera.shape[0])
                mask = cv2.warpAffine(smooth, cam.to_image, dimensions, flags=cv2.INTER_LINEAR)
                # Color premultiplicado para evitar halos negros en la vuelta.
                foreground = cv2.warpAffine(texture * smooth[..., None], cam.to_image, dimensions, flags=cv2.INTER_LINEAR)
                output = np.rint(np.clip(foreground + camera * (1 - mask[..., None]), 0, 255)).astype(np.uint8)
                if self.seamless:
                    raw_face = np.rint(np.clip(foreground / np.maximum(mask[...,None], 1e-5), 0, 255)).astype(np.uint8)
                    output, boundary_info = blend_boundary(raw_face, camera, mask)
                    tone_info["boundary"] = boundary_info
                result = MorphResult(output, mask, "morphing", alpha, geometry, triangles,
                                     control_alpha=control_alpha, tone=tone_info)
        except InvalidGeometry as error:
            self.tone_matcher.reset()
            result = MorphResult(camera.copy(), mask * 0, str(error), alpha,
                                 control_alpha=control_alpha, tone=tone_info)
        self.counts[result.status] = self.counts.get(result.status, 0) + 1
        return result

    def summary(self):
        return {"canvas_size": self.size, "landmarks": len(LANDMARK_IDS),
                "triangles": len(self.triangles) if self.triangles is not None else 0,
                "topology_initializations": int(self.triangles is not None),
                "feather_pixels": self.feather, "min_triangle_area": self.min_area,
                "schema": 2, "max_blend": self.max_blend, "color_match": self.color_match,
                "seamless": self.seamless,
                "color_tau_seconds": self.tone_matcher.tau,
                "frames_by_status": dict(self.counts)}
