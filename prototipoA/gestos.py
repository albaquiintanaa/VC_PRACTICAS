"""
Detector de gestos faciales en directo: une los landmarks de MediaPipe con los tres modelos
entrenados en los cuadernos VC_P5_gestos_*.ipynb y convierte sus salidas en los controles del juego.

Abre una ventana de prueba con los recortes que ve cada modelo, las probabilidades y los gestos
detectados. 

* Teclas: ESC (salir), + / - (tamaño del recorte del ojo), C (recentrar la cabeza),
L (mostrar los landmarks) y R (grabar la cámara en grabaciones/, para analizarla después en
VC_P5_gestos_prueba_directo.ipynb). 

- juego.py importa este módulo para leer los controles.

En cada fotograma:
  1. MediaPipe localiza los 478 landmarks de la cara.
  2. Con ellos se recortan los dos ojos y la boca, girados para que queden horizontales.
  3. Cada recorte pasa por su modelo: ojos -> CNN; boca y sonrisa -> HOG + SVM.
  4. EstadoGestos convierte las probabilidades en controles:
    - cabeza (punta de la nariz) -> mover la nave     
    - boca abierta -> disparar
    - sonrisa -> turbo             
    - guiño -> esquive hacia ese lado
    - ojos cerrados más de 0,6 s -> cargar la bomba; al abrirlos, explota
"""
import os
import time

import cv2
import numpy as np
import mediapipe as mp

from utilidades_gestos import ModeloGesto


# ----- Configuración -----
CAMARA = 1
USAR_DSHOW = False
MODELO_FACE = 'face_landmarker.task'

# Boca: HOG + SVM con el recorte - variante con menos falsos disparos en la prueba en directo
MODELOS = {'ojos': 'modelos/ojos_cnn', 'boca': 'modelos/boca_hog', 'sonrisa': 'modelos/sonrisa_hog'}
CARPETA_GRABACIONES = 'grabaciones'

# Índices de landmarks de MediaPipe Face Mesh
LABIOS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
          291, 185, 40, 39, 37, 0, 267, 269, 270, 409] # contorno exterior de los labios
OJO_A = (33, 133) # comisuras de un ojo (exterior, interior)
OJO_B = (362, 263) # comisuras del otro ojo (interior, exterior)
NARIZ = 1 # punta de la nariz

CONTORNO_OJOS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                 263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
LABIOS_INTERIOR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
MOSTRAR_LANDMARKS = False

# Recortes
COMISURAS_BOCA = (61, 291)
MARGEN_BOCA = 1.4      
TAM_BOCA = 64
FACTOR_OJO = 0.95
TAM_OJO = 24 

# Lógica temporal
UMBRAL_ON, UMBRAL_OFF = 0.6, 0.4
UMBRAL_OJOS_SONRIENDO = 0.85
SUAVIZADO = 0.5
T_BOMBA = 0.6
T_CARGA_MAX = 1.5
T_GUINO = 0.2


# ----- Recortes a partir de los landmarks -----
def puntos(landmarks, indices, w, h):
    """Coordenadas en píxeles de los landmarks indicados (normalizados entre 0 y 1)."""
    coords = []

    for i in indices:
        coords.append([landmarks[i].x * w, landmarks[i].y * h])
    return np.array(coords, dtype=np.float32)


def recorta_boca_recta(img_bgr, landmarks, margen=MARGEN_BOCA, tam=TAM_BOCA):
    """Recorte cuadrado de la boca sin girar, igual que el usado para preparar los datasets."""
    h, w = img_bgr.shape[:2]
    p = puntos(landmarks, LABIOS, w, h)
    centro = p.mean(axis=0)
    lado = int(max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * margen)
    x0, y0 = int(centro[0] - lado / 2), int(centro[1] - lado / 2)

    pad = cv2.copyMakeBorder(img_bgr, lado, lado, lado, lado, cv2.BORDER_REPLICATE) # por si se sale de la imagen
    recorte = pad[y0 + lado:y0 + 2 * lado, x0 + lado:x0 + 2 * lado]
    return cv2.resize(recorte, (tam, tam), interpolation=cv2.INTER_AREA), (centro, float(lado), 0.0)


def recorta_boca(img_bgr, landmarks, margen=MARGEN_BOCA, tam=TAM_BOCA):
    """
    Recorte cuadrado de la boca, girado para que la línea entre las comisuras quede horizontal.
    Mismo encuadre que en los datasets, pero alineado con la boca.
    Devuelve el recorte y su caja (centro, lado, ángulo) para dibujarla.
    """
    h, w = img_bgr.shape[:2]
    p = puntos(landmarks, LABIOS, w, h)
    a, b = puntos(landmarks, COMISURAS_BOCA, w, h)
    angulo = float(np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])))
    centro = p.mean(axis=0)

    # Tamaño de los labios con la boca ya alineada
    R = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), angulo, 1.0)
    p_rot = (R[:, :2] @ (p - centro).T).T
    lado = max(8.0, float(max(np.ptp(p_rot[:, 0]), np.ptp(p_rot[:, 1]))) * margen)

    # Una sola transformación afín gira, escala a tam x tam y centra la boca
    M = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), angulo, tam / lado)
    M[:, 2] += np.array([tam / 2, tam / 2]) - centro
    recorte = cv2.warpAffine(img_bgr, M, (tam, tam), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return recorte, (centro, lado, angulo)


def apertura_labios(landmarks, w, h):
    """
    Apertura geométrica de la boca: distancia entre el centro de los labios dividida por la
    anchura de la boca. Etiqueta las grabaciones de prueba.
    """
    p13, p14, p61, p291 = puntos(landmarks, (13, 14, 61, 291), w, h)
    return float(np.linalg.norm(p13 - p14) / max(np.linalg.norm(p61 - p291), 1e-6))


def recorta_ojo(img_bgr, landmarks, comisuras, factor=FACTOR_OJO, tam=TAM_OJO):
    """
    Recorte cuadrado del ojo, girado para que la línea entre sus comisuras quede horizontal.
    Imita el encuadre de CEW (ojo centrado, horizontal y ajustado, con poca ceja)
    """
    h, w = img_bgr.shape[:2]
    a, b = puntos(landmarks, comisuras, w, h)
    centro = (a + b) / 2
    angulo = np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0]))

    if abs(angulo) > 90: # evita que el ojo quede boca abajo
        angulo -= 180 * np.sign(angulo)

    lado = max(8.0, float(np.linalg.norm(b - a)) * factor)
    M = cv2.getRotationMatrix2D((float(centro[0]), float(centro[1])), angulo, tam / lado)
    M[:, 2] += np.array([tam / 2, tam / 2]) - centro
    recorte = cv2.warpAffine(img_bgr, M, (tam, tam), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    return recorte, (centro, lado, angulo)


# ----- Detector: landmarks + modelos -> probabilidades -----
class DetectorGestos:
    """Carga los tres modelos y calcula, para un fotograma con cara, la probabilidad de cada gesto."""

    def __init__(self, rutas=MODELOS):
        self.modelos = {}
        for g, r in rutas.items():
            self.modelos[g] = ModeloGesto.carga(r)

        self.factor_ojo = FACTOR_OJO
        self.tiempos = {} # ms de la última predicción de cada modelo
        for g in rutas:
            self.tiempos[g] = 0.0

    def analiza(self, img_bgr, landmarks):
        """Recorta ojos y boca, aplica los modelos y devuelve probabilidades, recortes y cajas."""
        h, w = img_bgr.shape[:2]
        ojos = []

        for c in (OJO_A, OJO_B):
            ojos.append(recorta_ojo(img_bgr, landmarks, c, self.factor_ojo))

        # izq es el ojo que se ve a la izquierda, ya que está en efecto espejo la imagen
        ojos.sort(key=lambda o: o[1][0][0])

        boca, caja_boca = recorta_boca(img_bgr, landmarks)

        t0 = time.perf_counter()
        p_ojos = self.modelos['ojos'].prob_lote([ojos[0][0], ojos[1][0]]) # los dos ojos en un solo lote
        t1 = time.perf_counter()
        p_boca = self.modelos['boca'].prob(boca)
        t2 = time.perf_counter()
        p_sonrisa = self.modelos['sonrisa'].prob(boca) # mismo recorte que la boca
        t3 = time.perf_counter()
        self.tiempos = {'ojos': (t1 - t0) * 1000, 'boca': (t2 - t1) * 1000, 'sonrisa': (t3 - t2) * 1000}

        nariz = landmarks[NARIZ]
        return {
            'landmarks': landmarks,
            'ojo_izq': float(p_ojos[0]), 'ojo_der': float(p_ojos[1]),
            'boca': p_boca, 'sonrisa': p_sonrisa,
            'nariz': (nariz.x, nariz.y),
            'recortes': {'ojo_izq': ojos[0][0], 'ojo_der': ojos[1][0], 'boca': boca},
            'cajas': {'ojo_izq': ojos[0][1], 'ojo_der': ojos[1][1], 'boca': caja_boca},
        }


# ----- Lógica temporal: probabilidades -> controles del juego -----
class EstadoGestos:
    """
    Convierte las probabilidades de cada fotograma en estados y eventos del juego:
      - suaviza las probabilidades (media móvil exponencial) para que la salida no tiemble;
      - usa histéresis (UMBRAL_ON / UMBRAL_OFF) para que un valor cerca de 0,5 no cambie el estado
        en cada fotograma;
      - distingue parpadeo (< T_BOMBA), carga de la bomba (>= T_BOMBA) y guiño (un solo ojo cerrado
        durante al menos T_GUINO);
      - mientras se sonríe, sube el umbral de los ojos, ya que al sonreír se entrecierran.
    """

    def __init__(self):
        self.p = {} 
        self.activo = {'ojo_izq': False, 'ojo_der': False, 'boca': False, 'sonrisa': False}
        self.t_ambos = None   
        self.t_guino = None     
        self.lado_guino = None
        self.guino_emitido = False
        self.parpadeos = 0
        self.centro_nariz = None  

    def recentra(self):
        """Toma la posición actual de la cabeza como el centro en el siguiente fotograma."""
        self.centro_nariz = None

    def actualiza(self, probs, t):
        """
        probs: salida de DetectorGestos.analiza (None si no hay cara); t: tiempo en segundos.
        Devuelve un diccionario con los controles:
            mover_x, mover_y -> desplazamiento de la cabeza respecto al centro (aprox. -1..1)
            disparo, turbo -> estados continuos, e intensidad_turbo (0..1)
            carga_bomba -> 0..1 mientras se mantienen los ojos cerrados
            eventos -> eventos puntuales: 'guiño_izq', 'guiño_der', 'bomba', 'parpadeo'
        """
        eventos = []

        if probs is None:
            self.t_ambos = self.t_guino = None
            self.lado_guino = None 
            return {'cara': False, 'mover_x': 0.0, 'mover_y': 0.0, 'disparo': False, 'turbo': False,
                    'intensidad_turbo': 0.0, 'carga_bomba': 0.0, 'eventos': eventos, 'p': self.p,
                    'activo': self.activo, 'parpadeos': self.parpadeos}

        # Suavizado e histéresis
        for g in ['sonrisa', 'boca', 'ojo_izq', 'ojo_der']:
            previo = self.p.get(g, probs[g])
            self.p[g] = SUAVIZADO * previo + (1 - SUAVIZADO) * probs[g]

            umbral_on = UMBRAL_ON
            if g.startswith('ojo') and self.activo['sonrisa']:
                umbral_on = UMBRAL_OJOS_SONRIENDO

            if self.activo[g] and self.p[g] < UMBRAL_OFF:
                self.activo[g] = False
            elif not self.activo[g] and self.p[g] > umbral_on:
                self.activo[g] = True

        izq, der = self.activo['ojo_izq'], self.activo['ojo_der']

        # Los dos ojos cerrados - parpadeo o bomba
        carga = 0.0

        if izq and der:
            if self.t_ambos is None:
                self.t_ambos = t
            dur = t - self.t_ambos
            if dur >= T_BOMBA:
                carga = min(1.0, (dur - T_BOMBA) / T_CARGA_MAX + 0.1)
        elif self.t_ambos is not None:
            dur = t - self.t_ambos
            if dur >= T_BOMBA:
                eventos.append('bomba')
                carga = min(1.0, (dur - T_BOMBA) / T_CARGA_MAX + 0.1)
            else:
                eventos.append('parpadeo')
                self.parpadeos += 1
            self.t_ambos = None

        # Guiño - durante al menos T_GUINO
        if izq and not der:
            lado = 'izq'
        elif der and not izq:
            lado = 'der'
        else:
            lado = None

        if lado is None or lado != self.lado_guino:
            if lado:
                self.t_guino = t
            else:
                self.t_guino = None
            self.lado_guino = lado
            self.guino_emitido = False
        elif not self.guino_emitido and t - self.t_guino >= T_GUINO:
            eventos.append(f'guiño_{lado}')
            self.guino_emitido = True

        # Posición de la cabeza respecto a la de reposo
        nx, ny = probs['nariz']

        if self.centro_nariz is None:
            self.centro_nariz = (nx, ny)

        mover_x = float(np.clip((nx - self.centro_nariz[0]) / 0.15, -1, 1))
        mover_y = float(np.clip((ny - self.centro_nariz[1]) / 0.12, -1, 1))

        return {'cara': True, 'mover_x': mover_x, 'mover_y': mover_y,
                'disparo': self.activo['boca'], 'turbo': self.activo['sonrisa'],
                'intensidad_turbo': self.p['sonrisa'], 'carga_bomba': carga,
                'eventos': eventos, 'p': self.p, 'activo': self.activo, 'parpadeos': self.parpadeos}


# ----- Visualización de la prueba -----
def dibuja_landmarks(img, landmarks, radio=1, rgb=False):
    """
    Dibuja los 478 landmarks en tenue y resalta los que usa el programa: contorno de los ojos
    (verde), labios (rojo), centro de los labios 13-14 para la apertura (blanco) y punta de la nariz
    para mover la nave (amarillo).
    """
    h, w = img.shape[:2]

    def color(bgr):
        if rgb:
            return bgr[::-1]
        return bgr

    for lm in landmarks:
        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), radio, color((90, 140, 90)), -1)

    for indices, c in [(CONTORNO_OJOS, (80, 230, 80)), (LABIOS + LABIOS_INTERIOR, (80, 80, 255)),
                       ([13, 14], (255, 255, 255)), ([NARIZ], (0, 230, 255))]:
        for i in indices:
            cv2.circle(img, (int(landmarks[i].x * w), int(landmarks[i].y * h)), radio + 1, color(c), -1)


def dibuja_panel(frame, analisis, ctrl, det, fps, ultimos_eventos):
    """Devuelve la imagen de la cámara con las cajas de los recortes y, a su derecha, el panel de estado."""
    h, w = frame.shape[:2]
    panel = np.full((h, 300, 3), 30, np.uint8)
    blanco, verde, rojo, amarillo = (255, 255, 255), (80, 220, 80), (80, 80, 255), (0, 220, 255)

    def texto(t, y, color=blanco, escala=0.5, grosor=1):
        cv2.putText(panel, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, grosor, cv2.LINE_AA)

    texto(f'FPS: {fps:4.1f}', 20)
    texto(f'factor ojo: {det.factor_ojo:.2f}  (+/-)', 40)

    if analisis is None:
        texto('Sin cara', 80, rojo, 0.7, 2)
    else:
        # Recortes ampliados: lo que ve cada modelo
        y = 60
        for nombre, escala in [('ojo_izq', 4), ('ojo_der', 4), ('boca', 1.5)]:
            rec = analisis['recortes'][nombre]
            if nombre.startswith('ojo'): # en gris, como lo ve el modelo de ojos
                vista = cv2.cvtColor(cv2.equalizeHist(cv2.cvtColor(rec, cv2.COLOR_BGR2GRAY)), cv2.COLOR_GRAY2BGR)
            else:
                vista = rec
            vista = cv2.resize(vista, None, fx=escala, fy=escala, interpolation=cv2.INTER_NEAREST)

            if nombre == 'ojo_izq':
                x = 10
            elif nombre == 'ojo_der':
                x = 110
            else:
                x = 200
            panel[y:y + vista.shape[0], x:x + vista.shape[1]] = vista

        texto('ojo izq      ojo der      boca', 175, (180, 180, 180), 0.4)

        # Barras de probabilidad suavizada
        etiquetas = {'ojo_izq': 'ojo izq cerrado', 'ojo_der': 'ojo der cerrado', 'boca': 'boca abierta',
                     'sonrisa': 'sonrisa'}
        y = 200

        for g, et in etiquetas.items():
            p = ctrl['p'].get(g, 0)
            if ctrl['activo'][g]:
                color = verde
            else:
                color = (120, 120, 120)
            cv2.rectangle(panel, (10, y), (10 + int(180 * p), y + 14), color, -1)
            cv2.rectangle(panel, (10, y), (190, y + 14), (90, 90, 90), 1)
            texto(f'{et} {p:.2f}', y + 12, blanco, 0.4)
            y += 24

        texto(f"ms: ojos {det.tiempos['ojos']:.1f} | boca {det.tiempos['boca']:.1f} | "
              f"sonr {det.tiempos['sonrisa']:.1f}", y + 6, (180, 180, 180), 0.38)

        # Controles del juego
        y += 34
        texto(f"cabeza x {ctrl['mover_x']:+.2f}  y {ctrl['mover_y']:+.2f}", y)

        if ctrl['disparo']:
            cv2.putText(panel, 'DISPARO', (10, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rojo, 2, cv2.LINE_AA)
        if ctrl['turbo']:
            cv2.putText(panel, 'TURBO', (160, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, amarillo, 2, cv2.LINE_AA)
        if ctrl['carga_bomba'] > 0:
            cv2.rectangle(panel, (10, y + 36), (10 + int(280 * ctrl['carga_bomba']), y + 52), (0, 140, 255), -1)
            texto('BOMBA cargando', y + 66, (0, 140, 255), 0.5, 1)

        texto(f"parpadeos: {ctrl['parpadeos']}", y + 90, (180, 180, 180), 0.45)

        for i, (ev, _) in enumerate(ultimos_eventos[-4:]):
            texto(ev.upper(), y + 115 + 22 * i, amarillo, 0.6, 2)

        # Landmarks y cajas de los recortes sobre la imagen
        if MOSTRAR_LANDMARKS:
            dibuja_landmarks(frame, analisis['landmarks'])

        for nombre in ['ojo_izq', 'ojo_der', 'boca']:
            (cx, cy), lado, ang = analisis['cajas'][nombre]
            caja = cv2.boxPoints(((float(cx), float(cy)), (lado, lado), ang)).astype(int)
            if nombre == 'boca':
                if ctrl['disparo']:
                    color = rojo
                else:
                    color = (200, 200, 200)
            else:
                if ctrl['activo'][nombre]:
                    color = verde
                else:
                    color = (200, 200, 200)
            cv2.polylines(frame, [caja], True, color, 1)
    return np.hstack([frame, panel])


# ----- Bucle principal -----
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

latest = None


def on_result(result, output_image, timestamp_ms):
    """
    Callback de MediaPipe. Además del resultado guarda la imagen exacta sobre la que se calcularon
    los landmarks, para que los recortes coincidan con los puntos aunque la cara se mueva.
    """
    global latest
    latest = (result, output_image.numpy_view().copy(), timestamp_ms)


def abre_camara():
    """Abre la webcam configurada en CAMARA a 640x480."""
    if USAR_DSHOW:
        cap = cv2.VideoCapture(CAMARA, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAMARA)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def main():
    """Ventana de prueba del detector: captura, analiza, dibuja y atiende las teclas."""
    print('Cargando modelos...')
    det = DetectorGestos()
    estado = EstadoGestos()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELO_FACE),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_faces=1,
        result_callback=on_result,
    )

    print('ESC: salir | + / -: tamaño del recorte del ojo | C: recentrar la cabeza | L: landmarks | R: grabar / parar')
    cap = abre_camara()
    ultimos_eventos, t_prev, fps, ts_prev = [], time.time(), 0.0, None
    grabador = None

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1) # espejo
            if grabador is not None:
                grabador.write(frame)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            landmarker.detect_async(mp_image, int(time.time() * 1000))

            # Solo se analiza cuando MediaPipe ha devuelto un resultado nuevo
            analisis, ctrl = None, None

            if latest is not None and latest[2] != ts_prev:
                result, img_rgb, ts_prev = latest
                if result.face_landmarks:
                    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                    analisis = det.analiza(img_bgr, result.face_landmarks[0])
                ctrl = estado.actualiza(analisis, time.time())

                for ev in ctrl['eventos']:
                    ultimos_eventos.append((ev, time.time()))
                    print(f'{time.strftime("%H:%M:%S")}  {ev}')

                # Solo se muestran los eventos de los últimos 2 s
                recientes = []
                for ev, t_ev in ultimos_eventos:
                    if time.time() - t_ev < 2.0:
                        recientes.append((ev, t_ev))
                ultimos_eventos = recientes

                ahora = time.time()
                fps = 0.9 * fps + 0.1 / max(ahora - t_prev, 1e-3)
                t_prev = ahora

                vista = dibuja_panel(frame, analisis, ctrl, det, fps, ultimos_eventos)
                if grabador is not None:
                    cv2.circle(vista, (vista.shape[1] - 25, 25), 9, (0, 0, 255), -1)
                    cv2.putText(vista, 'REC', (vista.shape[1] - 75, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Gestos', vista)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 27:
                break
            elif tecla in (ord('+'), ord('=')):
                det.factor_ojo = min(3.0, det.factor_ojo + 0.05)
            elif tecla == ord('-'):
                det.factor_ojo = max(0.8, det.factor_ojo - 0.05)
            elif tecla in (ord('c'), ord('C')):
                estado.recentra()
            elif tecla in (ord('l'), ord('L')):
                global MOSTRAR_LANDMARKS
                MOSTRAR_LANDMARKS = not MOSTRAR_LANDMARKS
            elif tecla in (ord('r'), ord('R')):
                if grabador is None:
                    os.makedirs(CARPETA_GRABACIONES, exist_ok=True)
                    ruta = os.path.join(CARPETA_GRABACIONES, time.strftime('prueba_%Y%m%d_%H%M%S.mp4'))
                    grabador = cv2.VideoWriter(ruta, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                               (frame.shape[1], frame.shape[0]))
                    print('Grabando en', ruta)
                else:
                    grabador.release()
                    grabador = None
                    print('Grabación terminada')

    if grabador is not None:
        grabador.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
