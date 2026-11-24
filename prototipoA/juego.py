"""
GESTO GALÁCTICO: juego de naves controlado con gestos de la cara.

Los controles con la cara los detecta gestos.py con los modelos entrenados:
    - mover la cabeza -> mover la nave
    - abrir la boca -> disparar (ráfaga mientras está abierta)
    - sonreír -> turbo (más velocidad y puntos x2)
    - guiñar un ojo -> esquive rápido hacia ese lado
    - cerrar los ojos más de 0,6 s -> el tiempo se congela y se carga la bomba; al abrirlos explota,
      más grande cuanto más tiempo se hayan tenido cerrados

Los controles con el teclado:
    - flechas -> mover la nave
    - ESPACIO -> disparo
    - SHIFT -> turbo
    - Q / E -> esquive
    - B mantenida -> bomba
    - P -> pausa
    - C -> recentrar la cabeza
    - L -> ver landmarks
    - ESC -> salir

Partes:
  1. HiloGestos ejecuta en segundo plano la cámara, MediaPipe y los modelos (gestos.py), para que
     el juego vaya a 60 FPS aunque el análisis de cada fotograma tarde más.
  2. Asteroide y Particula son los elementos del juego.
  3. Juego lleva la partida: entrada (cara + teclado), actualización, dibujo y bucle principal.

Los gráficos se dibujan con primitivas de pygame y los sonidos se generan en sonidos_juego.py.
"""
import math
import queue
import random
import threading
import time

import cv2
import pygame
import mediapipe as mp

import gestos as G
from sonidos_juego import Sonidos


# ----- Configuración -----
ANCHO, ALTO = 1000, 720
FPS = 60
SENSIBILIDAD_X, SENSIBILIDAD_Y = 1.4, 1.2
CADENCIA = 0.12 
T_ESQUIVE, DIST_ESQUIVE = 0.25, 200 # duración (s) y distancia (px) del esquive
RECARGA_BOMBA = 6.0
VIDAS = 3
ARCHIVO_RECORD = 'record.txt'

COLOR = {
    'fondo': (8, 8, 22), 'nave': (255, 105, 180), 'llama': (255, 160, 40), 'turbo': (120, 255, 255),
    'bala': (255, 90, 90), 'asteroide': (110, 100, 95), 'borde': (190, 175, 160), 'estrella': (255, 230, 80),
    'texto': (235, 235, 245), 'gris': (140, 140, 160), 'bomba': (190, 140, 255), 'ok': (90, 230, 120), 'inicio': (40, 120, 255),
}


# ----- Hilo de la cámara: MediaPipe + modelos -> controles -----
class HiloGestos(threading.Thread):
    """
    Ejecuta en segundo plano el mismo bucle que gestos.py y publica el último estado de los controles.
    Los eventos puntuales (guiños, bomba) se guardan en una cola para que el juego no pierda ninguno
    aunque lleguen entre dos de sus fotogramas.
    """

    def __init__(self):
        super().__init__(daemon=True) # el hilo se cierra solo al salir del juego
        self.lock = threading.Lock()
        self.eventos = queue.Queue()
        self.ctrl = None
        self.vista = None
        self.error = None
        self.listo = False
        self.activo = True
        self._latest = None
        self._recentrar = False
        self.landmarks_visibles = False

    def _on_result(self, result, output_image, timestamp_ms):
        """Callback de MediaPipe: guarda el resultado junto con la imagen analizada."""
        self._latest = (result, output_image.numpy_view().copy(), timestamp_ms)

    def recentra(self):
        """Pide tomar la posición actual de la cabeza como centro (se aplica en el hilo)."""
        self._recentrar = True

    def run(self):
        """Bucle del hilo: captura, MediaPipe, modelos y lógica temporal en cada fotograma."""
        try:
            det = G.DetectorGestos()
            estado = G.EstadoGestos()
            opciones = G.FaceLandmarkerOptions(
                base_options=G.BaseOptions(model_asset_path=G.MODELO_FACE),
                running_mode=G.VisionRunningMode.LIVE_STREAM,
                num_faces=1,
                result_callback=self._on_result,
            )

            cap = G.abre_camara()
            if not cap.isOpened():
                raise RuntimeError(f'No se pudo abrir la cámara {G.CAMARA} (cámbiala en gestos.py)')

            ts_prev, ts_ultimo = None, 0

            with G.FaceLandmarker.create_from_options(opciones) as landmarker:
                self.listo = True
                while self.activo:
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.01)
                        continue

                    frame = cv2.flip(frame, 1)
                    ts = max(int(time.time() * 1000), ts_ultimo + 1) # MediaPipe exige marcas de tiempo crecientes
                    ts_ultimo = ts
                    landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)

                    lat = self._latest
                    if lat is None or lat[2] == ts_prev:
                        continue

                    result, img_rgb, ts_prev = lat
                    analisis = None
                    if result.face_landmarks:
                        analisis = det.analiza(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), result.face_landmarks[0])

                    if self._recentrar:
                        estado.recentra()
                        self._recentrar = False

                    ctrl = estado.actualiza(analisis, time.time())
                    for ev in ctrl['eventos']:
                        self.eventos.put(ev)

                    vista = self._dibuja_vista(img_rgb, analisis, ctrl, self.landmarks_visibles)
                    with self.lock:
                        self.ctrl = dict(ctrl, p=dict(ctrl['p']), activo=dict(ctrl['activo']))
                        self.vista = vista
            cap.release()
        except Exception as e: 
            self.error = str(e)

    @staticmethod
    def _dibuja_vista(img_rgb, analisis, ctrl, con_landmarks=True):
        """Miniatura de 240x180 con las cajas de los recortes (en rojo si el gesto está activo) y los landmarks."""
        img = cv2.resize(img_rgb, (240, 180), interpolation=cv2.INTER_AREA)

        if analisis is not None:
            k = 240 / img_rgb.shape[1] # escala a la miniatura
            if con_landmarks:
                G.dibuja_landmarks(img, analisis['landmarks'], radio=0, rgb=True)

            for nombre in ['ojo_izq', 'ojo_der', 'boca']:
                (cx, cy), lado, ang = analisis['cajas'][nombre]
                if nombre == 'boca':
                    activo = ctrl['disparo']
                else:
                    activo = ctrl['activo'][nombre]

                caja = cv2.boxPoints(((float(cx) * k, float(cy) * k), (lado * k, lado * k), ang)).astype(int)
                if activo:
                    color = (255, 80, 80)
                else:
                    color = (220, 220, 220)
                cv2.polylines(img, [caja], True, color, 1)
        return img

    def lee(self):
        """Devuelve (controles, miniatura de la cámara) más recientes."""
        with self.lock:
            return self.ctrl, self.vista


# ----- Elementos del juego -----
class Asteroide:
    """Polígono irregular que cae girando; los grandes necesitan más impactos."""

    def __init__(self, dificultad, x=None, y=None, radio=None):
        self.r = radio or random.choice([22, 30, 42, 55])

        if x is not None:
            self.x = x
        else:
            self.x = random.uniform(self.r, ANCHO - self.r)
        if y is not None:
            self.y = y
        else:
            self.y = -self.r

        self.vy = random.uniform(110, 220) * (1 + 0.08 * dificultad) # caen más rápido con el tiempo
        self.vx = random.uniform(-40, 40)
        self.ang, self.vang = 0.0, random.uniform(-1.5, 1.5)

        if self.r < 30:
            self.vida = 1
        elif self.r < 50:
            self.vida = 2
        else:
            self.vida = 3

        # n vértices en círculo con un radio aleatorio cada uno
        n = random.randint(9, 13)
        self.forma = []
        for k in range(n):
            self.forma.append((math.cos(2 * math.pi * k / n), math.sin(2 * math.pi * k / n),
                               random.uniform(0.72, 1.1)))

    def mueve(self, dt, vel):
        """Avanza dt segundos; vel multiplica la caída (turbo)."""
        self.x += self.vx * dt
        self.y += self.vy * dt * vel
        self.ang += self.vang * dt

    def puntos(self):
        """Vértices del polígono girados según el ángulo actual."""
        c, s = math.cos(self.ang), math.sin(self.ang)
        vertices = []

        for cx, cy, f in self.forma:
            vertices.append((self.x + self.r * f * (cx * c - cy * s), self.y + self.r * f * (cx * s + cy * c)))
        return vertices

    def dibuja(self, sup):
        p = self.puntos()
        pygame.draw.polygon(sup, COLOR['asteroide'], p)
        pygame.draw.polygon(sup, COLOR['borde'], p, 2)


class Particula:
    """Chispa de las explosiones: sale en una dirección al azar, se frena y se apaga."""

    def __init__(self, x, y, color, vel=250, vida=0.7, tam=3):
        a, v = random.uniform(0, 2 * math.pi), random.uniform(0.2, 1) * vel
        self.x, self.y, self.vx, self.vy = x, y, math.cos(a) * v, math.sin(a) * v
        self.vida = self.vida0 = random.uniform(0.5, 1) * vida
        self.color, self.tam = color, tam

    def mueve(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vx *= 0.97 # rozamiento
        self.vy *= 0.97
        self.vida -= dt

    def dibuja(self, sup):
        k = max(0.0, self.vida / self.vida0)
        color = []

        for c in self.color:
            color.append(int(c * k))
        pygame.draw.circle(sup, tuple(color), (int(self.x), int(self.y)), max(1, int(self.tam * k + 0.5)))


# ----- Partida -----
class Juego:
    """Partida de GESTO GALÁCTICO: estados (menú, jugando, game over), lógica y dibujo."""

    def __init__(self):
        pygame.init()
        pygame.display.set_caption('GESTO GALÁCTICO')
        self.pantalla = pygame.display.set_mode((ANCHO, ALTO))
        self.lienzo = pygame.Surface((ANCHO, ALTO))
        self.reloj = pygame.time.Clock()
        self.fuente = pygame.font.SysFont('consolas', 20)
        self.fuente_media = pygame.font.SysFont('consolas', 30, bold=True)
        self.fuente_grande = pygame.font.SysFont('consolas', 56, bold=True)
        self.fuente_record = pygame.font.SysFont('consolas', 42, bold=True)
        self.fuente_peque = pygame.font.SysFont('consolas', 16, bold=True)
        self.snd = Sonidos()
        self.gestos = HiloGestos()
        self.gestos.start()
        self.record = self._lee_record()

        # Estrellas del fondo
        self.estrellas = []
        for _ in range(140):
            self.estrellas.append([random.uniform(0, ANCHO), random.uniform(0, ALTO), random.choice([1, 2, 3])])

        self.estado = 'menu'
        self.t_boca_menu = 0.0
        self.t_espera = 0.0
        self.carga_teclado = 0.0
        self.pausa = False
        self.reinicia()

    def reinicia(self):
        """Pone una partida nueva."""
        self.x, self.y = ANCHO / 2, ALTO - 110
        self.balas, self.asteroides, self.particulas, self.bonus, self.ondas = [], [], [], [], []
        self.puntos, self.vidas, self.t_juego = 0, VIDAS, 0.0
        self.t_disparo = 0.0
        self.t_spawn, self.t_bonus = 0.0, 8.0
        self.invulnerable = 1.5
        self.esquive = None
        self.recarga_bomba = 0.0
        self.carga = 0.0 
        self.t_tick_carga = 0.0
        self.temblor = 0.0
        self.turbo_previo = False
        self.t_sin_cara = 0.0

    def _lee_record(self):
        try:
            with open(ARCHIVO_RECORD) as f:
                return int(f.read().strip() or 0)
        except (OSError, ValueError):
            return 0

    def _guarda_record(self):
        if self.puntos > self.record:
            self.record = self.puntos
            try:
                with open(ARCHIVO_RECORD, 'w') as f:
                    f.write(str(self.record))
            except OSError:
                pass

    # ----- Entrada: cara + teclado -----
    def entrada(self, dt):
        """Combina los controles de la cara y del teclado en un único diccionario."""
        ctrl, _ = self.gestos.lee()
        teclas = pygame.key.get_pressed()
        hay_cara = ctrl is not None and ctrl['cara']

        if hay_cara:
            mover_x, mover_y, carga = ctrl['mover_x'], ctrl['mover_y'], ctrl['carga_bomba']
        else:
            mover_x, mover_y, carga = None, None, 0.0

        c = {
            'cara': hay_cara,
            'mover_x': mover_x,
            'mover_y': mover_y,
            'disparo': (hay_cara and ctrl['disparo']) or teclas[pygame.K_SPACE],
            'turbo': (hay_cara and ctrl['turbo']) or teclas[pygame.K_LSHIFT] or teclas[pygame.K_RSHIFT],
            'carga': carga,
            'teclado_x': teclas[pygame.K_RIGHT] - teclas[pygame.K_LEFT],
            'teclado_y': teclas[pygame.K_DOWN] - teclas[pygame.K_UP],
            'eventos': [],
        }

        if teclas[pygame.K_b]:
            self.carga_teclado = min(1.0, self.carga_teclado + dt / G.T_CARGA_MAX)
            c['carga'] = max(c['carga'], self.carga_teclado)
        elif self.carga_teclado > 0:
            c['eventos'].append('bomba')
            self.carga = max(self.carga, self.carga_teclado)
            self.carga_teclado = 0.0

        while not self.gestos.eventos.empty():
            c['eventos'].append(self.gestos.eventos.get_nowait())
        return c

    # ----- Actualización -----
    def actualiza(self, dt, c):
        """Avanza la partida dt segundos según los controles c."""
        
        # Bomba: mientras se carga  el tiempo se congela, porque el jugador no ve la pantalla
        bomba_lista = self.recarga_bomba <= 0

        if 'bomba' in c['eventos'] and bomba_lista:
            self.carga = max(self.carga, c['carga'], 0.1)
            self.explota_bomba()
            self.carga = 0.0
        elif c['carga'] > 0 and bomba_lista:
            self.carga = max(self.carga, c['carga'])
            self.t_tick_carga -= dt
            if self.t_tick_carga <= 0:
                self.snd.play('carga', self.carga)
                self.t_tick_carga = 0.15
            self.snd.volumen_musica(0.15)
            return

        self.snd.volumen_musica(0.5)
        self.carga = 0.0

        turbo = c['turbo']
        if turbo:
            vel = 1.8
        else:
            vel = 1.0

        if turbo and not self.turbo_previo:
            self.snd.play('turbo')
        self.turbo_previo = turbo

        self.t_juego += dt
        dificultad = self.t_juego / 20 # sube un punto cada 20 s
        self.invulnerable = max(0.0, self.invulnerable - dt)
        self.recarga_bomba = max(0.0, self.recarga_bomba - dt)
        self.temblor = max(0.0, self.temblor - dt)

        # Movimiento de la nave
        if c['mover_x'] is not None and c['teclado_x'] == 0 and c['teclado_y'] == 0:
            tx = ANCHO / 2 + c['mover_x'] * SENSIBILIDAD_X * (ANCHO / 2 - 50)
            ty = ALTO - 150 + c['mover_y'] * SENSIBILIDAD_Y * 120
            self.x += (tx - self.x) * min(1.0, dt * 10)
            self.y += (ty - self.y) * min(1.0, dt * 10)
        else:
            self.x += c['teclado_x'] * 520 * dt
            self.y += c['teclado_y'] * 400 * dt

        # Esquive: desplazamiento rápido sin recibir daño
        for ev in c['eventos']:
            if ev in ('guiño_izq', 'guiño_der'):
                if ev == 'guiño_izq':
                    direccion = -1
                else:
                    direccion = 1
                self.esquive = (direccion, T_ESQUIVE)
                self.invulnerable = max(self.invulnerable, T_ESQUIVE + 0.15)
                self.snd.play('esquive')

        teclas = pygame.key.get_pressed()
        if self.esquive is None and (teclas[pygame.K_q] or teclas[pygame.K_e]):
            if teclas[pygame.K_q]:
                direccion = -1
            else:
                direccion = 1
            self.esquive = (direccion, T_ESQUIVE)
            self.invulnerable = max(self.invulnerable, T_ESQUIVE + 0.15)
            self.snd.play('esquive')

        if self.esquive is not None:
            d, t = self.esquive
            self.x += d * DIST_ESQUIVE / T_ESQUIVE * dt
            self.particulas.append(Particula(self.x - d * 20, self.y, COLOR['turbo'], vel=60, vida=0.3))
            if t - dt > 0:
                self.esquive = (d, t - dt)
            else:
                self.esquive = None

        self.x = min(max(self.x, 30), ANCHO - 30)
        self.y = min(max(self.y, ALTO * 0.45), ALTO - 40)

        # Disparo
        self.t_disparo -= dt
        if c['disparo'] and self.t_disparo <= 0:
            self.balas.append([self.x, self.y - 28])
            self.t_disparo = CADENCIA
            self.snd.play('laser')

        for b in self.balas:
            b[1] -= 900 * dt

        balas = []
        for b in self.balas:
            if b[1] > -20:
                balas.append(b)
        self.balas = balas

        # Aparición de asteroides y estrellas de bonus
        self.t_spawn -= dt * vel
        if self.t_spawn <= 0:
            self.asteroides.append(Asteroide(dificultad))
            self.t_spawn = max(0.25, 1.1 - 0.06 * dificultad) * random.uniform(0.6, 1.3)

        self.t_bonus -= dt
        if self.t_bonus <= 0:
            self.bonus.append([random.uniform(40, ANCHO - 40), -20])
            self.t_bonus = random.uniform(7, 12)

        for a in self.asteroides:
            a.mueve(dt, vel)
        for e in self.bonus:
            e[1] += 160 * dt * vel

        if turbo:
            vel_estrellas = 2
        else:
            vel_estrellas = 1
        for s in self.estrellas:
            s[1] += s[2] * 40 * dt * vel * vel_estrellas
            if s[1] > ALTO:
                s[0], s[1] = random.uniform(0, ANCHO), 0

        # Colisiones bala-asteroide
        if turbo:
            multiplicador = 2
        else:
            multiplicador = 1
        nuevos = []

        for a in self.asteroides:
            for b in self.balas:
                if (a.x - b[0]) ** 2 + (a.y - b[1]) ** 2 < a.r ** 2:
                    for _ in range(4):
                        self.particulas.append(Particula(b[0], b[1], COLOR['bala'], 120, 0.3, 2))
                    b[1] = -100
                    a.vida -= 1
                    if a.vida <= 0:
                        self.puntos += int(10 * a.r / 10) * multiplicador
                        self.explosion(a.x, a.y, a.r)
                        if a.r >= 42: # los grandes se parten en dos
                            for _ in range(2):
                                nuevos.append(Asteroide(dificultad, a.x, a.y, 22))
                    break

        asteroides = []
        for a in self.asteroides:
            if a.vida > 0 and a.y < ALTO + 80:
                asteroides.append(a)
        self.asteroides = asteroides + nuevos

        # Colisiones con la nave: estrellas de bonus y asteroides
        for e in self.bonus:
            if (e[0] - self.x) ** 2 + (e[1] - self.y) ** 2 < 35 ** 2:
                for _ in range(20):
                    self.particulas.append(Particula(e[0], e[1], COLOR['estrella'], 150, 0.6))
                e[1] = ALTO + 100
                self.puntos += 100 * multiplicador
                self.snd.play('bonus')

        bonus = []
        for e in self.bonus:
            if e[1] < ALTO + 30:
                bonus.append(e)
        self.bonus = bonus

        if self.invulnerable <= 0:
            for a in self.asteroides:
                if (a.x - self.x) ** 2 + (a.y - self.y) ** 2 < (a.r * 0.85 + 16) ** 2:
                    self.golpe(a)
                    break

        # Partículas y ondas de la bomba
        for p in self.particulas:
            p.mueve(dt)

        particulas = []
        for p in self.particulas:
            if p.vida > 0:
                particulas.append(p)
        self.particulas = particulas

        for o in self.ondas:
            o[2] += dt

        ondas = []
        for o in self.ondas:
            if o[2] < 0.6:
                ondas.append(o)
        self.ondas = ondas

    def explosion(self, x, y, r):
        """Explosión de un asteroide: partículas proporcionales a su tamaño y sonido."""
        for _ in range(int(r)):
            color = random.choice([(255, 200, 80), (255, 120, 40), (200, 200, 200)])
            self.particulas.append(Particula(x, y, color, vel=r * 6, vida=0.8, tam=3))
        self.snd.play('explosion')

    def explota_bomba(self):
        """Destruye los asteroides dentro de un radio que crece con la carga (de 160 a 600 px)."""
        radio = 160 + 440 * self.carga
        self.ondas.append([self.x, self.y, 0.0, radio])

        destruidos = []
        for a in self.asteroides:
            if (a.x - self.x) ** 2 + (a.y - self.y) ** 2 < (radio + a.r) ** 2:
                destruidos.append(a)

        for a in destruidos:
            self.puntos += int(a.r)
            for _ in range(int(a.r / 2)):
                self.particulas.append(Particula(a.x, a.y, COLOR['bomba'], vel=a.r * 5, vida=0.9))

        asteroides = []
        for a in self.asteroides:
            if a not in destruidos:
                asteroides.append(a)
        self.asteroides = asteroides

        self.recarga_bomba = RECARGA_BOMBA
        self.temblor = 0.6
        self.snd.play('bomba')

    def golpe(self, a):
        """Un asteroide choca con la nave: se pierde una vida y, sin vidas, se acaba la partida."""
        self.vidas -= 1
        self.invulnerable = 2.0
        self.temblor = 0.4
        self.explosion(a.x, a.y, a.r)
        a.vida = 0
        self.asteroides.remove(a)
        self.snd.play('golpe')

        if self.vidas <= 0:
            self.estado = 'gameover'
            self.t_boca_menu = 0.0
            self.t_espera = 2.0
            self._guarda_record()
            self.snd.play('game_over')

    # ----- Dibujo -----
    def texto(self, sup, t, pos, fuente=None, color=None, centro=False):
        """Escribe un texto en la superficie, alineado arriba a la izquierda o centrado en pos."""
        img = (fuente or self.fuente).render(t, True, color or COLOR['texto'])
        if centro:
            r = img.get_rect(center=pos)
        else:
            r = img.get_rect(topleft=pos)
        sup.blit(img, r)

    def dibuja_nave(self, sup, turbo):
        """Nave con su llama; parpadea mientras es invulnerable."""
        if self.invulnerable > 0 and int(self.invulnerable * 12) % 2 == 0:
            return

        x, y = self.x, self.y
        largo = random.uniform(18, 26)
        if turbo:
            largo *= 1.8
            color_llama = COLOR['turbo']
        else:
            color_llama = COLOR['llama']

        pygame.draw.polygon(sup, color_llama, [(x - 8, y + 14), (x + 8, y + 14), (x, y + 14 + largo)])
        pygame.draw.polygon(sup, COLOR['nave'], [(x, y - 26), (x - 22, y + 16), (x, y + 8), (x + 22, y + 16)])
        pygame.draw.polygon(sup, (255, 255, 255), [(x, y - 26), (x - 22, y + 16), (x, y + 8), (x + 22, y + 16)], 2)
        pygame.draw.circle(sup, (255, 210, 235), (int(x), int(y - 4)), 5)

    def dibuja_mundo(self, c):
        """Fondo de estrellas, bonus, asteroides, balas, partículas, ondas de la bomba y nave."""
        sup = self.lienzo
        sup.fill(COLOR['fondo'])

        colores = {1: (70, 70, 100), 2: (140, 140, 170), 3: (230, 230, 255)}
        for x, y, capa in self.estrellas:
            pygame.draw.circle(sup, colores[capa], (int(x), int(y)), capa // 2 + 1)

        for e in self.bonus:
            pts = []
            for k in range(10):
                if k % 2 == 0:
                    radio = 14
                else:
                    radio = 6
                pts.append((e[0] + radio * math.cos(math.pi * k / 5 - math.pi / 2),
                            e[1] + radio * math.sin(math.pi * k / 5 - math.pi / 2)))
            pygame.draw.polygon(sup, COLOR['estrella'], pts)

        for a in self.asteroides:
            a.dibuja(sup)
        for b in self.balas:
            pygame.draw.line(sup, COLOR['bala'], (b[0], b[1]), (b[0], b[1] + 14), 4)
        for p in self.particulas:
            p.dibuja(sup)

        for x, y, t, radio in self.ondas:
            r = int(radio * min(1.0, t / 0.35))
            pygame.draw.circle(sup, COLOR['bomba'], (int(x), int(y)), max(2, r), max(1, int(10 * (1 - t / 0.6))))

        if self.estado == 'jugando':
            self.dibuja_nave(sup, c['turbo'])

    def dibuja_hud(self, c, vista, solo_camara=False):
        """Marcador, vidas, barra de la bomba y cámara (en el menú y el game over, solo la cámara)."""
        sup = self.pantalla
        self.dibuja_camara(vista)
        if solo_camara:
            return

        self.texto(sup, f'PUNTOS {self.puntos}', (16, 12), self.fuente_media)
        if c['turbo']:
            self.texto(sup, 'TURBO x2', (16, 48), self.fuente_media, COLOR['turbo'])

        for i in range(self.vidas):
            x = 30 + 34 * i
            pygame.draw.polygon(sup, COLOR['nave'], [(x, ALTO - 50), (x - 12, ALTO - 28), (x + 12, ALTO - 28)])

        # Barra de la bomba
        x0, y0, w = ANCHO // 2 - 150, ALTO - 36, 300
        pygame.draw.rect(sup, (60, 60, 80), (x0, y0, w, 16), border_radius=6)

        if self.recarga_bomba > 0:
            k = 1 - self.recarga_bomba / RECARGA_BOMBA
            pygame.draw.rect(sup, COLOR['gris'], (x0, y0, int(w * k), 16), border_radius=6)
            self.texto(sup, 'bomba recargando', (ANCHO // 2, y0 - 14), centro=True, color=COLOR['gris'])
        else:
            pygame.draw.rect(sup, COLOR['bomba'], (x0, y0, w, 16), border_radius=6)
            self.texto(sup, 'BOMBA LISTA: cierra los ojos', (ANCHO // 2, y0 - 14), centro=True, color=COLOR['bomba'])

    def dibuja_camara(self, vista):
        """Imagen de la cámara con los recortes marcados y el estado de cada gesto."""
        sup = self.pantalla

        if vista is not None:
            surf = pygame.image.frombuffer(vista.tobytes(), (vista.shape[1], vista.shape[0]), 'RGB')
            sup.blit(surf, (ANCHO - 256, 12))
            pygame.draw.rect(sup, COLOR['gris'], (ANCHO - 257, 11, 242, 182), 1)

            ctrl, _ = self.gestos.lee()
            if ctrl is not None:
                indicadores = [('OJO IZQ', ctrl['activo']['ojo_izq']), ('OJO DER', ctrl['activo']['ojo_der']),
                               ('BOCA', ctrl['disparo']), ('SONRISA', ctrl['turbo'])]

                for k, (nombre, on) in enumerate(indicadores): # rejilla de 2 x 2 bajo la cámara
                    r = pygame.Rect(ANCHO - 256 + (k % 2) * 121, 198 + (k // 2) * 26, 119, 23)
                    if on:
                        fondo, color_texto = (255, 255, 255), (0, 0, 0)
                    else:
                        fondo, color_texto = (40, 40, 55), COLOR['gris']
                    pygame.draw.rect(sup, fondo, r, border_radius=5)
                    img = self.fuente_peque.render(nombre, True, color_texto)
                    sup.blit(img, img.get_rect(center=r.center))
        elif self.gestos.error:
            self.texto(sup, 'Sin cámara: modo teclado', (ANCHO - 290, 16), color=COLOR['bala'])
        elif not self.gestos.listo:
            self.texto(sup, 'Iniciando cámara...', (ANCHO - 250, 16), color=COLOR['texto'])

    def dibuja_carga(self):
        """Pantalla congelada mientras se carga la bomba, con el radio que tendrá la explosión."""
        capa = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA)
        capa.fill((0, 0, 30, 150))

        radio = int(160 + 440 * self.carga)
        pygame.draw.circle(capa, (*COLOR['bomba'], 70), (int(self.x), int(self.y)), radio)
        pygame.draw.circle(capa, (210, 170, 255, 220), (int(self.x), int(self.y)), radio, 3)
        self.pantalla.blit(capa, (0, 0))

        self.texto(self.pantalla, f'CARGANDO BOMBA {int(100 * self.carga)} %', (ANCHO // 2, ALTO // 2 - 40),
                   self.fuente_media, COLOR['bomba'], centro=True)
        self.texto(self.pantalla, 'abre los ojos para lanzarla', (ANCHO // 2, ALTO // 2), centro=True)

    def dibuja_menu(self, titulo, lineas):
        """Pantalla de menú o de game over: título, líneas de texto y progreso de 'abre la boca'."""
        capa = pygame.Surface((ANCHO, ALTO), pygame.SRCALPHA)
        capa.fill((0, 0, 0, 160))
        self.pantalla.blit(capa, (0, 0))
        self.texto(self.pantalla, titulo, (ANCHO // 2, 170), self.fuente_grande, COLOR['nave'], centro=True)

        for i, (t, color) in enumerate(lineas):
            self.texto(self.pantalla, t, (ANCHO // 2, 260 + 34 * i), color=color or COLOR['texto'], centro=True)

        if self.t_boca_menu > 0:
            pygame.draw.rect(self.pantalla, COLOR['inicio'], (ANCHO // 2 - 100, ALTO - 120, int(200 * min(1, self.t_boca_menu / 0.6)), 10))

    # ----- Bucle principal -----
    def ejecuta(self):
        """Bucle del juego a 60 FPS: eventos, entrada, actualización según el estado y dibujo."""
        self.snd.inicia_musica()

        while True:
            dt = min(self.reloj.tick(FPS) / 1000, 0.05) 

            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    self.gestos.activo = False
                    pygame.quit()
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                    self.gestos.recentra()
                if e.type == pygame.KEYDOWN and e.key == pygame.K_l:
                    self.gestos.landmarks_visibles = not self.gestos.landmarks_visibles
                if e.type == pygame.KEYDOWN and e.key == pygame.K_p and self.estado == 'jugando':
                    self.pausa = not self.pausa

            c = self.entrada(dt)
            _, vista = self.gestos.lee()

            if self.estado in ('menu', 'gameover'):

                # Empieza manteniendo la boca abierta 0,6 s o con ESPACIO
                self.t_espera = max(0.0, self.t_espera - dt)
                if c['disparo'] and self.t_espera <= 0:
                    self.t_boca_menu = self.t_boca_menu + dt
                else:
                    self.t_boca_menu = 0.0

                if self.t_boca_menu >= 0.6:
                    self.reinicia()
                    self.estado = 'jugando'
                    self.gestos.recentra()
                    self.snd.play('inicio')

                for s in self.estrellas:
                    s[1] = (s[1] + s[2] * 30 * dt) % ALTO
            elif not self.pausa:
                # Pausa automática si se pierde la cara más de 1s
                if c['cara'] or vista is None:
                    self.t_sin_cara = 0.0
                else:
                    self.t_sin_cara = self.t_sin_cara + dt

                if self.t_sin_cara < 1.0:
                    self.actualiza(dt, c)

            # Temblor de pantalla tras un golpe o la bomba
            self.dibuja_mundo(c)
            if self.temblor > 0:
                off = (random.randint(-8, 8), random.randint(-8, 8))
            else:
                off = (0, 0)
            self.pantalla.fill((0, 0, 0))
            self.pantalla.blit(self.lienzo, off)

            if self.estado == 'jugando':
                if self.carga > 0 and self.recarga_bomba <= 0:
                    self.dibuja_carga()
                self.dibuja_hud(c, vista)
                if self.pausa:
                    self.texto(self.pantalla, 'PAUSA (P)', (ANCHO // 2, ALTO // 2), self.fuente_grande, centro=True)
                elif self.t_sin_cara >= 1.0:
                    self.texto(self.pantalla, 'No se ve tu cara', (ANCHO // 2, ALTO // 2), self.fuente_media,
                               COLOR['bala'], centro=True)
            elif self.estado == 'menu':
                self.dibuja_hud(c, vista, solo_camara=True)
                controles = [
                    ('Mueve la cabeza', 'mover la nave'),
                    ('Abre la boca', 'disparar'),
                    ('Sonríe', 'turbo (puntos x2)'),
                    ('Guiña un ojo', 'esquivar hacia ese lado'),
                    ('Cierra los ojos', 'congelar el tiempo y cargar la bomba'),
                ]
                # Todas las líneas con la misma longitud: con fuente monoespaciada empiezan y acaban alineadas
                largo = max(len(g) + len(a) for g, a in controles) + 6
                self.dibuja_menu('GESTO GALÁCTICO', [
                    ('Pilota la nave con la cara', COLOR['gris']),
                    ('', None),
                    *[(f"{g} {'.' * (largo - len(g) - len(a) - 2)} {a}", None) for g, a in controles],
                    ('', None),
                    ('Abre la boca (o pulsa ESPACIO) para empezar', COLOR['inicio']),
                    (f'Récord: {self.record}', COLOR['estrella']),
                ])
            else:
                self.dibuja_hud(c, vista, solo_camara=True)
                nuevo_record = self.puntos >= self.record and self.puntos > 0
                # Si hay récord se deja una línea libre arriba para escribirlo bajo GAME OVER
                hueco = [('', None)] if nuevo_record else []
                self.dibuja_menu('GAME OVER', [
                    *hueco,
                    (f'Puntos: {self.puntos}', COLOR['texto']),
                    (f'Récord: {self.record}', COLOR['estrella']),
                    ('', None),
                    ('Abre la boca (o pulsa ESPACIO) para jugar otra vez', COLOR['inicio']),
                ])
                if nuevo_record:
                    self.texto(self.pantalla, '¡NUEVO RÉCORD!', (ANCHO // 2, 240), self.fuente_record,
                               COLOR['estrella'], centro=True)
            pygame.display.flip()


if __name__ == '__main__':
    Juego().ejecuta()
