"""
Sonidos y música de GESTO GALÁCTICO, generados por síntesis con numpy.

Uso (desde juego.py):
    snd = Sonidos()
    snd.play('laser') # efecto puntual
    snd.inicia_musica() # música en bucle

Cada efecto combina ondas cuadradas o senoidales, barridos de frecuencia y ruido,
multiplicados por una envolvente (subida rápida y caída exponencial).
"""
import os

import numpy as np
import pygame

FS = 44100 # frecuencia de muestreo (Hz)
CARPETA = os.path.join('assets', 'sonidos')


def _env(n, ataque=0.005, caida=None):
    """Envolvente: subida rápida y caída exponencial (caida en segundos)."""
    t = np.arange(n) / FS
    e = np.minimum(1, t / max(ataque, 1e-4))
    if caida:
        e *= np.exp(-t / caida)
    return e


def _a_sonido(x, volumen=0.5):
    """Convierte una señal entre -1 y 1 en un sonido estéreo de 16 bits de pygame."""
    x = np.clip(x * volumen, -1, 1)
    estereo = np.repeat((x * 32767).astype(np.int16)[:, None], 2, axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(estereo))


def _barrido(f0, f1, dur, forma='cuadrada'):
    """Tono cuya frecuencia va de f0 a f1 (exponencialmente) en `dur` segundos."""
    n = int(FS * dur)
    f = f0 * (f1 / f0) ** (np.arange(n) / n)
    fase = 2 * np.pi * np.cumsum(f) / FS
    return np.sign(np.sin(fase)) if forma == 'cuadrada' else np.sin(fase)


def _ruido(dur, suavizado=1):
    """Ruido blanco de `dur` segundos; con suavizado > 1 se vuelve más grave (para explosiones)."""
    x = np.random.default_rng(0).uniform(-1, 1, int(FS * dur))
    if suavizado > 1: # filtro paso bajo sencillo (media móvil)
        x = np.convolve(x, np.ones(suavizado) / suavizado, mode='same')
    return x


def sintetiza():
    """Genera todos los efectos del juego y los devuelve en un diccionario {nombre: sonido}."""
    s = {}

    # Láser: barrido descendente de onda cuadrada
    x = _barrido(1400, 300, 0.12)
    s['laser'] = _a_sonido(x * _env(len(x), caida=0.05), 0.18)
    
    # Explosión pequeña (asteroide): ruido con caída rápida
    x = _ruido(0.35, 6)
    s['explosion'] = _a_sonido(x * _env(len(x), caida=0.09), 0.6)
    
    # Bomba: ruido grave largo + barrido descendente
    x = _ruido(1.4, 40) * 3 + 0.5 * _barrido(160, 30, 1.4, 'seno')
    s['bomba'] = _a_sonido(x * _env(len(x), caida=0.45), 0.7)
   
    # Golpe a la nave
    x = _barrido(400, 60, 0.4) * 0.6 + _ruido(0.4, 3) * 0.4
    s['golpe'] = _a_sonido(x * _env(len(x), caida=0.15), 0.45)
    
    # Turbo: "whoosh" de ruido filtrado que sube y baja
    x = _ruido(0.6, 25) * 4
    s['turbo'] = _a_sonido(x * np.sin(np.linspace(0, np.pi, len(x))), 0.35)
   
    # Esquive (guiño): barrido ascendente corto
    x = _barrido(300, 1200, 0.1, 'seno')
    s['esquive'] = _a_sonido(x * _env(len(x), caida=0.06), 0.35)
    
    # Bonus - estrella
    notas = [880, 1109, 1319, 1760]
    x = np.concatenate([np.sign(np.sin(2 * np.pi * f * np.arange(int(FS * 0.06)) / FS)) for f in notas])
    s['bonus'] = _a_sonido(x * _env(len(x), caida=0.2), 0.15)
    
    # Carga de la bomba: 12 tonos cada vez más agudos (se reproduce uno según el nivel de carga)
    s['carga'] = []
    for i in range(12):
        f = 220 * 2 ** (i / 6)
        x = _barrido(f, f * 1.05, 0.12, 'seno') + 0.3 * _barrido(2 * f, 2.1 * f, 0.12, 'seno')
        s['carga'].append(_a_sonido(x * _env(len(x), ataque=0.01), 0.25))
    
    # Game over: cuatro notas descendentes
    x = np.concatenate([_barrido(f, f * 0.97, 0.28) * _env(int(FS * 0.28), caida=0.2)
                        for f in [523, 415, 330, 262]])
    s['game_over'] = _a_sonido(x, 0.25)
    
    # Inicio de partida: arpegio ascendente
    x = np.concatenate([_barrido(f, f, 0.08) * _env(int(FS * 0.08), caida=0.07) for f in [262, 330, 392, 523, 659]])
    s['inicio'] = _a_sonido(x, 0.2)
    return s


def musica():
    """Bucle de música chiptune de 8 compases (bajo + arpegio), generado con ondas cuadradas."""
    bpm, compases = 132, 8
    negra = 60 / bpm
    
    progresion = [(110.0, [0, 3, 7]), (87.31, [0, 4, 7]), (130.81, [0, 4, 7]), (98.0, [0, 4, 7])]
    pista = []

    for c in range(compases):
        raiz, acorde = progresion[c % 4]
        for tiempo in range(4):
            n = int(FS * negra)
            t = np.arange(n) / FS
            bajo = np.sign(np.sin(2 * np.pi * raiz * t)) * 0.35 * np.exp(-t / 0.25)
            arp = np.zeros(n)
            sub = n // 4
            for k in range(4):   # semicorcheas de arpegio
                f = raiz * 4 * 2 ** (acorde[(tiempo + k) % 3] / 12)
                tt = np.arange(sub) / FS
                arp[k * sub:(k + 1) * sub] = np.sign(np.sin(2 * np.pi * f * tt)) * 0.12 * np.exp(-tt / 0.06)
            hihat = np.zeros(n)   # golpe de ruido corto a mitad de cada tiempo, como un platillo
            hh = _ruido(0.03)[:int(FS * 0.03)]
            hihat[n // 2:n // 2 + len(hh)] = hh * 0.08 * np.linspace(1, 0, len(hh))
            pista.append(bajo + arp + hihat)
    return _a_sonido(np.concatenate(pista), 0.5)


class Sonidos:
    """Inicia el mezclador de pygame, genera los sonidos y los reproduce."""

    def __init__(self):
        self.ok = True
        try:
            pygame.mixer.init(frequency=FS, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(24)
        except pygame.error:
            self.ok = False
            return
        self.s = sintetiza()
        self.musica = musica()
        
        # Sustituye por archivos propios si existen
        for nombre in list(self.s):
            ruta = os.path.join(CARPETA, f'{nombre}.wav')
            if nombre != 'carga' and os.path.exists(ruta):
                self.s[nombre] = pygame.mixer.Sound(ruta)
        ruta = os.path.join(CARPETA, 'musica.wav')
        if os.path.exists(ruta):
            self.musica = pygame.mixer.Sound(ruta)
        self.canal_musica = None

    def play(self, nombre, nivel=None):
        """Reproduce un efecto. Para 'carga', nivel (0..1) elige el tono: más agudo cuanto más carga."""
        if not self.ok:
            return
        snd = self.s[nombre]
        if isinstance(snd, list):
            snd = snd[min(len(snd) - 1, int(nivel * len(snd)))]
        snd.play()

    def inicia_musica(self):
        """Empieza la música en bucle si no está sonando ya."""
        if self.ok and (self.canal_musica is None or not self.canal_musica.get_busy()):
            self.canal_musica = self.musica.play(loops=-1)
            self.canal_musica.set_volume(0.5)

    def volumen_musica(self, v):
        """Cambia el volumen de la música (se baja mientras se carga la bomba)."""
        if self.ok and self.canal_musica is not None:
            self.canal_musica.set_volume(v)
