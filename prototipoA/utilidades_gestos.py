"""
Funciones comunes de los cuadernos de entrenamiento (VC_P5_gestos_*.ipynb) y del juego.

Uso:
    from utilidades_gestos import carga_dataset, kfold_agrupado, ModeloGesto, ...

Qué contiene:
  1. Carga y preprocesado de imágenes. El juego usa exactamente el mismo preprocesado que el
     entrenamiento, para que los modelos reciban el mismo tipo de imagen con el que aprendieron.
  2. Descriptores clásicos: píxeles, HOG y LBP por celdas.
  3. Modelos: SVM con búsqueda en rejilla, KNN y una CNN pequeña con aumento de datos.
  4. Evaluación: k-fold agrupado por persona, tablas de métricas y latencia.
  5. Resúmenes: modelo elegido y facilidad de uso de cada método.
  6. ModeloGesto: guarda y carga un modelo entrenado y da la probabilidad del gesto.
"""
import os
import glob
import json
import time

import cv2
import numpy as np
import joblib
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ----- Carga y preprocesado -----
def recorte_central(img, fraccion):
    """Recorta el centro de la imagen conservando `fraccion` del alto y del ancho."""
    if fraccion >= 1:
        return img
    h, w = img.shape[:2]
    ch, cw = int(round(h * fraccion)), int(round(w * fraccion))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return img[y0:y0 + ch, x0:x0 + cw]


def preprocesa(img_bgr, tam, gris=True, ecualizar=False):
    """
    Pasa a gris (opcional), redimensiona a tam x tam y ecualiza el histograma (opcional).
    La ecualización reparte los niveles de gris por todo el rango y reduce las diferencias de
    iluminación entre cámaras; en color se ecualiza solo el brillo para no alterar los colores.
    """
    img = img_bgr

    if gris and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[:2] != (tam, tam):
        img = cv2.resize(img, (tam, tam), interpolation=cv2.INTER_AREA)
    if ecualizar:
        if img.ndim == 2:
            img = cv2.equalizeHist(img)
        else:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return img


def carga_dataset(carpeta, clases, tam, grupo_de=None, gris=True, ecualizar=False, fraccion=1.0):
    """
    Carga un dataset con una subcarpeta por clase y devuelve (X, y, grupos, rutas).
    La clase i de `clases` recibe la etiqueta i; la segunda es la positiva (cerrado, abierta, sonrie).
    grupo_de(nombre_archivo) da la persona de cada imagen, para que el k-fold no reparta a una
    misma persona entre entrenamiento y test. fraccion < 1 recorta antes el centro de la imagen.
    """
    X, y, grupos, rutas = [], [], [], []
    
    for etiqueta, clase in enumerate(clases):
        for f in sorted(glob.glob(os.path.join(carpeta, clase, '*.jpg'))):
            img = recorte_central(cv2.imread(f), fraccion)
            X.append(preprocesa(img, tam, gris, ecualizar))
            y.append(etiqueta)
            nombre = os.path.basename(f)
            if grupo_de:
                grupos.append(grupo_de(nombre))
            else:
                grupos.append(nombre)   # sin grupos, cada imagen es el suyo
            rutas.append(f)
    return np.array(X), np.array(y), np.array(grupos), rutas


# ----- Descriptores -----
def desc_pixeles(img):
    """Los píxeles en un vector, escalados a [0, 1]. Referencia más simple."""
    return img.astype(np.float32).ravel() / 255.0


def desc_hog(img, celdas=None):
    """
    HOG con una rejilla de celdas x celdas, 9 orientaciones y bloques de 2x2 celdas.
    Por defecto usa celdas de unos 8 píxeles (mínimo 4x4): 24x24 -> 4x4 y 64x64 -> 8x8.
    """
    if celdas is None:
        celdas = max(4, img.shape[0] // 8)
    ppc = img.shape[0] // celdas
    return hog(img, orientations=9, pixels_per_cell=(ppc, ppc),
               cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)


def desc_lbp(img, celdas=None):
    """
    Histogramas LBP (8 vecinos, radio 1, patrones uniformes no invariantes a rotación)
    de cada celda de la rejilla, concatenados. Basado en lbphist del cuaderno de la práctica.
    Por defecto 3x3 celdas en imágenes pequeñas (<= 32 px) y 4x4 en las mayores.
    """
    if celdas is None:
        if img.shape[0] <= 32:
            celdas = 3
        else:
            celdas = 4

    lbp = local_binary_pattern(img, 8, 1, method='nri_uniform')
    n_bins = 59 # nº de patrones distintos con 8 vecinos y método 'nri_uniform'
    h, w = img.shape[:2]
    ch, cw = h // celdas, w // celdas
    hists = []
    for i in range(celdas):
        for j in range(celdas):
            celda = lbp[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            hist, _ = np.histogram(celda, bins=n_bins, range=(0, n_bins), density=True)
            hists.append(hist)
    return np.concatenate(hists)


DESCRIPTORES = {'pixeles': desc_pixeles, 'hog': desc_hog, 'lbp': desc_lbp}


def extrae(X, tipo):
    """Aplica el descriptor tipo (pixeles, hog o lbp) a todas las imágenes de X."""
    f = DESCRIPTORES[tipo]
    return np.array([f(img) for img in X], dtype=np.float32)


# ----- Modelos -----
def crea_svm():
    """SVM con núcleo RBF y normalización
    C y gamma se eligen con una búsqueda en rejilla (cv=3)"""
    rejilla = {'svc__C': [1, 10, 100], 'svc__gamma': ['scale', 0.01, 0.001]}
    return GridSearchCV(make_pipeline(StandardScaler(), SVC(kernel='rbf', class_weight='balanced')),
                        rejilla, cv=3, n_jobs=-1)


def crea_knn(k=5):
    """KNN por mayoría entre los k vecinos más cercanos (k=5)."""
    return KNeighborsClassifier(n_neighbors=k)


def crea_cnn(tam, canales=1, aumento=True):
    """
    CNN pequeña (~27.500 parámetros): 3 capas convolucionales, una densa y salida sigmoide.
    Con aumento - en cada época las imágenes se voltean, desplazan, amplían y cambian de brillo
    y contraste al azar, para que la red no dependa del encuadre ni de la luz exactos.
    Las capas de aumento solo actúan al entrenar; al predecir dejan pasar la imagen tal cual.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    if aumento:
        capas_aumento = [
            layers.RandomFlip('horizontal'),
            layers.RandomTranslation(0.1, 0.1, fill_mode='nearest'),
            layers.RandomZoom((-0.25, 0.15), fill_mode='nearest'),
            layers.RandomBrightness(0.2, value_range=(0, 255)),
            layers.RandomContrast(0.3),
        ]
    else:
        capas_aumento = []
    modelo = keras.Sequential([
        keras.Input(shape=(tam, tam, canales)),
        *capas_aumento,
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid'),
    ])
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return modelo


def como_tensor(X):
    """Prepara las imágenes para la CNN: float32 y, si son en gris, (n, tam, tam) -> (n, tam, tam, 1)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 3:
        return X[..., None] # añade el canal
    return X


# ----- Evaluación -----
def metricas(y_true, y_pred):
    """Accuracy, precisión, recall y F1 respecto a la clase positiva (etiqueta 1)."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def kfold_agrupado(X, y, grupos, metodos, k=5, epocas_cnn=30, semilla=0, verbose=True, X_cnn=None):
    """
    Validación cruzada y agrupada por persona.
    Lista con 'Píxeles + KNN', 'HOG + SVM', 'LBP + SVM' y/o 'CNN'.
    X debe venir ya preprocesado (en gris para los descriptores); X_cnn permite dar a la CNN
    otra versión de las imágenes, por ejemplo en color.
    Devuelve, por método, las métricas de cada fold, la matriz de confusión acumulada y el
    tiempo de entrenamiento de cada fold.
    """
    if X_cnn is None:
        X_cnn = X # sin versión propia, la CNN usa las mismas imágenes
    import tensorflow as tf
    cfg = {
        'Píxeles + KNN': ('pixeles', crea_knn),
        'HOG + SVM': ('hog', crea_svm),
        'LBP + SVM': ('lbp', crea_svm),
    }
    # Los descriptores no dependen - una sola vez para todo el conjunto
    feats = {tipo: extrae(X, tipo) for m, (tipo, _) in cfg.items() if m in metodos}
    res = {m: {'folds': [], 'cm': np.zeros((2, 2), int), 't_entreno': []} for m in metodos}

    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semilla)
    for fold, (tr, te) in enumerate(skf.split(X, y, grupos), start=1):
        for m in metodos:
            t0 = time.time()
            if m == 'CNN':
                tf.keras.utils.set_random_seed(semilla + fold)
                if X_cnn.ndim == 3:
                    canales = 1
                else:
                    canales = X_cnn.shape[3]
                red = crea_cnn(X_cnn.shape[1], canales)
                red.fit(como_tensor(X_cnn[tr]), y[tr], epochs=epocas_cnn, batch_size=64, verbose=0)
                y_pred = (red.predict(como_tensor(X_cnn[te]), verbose=0).ravel() >= 0.5).astype(int)
            else:
                tipo, crea = cfg[m]
                clf = crea().fit(feats[tipo][tr], y[tr])
                y_pred = clf.predict(feats[tipo][te])
            res[m]['t_entreno'].append(time.time() - t0)
            res[m]['folds'].append(metricas(y[te], y_pred))
            res[m]['cm'] += confusion_matrix(y[te], y_pred, labels=[0, 1])
        if verbose:
            print(f'Fold {fold}: ' + ' | '.join(f"{m} F1={res[m]['folds'][-1]['f1']:.3f}" for m in metodos))
    return res


def tabla_resultados(res):
    """Imprime la media ± desviación entre folds de cada métrica, por método."""
    print(f"{'Método':15s} {'Accuracy':>15s} {'Precisión':>15s} {'Recall':>15s} {'F1':>15s}")
    for m, r in res.items():
        fila = []
        for k in ['accuracy', 'precision', 'recall', 'f1']:
            v = np.array([f[k] for f in r['folds']])
            fila.append(f'{v.mean():.3f} ± {v.std():.3f}')
        print(f'{m:15s} ' + ' '.join(f'{c:>15s}' for c in fila))


def f1_medio(res, metodo):
    """F1 medio y desviación entre folds de un método."""
    v = np.array([f['f1'] for f in res[metodo]['folds']])
    return v.mean(), v.std()


# ----- Resumen -----
def guarda_resumen(gesto, filas, elegido, motivo, columnas_f1, carpeta_res='resultados',
                   carpeta_fig='figuras', titulo=None):
    """
    Guarda la comparación final de métodos y el modelo elegido.
    filas: un diccionario por método con 'Método', las columnas de `columnas_f1` y, opcionalmente,
    'ms' (latencia por imagen). Genera figuras/<gesto>_resumen.png, resultados/<gesto>_resumen.md 
    y .json (tabla, modelo elegido y motivo). El mejor con una estrella.
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(carpeta_res, exist_ok=True)
    os.makedirs(carpeta_fig, exist_ok=True)
    metodos = [f['Método'] for f in filas]
    hay_ms = any(f.get('ms') is not None for f in filas)

    if hay_ms:
        n_graficas, ancho_fig = 2, 12
    else:
        n_graficas, ancho_fig = 1, 7

    fig, axs = plt.subplots(1, n_graficas, figsize=(ancho_fig, 4), squeeze=False)
    ax = axs[0, 0]
    ancho = 0.8 / len(columnas_f1)

    for j, col in enumerate(columnas_f1):
        xs = np.arange(len(metodos)) + (j - (len(columnas_f1) - 1) / 2) * ancho
        vals = [f.get(col) for f in filas]
        alturas = []
        for v in vals:
            if v is None:
                alturas.append(0) # Sin dato, barra vacía
            else:
                alturas.append(v)
        barras = ax.bar(xs, alturas, ancho, label=col)

        for xi, v, b, m in zip(xs, vals, barras, metodos):
            if v is None:
                ax.text(xi, 0.02, 'n/d', ha='center', fontsize=7, color='gray')
                continue
            ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', fontsize=8)
            if m == elegido:
                b.set_edgecolor('black'); b.set_linewidth(2)

    ax.set_xticks(range(len(metodos)))
    etiquetas = []

    for m in metodos:
        if m == elegido:
            etiquetas.append(f'★ {m}')
        else:
            etiquetas.append(m)

    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_ylim(0, 1.12)

    if all(c.startswith('F1') for c in columnas_f1):
        ax.set_ylabel('F1')
    else:
        ax.set_ylabel('proporción')

    ax.set_title(f'{gesto}: comparación de métodos (★ = elegido)')
    ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(columnas_f1), frameon=False)
    
    if hay_ms:
        ax2 = axs[0, 1]
        ms = [f.get('ms') for f in filas]
        colores = []
        for m in metodos:
            if m == elegido:
                colores.append('#DD8452')   # naranja: el elegido
            else:
                colores.append('#8C8C8C')   # gris: el resto
        ax2.bar(metodos, [v or 0 for v in ms], color=colores)
        for i, v in enumerate(ms):
            if v is None:
                ax2.text(i, 0, 'n/d', ha='center', va='bottom', fontsize=8, color='gray')
            else:
                ax2.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8, color='black')
        ax2.set_ylabel('ms por imagen')
        ax2.set_title('Latencia (menos es mejor)')
        ax2.tick_params(axis='x', labelsize=9)
        
    plt.tight_layout()
    ruta_fig = os.path.join(carpeta_fig, f'{gesto}_resumen.png')
    plt.savefig(ruta_fig, dpi=120)
    plt.show()

    cols = ['Método'] + list(columnas_f1)
    if hay_ms:
        cols.append('ms por imagen')
    lineas = [titulo or f'## Modelo de {gesto}', '',
              '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for f in filas:
        if f['Método'] == elegido:
            celdas = [f"**{f['Método']}** ★"]
        else:
            celdas = [f['Método']]
        for c in columnas_f1:
            if f.get(c) is None:
                celdas.append('—')
            else:
                celdas.append(f'{f[c]:.3f}')
        if hay_ms:
            if f.get('ms') is None:
                celdas.append('—')
            else:
                celdas.append(f"{f['ms']:.1f}")
        lineas.append('| ' + ' | '.join(celdas) + ' |')

    lineas += ['', f'**Modelo elegido:** {elegido}', '', f'**Motivo:** {motivo}', '',
               f'![Resumen {gesto}](../{ruta_fig.replace(os.sep, "/")})']
    
    with open(os.path.join(carpeta_res, f'{gesto}_resumen.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lineas) + '\n')
    with open(os.path.join(carpeta_res, f'{gesto}_resumen.json'), 'w', encoding='utf-8') as fh:
        json.dump({'gesto': gesto, 'elegido': elegido, 'motivo': motivo, 'metodos': filas},
                  fh, ensure_ascii=False, indent=1, default=float)
        
    print('\n'.join(lineas[:-2]))
    print(f'\nGuardado: {ruta_fig}, {carpeta_res}/{gesto}_resumen.md y .json')


AJUSTES = {
    'Píxeles + KNN': ('k (vecinos)', 'No entrena: guarda todas las imágenes y compara con todas al predecir.'),
    'HOG + SVM': ('celdas, orientaciones, bloques + C y gamma (rejilla)',
                  'Hay que diseñar el descriptor; la búsqueda en rejilla de la SVM multiplica el tiempo de entrenamiento.'),
    'LBP + SVM': ('celdas, vecinos, radio, tipo de patrón + C y gamma (rejilla)',
                  'Igual que HOG; el descriptor es más lento de calcular.'),
    'CNN': ('arquitectura, épocas, lote, optimizador, aumento de datos',
            'No hay que diseñar descriptores, pero sí la red; necesita TensorFlow. Su tiempo de entrenamiento depende de las épocas, no del número de vectores soporte, y en CPU es similar al de la SVM con búsqueda en rejilla.'),
    'MediaPipe': ('solo el umbral', 'Sin entrenamiento: ya se ejecuta para obtener los landmarks.'),
    'DeepFace': ('detector de caras y umbral', 'Sin entrenamiento; necesita detectar la cara y es la más lenta al predecir.'),
}


def tabla_facilidad(gesto, res, latencias, carpeta_modelos='modelos', referencias=None,
                    carpeta_res='resultados', carpeta_fig='figuras'):
    """
    Compara la facilidad de uso y el coste de cada método: tiempo de entrenamiento por fold,
    tiempo de predicción, tamaño del modelo guardado y qué hay que ajustar.
    res: resultados del k-fold; latencias: ms por imagen de los modelos finales;
    referencias: {nombre: ms o None} de los métodos sin entrenar (MediaPipe, DeepFace).
    Guarda resultados/<gesto>_facilidad.md y figuras/<gesto>_facilidad.png.
    """
    import matplotlib.pyplot as plt
    tipo_de = {'HOG + SVM': 'hog', 'LBP + SVM': 'lbp', 'CNN': 'cnn'}
    filas = []
    for m, r in res.items():
        if 't_entreno' not in r: # las referencias sin entrenar se añaden aparte
            continue
        tam = None

        if m in tipo_de:
            ruta = os.path.join(carpeta_modelos, f'{gesto}_{tipo_de[m]}')
            if tipo_de[m] == 'cnn':
                ruta += '.keras'
            else:
                ruta += '.joblib'
            if os.path.exists(ruta):
                tam = os.path.getsize(ruta) / 1e6

        ajustes, nota = AJUSTES[m]
        filas.append({'Método': m, 'entreno_s': float(np.mean(r['t_entreno'])), 'ms': latencias.get(m),
                      'MB': tam, 'ajustes': ajustes, 'nota': nota})
        
    for nombre, ms in (referencias or {}).items():
        if 'MediaPipe' in nombre:
            clave = 'MediaPipe'
        else:
            clave = 'DeepFace'
        ajustes, nota = AJUSTES[clave]
        filas.append({'Método': nombre, 'entreno_s': 0.0, 'ms': ms, 'MB': None, 'ajustes': ajustes, 'nota': nota})

    def fmt(v, f):
        """Formatea un número, o pone — si no hay dato."""
        if v is None:
            return '—'
        return format(v, f)

    lineas = [f'## Facilidad de uso y coste: {gesto}', '',
              '| Método | Entrenamiento por fold (s) | Predicción (ms/imagen) | Tamaño (MB) | Qué hay que ajustar | Observaciones |',
              '|---|---|---|---|---|---|']
    for f in filas:
        lineas.append(f"| {f['Método']} | {fmt(f['entreno_s'], '.1f')} | {fmt(f['ms'], '.1f')} | "
                      f"{fmt(f['MB'], '.2f')} | {f['ajustes']} | {f['nota']} |")
    os.makedirs(carpeta_res, exist_ok=True)
    with open(os.path.join(carpeta_res, f'{gesto}_facilidad.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lineas) + f'\n\n![Facilidad {gesto}](../{carpeta_fig}/{gesto}_facilidad.png)\n')
    print('\n'.join(lineas))

    fig, axs = plt.subplots(1, 3, figsize=(15, 3.6))
    nombres = [f['Método'] for f in filas]
    for ax, clave, titulo in zip(axs, ['entreno_s', 'ms', 'MB'],
                                 ['Entrenamiento por fold (s)', 'Predicción (ms por imagen)', 'Tamaño del modelo (MB)']):
        vals = [f[clave] for f in filas]
        ax.bar(nombres, [v or 0 for v in vals], color='#4C72B0')
        for i, v in enumerate(vals):
            if v is None:
                ax.text(i, 0, 'n/d', ha='center', va='bottom', fontsize=8, color='gray')
            else:
                if v >= 0.1 or v == 0:
                    texto = f'{v:.1f}'
                else:
                    texto = f'{v:.2f}'
                ax.text(i, v, texto, ha='center', va='bottom', fontsize=8, color='black')
        ax.set_title(titulo)
        ax.tick_params(axis='x', labelrotation=20, labelsize=8)

    plt.suptitle(f'{gesto}: facilidad de uso y coste')
    plt.tight_layout()
    os.makedirs(carpeta_fig, exist_ok=True)
    plt.savefig(os.path.join(carpeta_fig, f'{gesto}_facilidad.png'), dpi=120)
    plt.show()
    return filas


def mide_latencia(modelo_gesto, imgs, repeticiones=1):
    """Tiempo medio (ms) en clasificar una imagen cada vez, como ocurre en el juego fotograma a fotograma."""
    modelo_gesto.prob(imgs[0]) # la primera llamada es más lenta - no se mide
    t0 = time.perf_counter()

    for _ in range(repeticiones):
        for img in imgs:
            modelo_gesto.prob(img)
    return (time.perf_counter() - t0) / (len(imgs) * repeticiones) * 1000


class ModeloGesto:
    """
    Un modelo entrenado (SVM con su descriptor o CNN) junto con su preprocesado.
    prob(img_bgr) devuelve la probabilidad de la clase positiva (cerrado, abierta, sonrie).
    Se guarda en dos archivos: <nombre>.json (tipo y preprocesado) y <nombre>.keras o .joblib.
    """

    def __init__(self, tipo, modelo, tam, gris, clases, ecualizar=False):
        self.tipo, self.modelo, self.tam, self.gris, self.clases = tipo, modelo, tam, gris, clases
        self.ecualizar = ecualizar
        self._red = None
        if tipo == 'cnn':
            import tensorflow as tf
            if gris:
                canales = 1
            else:
                canales = 3
            # compila la CNN 2ms vs 25ms por imagen con Keras
            self._red = tf.function(lambda x: self.modelo(x, training=False),
                                    input_signature=[tf.TensorSpec([None, tam, tam, canales], tf.float32)])

    @classmethod
    def entrena(cls, tipo, X, y, tam, gris, clases, ecualizar=False, epocas_cnn=30, semilla=0):
        """Entrena el modelo final con todos los datos. X debe venir preprocesado con los mismos gris y ecualizar."""
        if tipo == 'cnn':
            import tensorflow as tf
            tf.keras.utils.set_random_seed(semilla)
            modelo = crea_cnn(tam, 1 if gris else 3)
            modelo.fit(como_tensor(X), y, epochs=epocas_cnn, batch_size=64, verbose=0)
        else:
            busqueda = crea_svm().fit(extrae(X, tipo), y)

            params = {k.replace('svc__', ''): v for k, v in busqueda.best_params_.items()}
            modelo = make_pipeline(StandardScaler(),
                                   CalibratedClassifierCV(SVC(kernel='rbf', class_weight='balanced', **params),
                                                          ensemble=False, cv=5))
            modelo.fit(extrae(X, tipo), y)
        return cls(tipo, modelo, tam, gris, clases, ecualizar)

    def prob(self, img):
        """Probabilidad de la clase positiva para una imagen (en BGR o ya preprocesada)."""
        img = preprocesa(img, self.tam, self.gris, self.ecualizar)
        if self.tipo == 'cnn':
            return float(self._red(como_tensor(img[None])).numpy().ravel()[0])
        x = DESCRIPTORES[self.tipo](img)[None]
        return float(self.modelo.predict_proba(x)[0, 1])

    def prob_lote(self, imgs):
        """Probabilidades de muchas imágenes a la vez."""
        X = np.array([preprocesa(i, self.tam, self.gris, self.ecualizar) for i in imgs])
        if self.tipo == 'cnn':
            return self._red(como_tensor(X)).numpy().ravel()
        return self.modelo.predict_proba(extrae(X, self.tipo))[:, 1]

    def guarda(self, ruta_base):
        """Guarda los metadatos en ruta_base.json y el modelo en ruta_base.keras o .joblib."""
        os.makedirs(os.path.dirname(ruta_base) or '.', exist_ok=True)
        meta = {'tipo': self.tipo, 'tam': self.tam, 'gris': self.gris, 'clases': self.clases,
                'ecualizar': self.ecualizar}
        with open(ruta_base + '.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        if self.tipo == 'cnn':
            self.modelo.save(ruta_base + '.keras')
        else:
            joblib.dump(self.modelo, ruta_base + '.joblib')

    @classmethod
    def carga(cls, ruta_base):
        """Carga un modelo guardado con guarda()."""
        with open(ruta_base + '.json', encoding='utf-8') as f:
            meta = json.load(f)
        if meta['tipo'] == 'cnn':
            from tensorflow import keras
            modelo = keras.models.load_model(ruta_base + '.keras')
        else:
            modelo = joblib.load(ruta_base + '.joblib')
        return cls(meta['tipo'], modelo, meta['tam'], meta['gris'], meta['clases'],
                   meta.get('ecualizar', False))