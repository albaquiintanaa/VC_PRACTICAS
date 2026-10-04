# Práctica 2 — Funciones básicas de OpenCV

Práctica de la asignatura **Visión por Computador** dedicada al procesamiento básico de imágenes y vídeo con OpenCV. El trabajo incluye detección de bordes, recuento de píxeles, umbralización, comparación de detectores y dos demostradores interactivos con webcam.

## 👥 Autores

Este trabajo ha sido realizado por el **`Grupo 27`**: 
- [Alba Quintana Ramos](https://github.com/albaquiintanaa)
- [Néstor Jesús Henríquez Medina](https://github.com/NestorJHM)

## Contenidos

1. [Requisitos y ejecución](#requisitos-y-ejecución)
2. [Tarea 1 — Recuento de píxeles por filas](#tarea-1--recuento-de-píxeles-por-filas)
3. [Tarea 2 — Comparación entre Canny y Sobel](#tarea-2--comparación-entre-canny-y-sobel)
4. [Tarea 3 — Pelota interactiva](#tarea-3--pelota-interactiva)
5. [Tarea Extra — Seguimiento por columnas y efecto Cristal Roto](#tarea-extra--seguimiento-por-columnas-y-efecto-cristal-roto)
6. [Estructura del proyecto](#estructura-del-proyecto)
7. [Referencias](#referencias)

---

## Requisitos y ejecución

### Dependencias
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Pillow (`Pillow`)

Las dependencias pueden instalarse con:

```bash
pip install opencv-python numpy matplotlib Pillow
```

### Ejecución

1. Clonar o descargar este repositorio (rama `P2`).
2. Abrir `VC_P2.ipynb` con Jupyter Notebook o VS Code.
3. Cada celda es independiente en su ejecución.
4. Para los demostradores interactivos (Tarea 3 y Extra), permitir el acceso a la webcam.

---
# Tareas realizadas
---

## Tarea 1 — Recuento de píxeles por filas

### Objetivo

Partiendo de la imagen binaria generada por el detector de bordes Canny sobre `mandril.jpg`, se calcula el número de píxeles blancos de cada fila. Después se determina:

- La fila que contiene el máximo número de píxeles blancos (`maxfil`).
- El valor máximo obtenido.
- Las filas cuyo recuento supera el 90 % del máximo (`0.90 * maxfil`).
- La posición de todas las filas seleccionadas.

### Procedimiento

La suma por filas se realiza con `cv2.reduce`. Como la salida de Canny contiene valores `0` y `255`, el resultado se divide entre `255` y entre el número de columnas para obtener una proporción de píxeles blancos por fila.

La fila máxima se obtiene con `np.argmax`. Sobre la imagen de Canny se dibujan:

- 🟡 **En amarillo:** la fila con el máximo recuento (`fila_max = 100`).
- 🔴 **En rojo:** las demás filas que superan el 90 % del máximo.
- Una flecha para destacar visualmente la fila con el máximo recuento (`plt.annotate`).

También se representa a la derecha una gráfica con el perfil de respuesta de Canny para todas las filas.

### Resultado de la Tarea 1

Este procedimiento permite identificar las alturas de la imagen en las que existe una mayor concentración de bordes horizontales o de cambios relevantes en la escena.

![Resultado de la Tarea 1](img/tarea_1.png)

---

## Tarea 2 — Comparación entre Canny y Sobel

### Objetivo

Aplicar un umbral a la imagen obtenida mediante Sobel (convertida a 8 bits) y comparar sus resultados con los producidos por Canny. Para ambos métodos se calcula:

- El número de píxeles no nulos por fila y por columna.
- El máximo por filas y el máximo por columnas.
- Las filas y columnas que superan el 90 % de su máximo correspondiente.
- El número total y el porcentaje de píxeles no nulos.

### Procedimiento

La imagen `mandril.jpg` se convierte a escala de grises y se suaviza mediante un filtro gaussiano. A continuación:

1. Se calculan las derivadas horizontal y vertical con Sobel.
2. El resultado se convierte a 8 bits con `cv2.convertScaleAbs`.
3. Se aplica un umbral de valor `130`.
4. Se obtiene Canny utilizando los umbrales `100` y `200`.
5. Se cuentan los píxeles no nulos mediante `np.count_nonzero` (`axis=1` para filas y `axis=0` para columnas).
6. Se remarcan las filas y columnas seleccionadas sobre las imágenes.

Código de colores utilizado:

* 🔴 **Rojo:** filas que superan el 90 % del máximo.
* 🔵 **Cian:** columnas que superan el 90 % del máximo.
* 🟡 **Amarillo:** fila y columna con el máximo absoluto.

### Resultados obtenidos

| Método | Máximo por fila | Máximo por columna | Filas seleccionadas | Columnas seleccionadas | Píxeles no nulos |
|---|---:|---:|---:|---:|---:|
| Canny (100, 200) | 184 | 138 | 5 (`[6, 8, 20, 24, 100]`) | 3 (`[92, 104, 119]`) | 34 763 (13,26 %) |
| Sobel umbralizado (130) | 161 | 179 | 7 (`[3, 4, 20, 51, 81, 82, 83]`) | 1 (`[288]`) | 31 199 (11,90 %) |

El detector Canny produce contornos generalmente más finos y continuos y conserva más detalle del pelo del mandril.

Por contra, Sobel umbralizado genera trazos más gruesos y fragmentados, y el resultado depende de manera más directa del umbral seleccionado (concentrando gran cantidad de píxeles blancos en bordes verticales de fuerte contraste y descartando otros).

También se muestra la diferencia absoluta entre las dos salidas y una versión umbralizada para localizar las discrepancias más significativas.

### Resultado de la Tarea 2

![Resultado de la Tarea 2](img/tarea_2.png)

---

## Tarea 3 — Pelota interactiva

### Objetivo

Superponer una pelota virtual sobre la imagen de la webcam y simular su movimiento mediante gravedad, velocidad y rebotes. El usuario puede golpearla hacia arriba moviendo la mano cerca de ella.

### Detección del golpe

La región cuadrada que rodea la posición actual de la pelota se extrae en dos fotogramas consecutivos. Para cada región:

1. Se calcula un histograma de niveles de gris con `cv2.calcHist`.
2. El histograma se normaliza mediante la norma L1 (`cv2.normalize`).
3. Ambos histogramas se comparan utilizando la distancia de Bhattacharyya (`cv2.compareHist`).
4. Si la variación supera `UMBRAL_GOLPE` (`0.20`), se aplica un impulso vertical ascendente y se muestra brevemente un círculo exterior indicando el impacto.

La comparación utiliza las mismas coordenadas en ambos fotogramas. Por ello, los bordes y colores estáticos del fondo no provocan golpes por sí solos; es necesario que exista una variación temporal cerca de la pelota.

_Para evitar que un único movimiento se interprete como varios impactos, se establece un intervalo mínimo de `0,20` segundos entre golpes._

### Física de la pelota

La simulación utiliza el tiempo real transcurrido (`dt`) entre fotogramas. En cada iteración:

```python
velocidad[1] += GRAVEDAD * dt
posicion += velocidad * dt
```

La pelota rebota contra los cuatro límites de la imagen. El coeficiente de rebote reduce su velocidad después de cada colisión para representar una pérdida gradual de energía.

### Controles

| Tecla | Acción |
|---|---|
| `R` | Reinicia la posición y la velocidad de la pelota |
| `ESC` | Cierra el demostrador |

### Resultado de la Tarea 3: Vídeo demostración

https://github.com/user-attachments/assets/38693ce3-5481-4417-9f3d-fbae1b4d8f1b

> `Nota:` _Si el reproductor no se visualiza correctamente, el archivo de vídeo también se encuentra disponible en la ruta [`videos/tarea_3.mp4`](./videos/tarea_3.mp4)._

---

## Tarea Extra — Seguimiento por columnas y efecto Cristal Roto

### Objetivo

Reinterpretar directamente el sistema de seguimiento horizontal de *My little piece of privacy* combinándolo con un efecto visual reactivo de "rotura de cristal" cuando se produce un movimiento brusco o impacto frente a la cámara.

### Procedimiento

En cada fotograma capturado por la webcam se aplica un volteo horizontal (`cv2.flip`) y se convierte a escala de grises. A continuación:

1. Se calcula la diferencia absoluta entre el fotograma actual y el anterior mediante `cv2.absdiff`.
2. Se binariza la diferencia aplicando un umbral de valor `25` (`cv2.threshold`).
3. Se suman los píxeles blancos de cada columna con `cv2.reduce` y se normaliza el resultado dividiendo entre `255 * alto`.
4. Se obtiene la columna con mayor actividad usando `np.argmax`.

Código de colores utilizado en pantalla:

* 🔴 **Rojo:** línea vertical que marca la columna con mayor movimiento (si supera el 5 % de activación).
* 🟢 **Verde:** histograma inferior de barras que representa en tiempo real el nivel de movimiento por cada columna.

### Efecto de impacto ("Cristal Roto")

Cuando la proporción total de píxeles activos en la imagen binarizada alcanza o supera el **60 %** del área de la escena:

```python
if np.sum(imgdif == 255) / imgdif.size >= 0.60:
    frames_congelados = 80
    ultimo_frame_valido = frame.copy()

```

Durante `80` fotogramas (~2 segundos), la imagen se congela mostrando el último fotograma válido y superponiendo la textura `cristal.png` mediante una máscara binaria.

### Resultado de la Tarea Extra: Vídeo demostración

https://github.com/user-attachments/assets/1aec5732-811c-4bbf-a711-c5a419a50a1d

> `Nota:` _Si el reproductor no se visualiza correctamente, el archivo de vídeo también se encuentra disponible en la ruta [`videos/tarea_extra.mp4`](./videos/tarea_extra.mp4)._

---

## Estructura del proyecto

```text
VC_PRACTICAS/
├── img/
│   ├── tarea_1.png
│   └── tarea_2.png
├── videos/
│   ├── tarea_3.mp4
│   └── tarea_extra.mp4
├── .gitattributes
├── README.md
├── VC_P2.ipynb
├── cristal.png
└── mandril.jpg

```

## Referencias

* [OpenCV](https://opencv.org/)
* [Documentación de OpenCV](https://docs.opencv.org/)
* [Documentación de Matplotlib (`annotate`)](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html)
* [My Little Piece of Privacy — Niklas Roy](https://www.niklasroy.com/project/88/my-little-piece-of-privacy)
* [Messa di Voce — Golan Levin y Zachary Lieberman](https://youtu.be/GfoqiyB1ndE?feature=shared)
* [Virtual Air Guitar](https://youtu.be/FIAmyoEpV5c?feature=shared)

---
