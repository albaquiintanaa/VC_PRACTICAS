"""
Entrenamiento del detector de matrículas RT-DETRv2-S (ResNet-18) en GPU.

Uso, desde la carpeta P4 y con el entorno VC_P4 activado:
    python train.py

Qué hace, en orden:
  1. Comprueba que PyTorch ve la GPU; si no, se detiene (así no se queda entrenando en la CPU sin avisar).
  2. Descarga los pesos de partida (RT-DETRv2-S entrenado en COCO) si no están en weights/.
  3. Carga la configuración del yml y lanza el solver del repositorio oficial, que entrena y evalúa
     sobre validación al final de cada época.
  4. Al terminar, lee log.txt y muestra la mejor época.

Hacemos un ajuste fino (fine-tuning): el modelo ya viene entrenado con las 80 clases de COCO y solo
tiene que aprender la clase nueva (matrícula), por lo que aprende mucho más rápido que desde cero.

La configuración está en RT-DETR/rtdetrv2_pytorch/rtdetrv2_matriculas.yml. Antes de entrenar hay que
aplicar el parche de parada temprana (sección 3.1 de VC_P4_RTDETR.ipynb), y el análisis del
entrenamiento está en la sección 4 del mismo cuaderno.

"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch

# Configuración
P4 = Path(__file__).resolve().parent
REPO = P4 / "RT-DETR" / "rtdetrv2_pytorch"
CFG = REPO / "rtdetrv2_matriculas.yml"
SALIDA = REPO / "output" / "matriculas_r18_800"
PESOS_COCO = P4 / "weights" / "rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
URL_COCO = "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth"
SEMILLA = 0   # semilla fija para que el entrenamiento sea reproducible


# Pasos
def comprobar_gpu():
    """Detiene el script si PyTorch no ve la GPU; si la ve, muestra su nombre y su memoria."""
    if not torch.cuda.is_available():
        sys.exit("PyTorch NO ve la GPU. Activa el entorno VC_P4 (torch con CUDA) antes de entrenar.")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name} · {p.total_memory / 2**30:.1f} GB · torch {torch.__version__}")


def descargar_si_falta(ruta, url):
    """
    Descarga un fichero si todavía no existe.
    Se descarga primero a un .part y se renombra al acabar: si la descarga se corta,
    no queda un fichero de pesos incompleto que parezca bueno.
    """
    if ruta.is_file():
        return
    print(f"Descargando {ruta.name} ...")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    tmp = ruta.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(ruta)


def entrenar():
    """
    Monta el solver del repositorio oficial con nuestra configuración y lo entrena.
    Reproduce lo que hace tools/train.py del repositorio, pero sin argumentos de consola.
    """
    # El repositorio importa sus módulos como "src.xxx": su carpeta tiene que estar en el path
    sys.path.insert(0, str(REPO))
    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS

    dist_utils.setup_distributed(0, "builtin", seed=SEMILLA)
    cfg = YAMLConfig(
        str(CFG),
        tuning=str(PESOS_COCO),        
        device="cuda:0",
        use_amp=True,                  
        seed=SEMILLA,
        output_dir=str(SALIDA),        
        PResNet={"pretrained": False},
    )
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)   
    solver.fit()                                
    dist_utils.cleanup()


def resumen():
    """Lee log.txt (una línea JSON por época) y muestra la mejor época según el mAP50-95 de validación."""
    log = SALIDA / "log.txt"
    if not log.is_file():
        print("No hay log.txt; ¿se interrumpió el entrenamiento?")
        return

    epocas = []
    for linea in log.read_text(encoding="utf-8").splitlines():
        if linea.strip():
            epocas.append(json.loads(linea))

    # test_coco_eval_bbox sigue el orden de pycocotools: [0] = mAP50-95, [1] = mAP50
    mejor = max(epocas, key=lambda e: e["test_coco_eval_bbox"][0])
    map5095, map50 = mejor["test_coco_eval_bbox"][:2]
    print(f"\nÉpocas entrenadas: {len(epocas)}. Mejor época: {mejor['epoch']}")
    print(f"  mAP50-95 = {map5095:.4f}   mAP50 = {map50:.4f}")
    print(f"  Pesos: {SALIDA / 'best.pth'}")


def main():
    """Ejecuta los cuatro pasos en orden y mide el tiempo total del entrenamiento."""
    print(f"yml: {CFG}\nsalida: {SALIDA}")
    comprobar_gpu()
    descargar_si_falta(PESOS_COCO, URL_COCO)
    if not CFG.is_file():
        sys.exit(f"No existe {CFG}")

    os.chdir(REPO)   # el yml incluye otros yml con rutas relativas a la carpeta del repositorio
    t0 = time.time()
    entrenar()
    print(f"Tiempo total: {(time.time() - t0) / 3600:.2f} h")
    resumen()


if __name__ == "__main__":
    main()
