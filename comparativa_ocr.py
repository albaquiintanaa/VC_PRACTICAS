"""Evaluación pareada de PaddleOCR y SmolVLM. Ejecutar desde VC_P4_YOLO_OCR.ipynb."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODEL = 'HuggingFaceTB/SmolVLM-256M-Instruct'
PROMPT = ('Read the vehicle license plate. Return only the letters and digits, without '
          'explanation. For two rows, read the top row first, then the bottom row. '
          'If unreadable, return an empty string.')


def normalizar(texto):
    return re.sub(r'[^A-Z0-9]', '', texto.upper())


def json_out(ruta, obj):
    ruta.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def csv_out(ruta, filas, campos):
    with ruta.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)


def preparar(args):
    from PIL import Image
    base = args.dataset / 'test'
    imagenes = sorted(p for p in (base / 'images').iterdir()
                      if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
    etiquetas = {p.stem: p for p in (base / 'labels').glob('*.json')}
    if (not imagenes or len({p.stem for p in imagenes}) != len(imagenes)
            or set(etiquetas) != {p.stem for p in imagenes}):
        raise ValueError('Imágenes y JSON de test no se corresponden uno a uno.')
    ruta_revision = getattr(args, 'revision', ROOT/'reports/transcripciones/revision.json')
    revisiones = json.loads(ruta_revision.read_text(encoding='utf-8-sig')) if ruta_revision.exists() else []
    revision = {(r['imagen'], r['indice']): r for r in revisiones}
    candidatos, pendientes, exclusiones = [], [], []
    con_texto = sin_texto = 0
    for ruta in imagenes:
        datos = json.loads(etiquetas[ruta.stem].read_text(encoding='utf-8-sig'))
        if not isinstance(datos, list):
            raise ValueError(f'JSON inválido: {ruta.name}')
        with Image.open(ruta) as im:
            w, h = im.size
        for indice, a in enumerate(datos):
            rect = a['rectangle']
            if (not isinstance(rect, list) or len(rect) != 4 or any(
                    isinstance(v, bool) or not isinstance(v, (float, int))
                    or not math.isfinite(v) for v in rect)):
                raise ValueError(f'Caja inválida: {ruta.name}, {indice}')
            cx, cy, bw, bh = rect
            x1, y1, x2, y2 = cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2
            if bw <= 0 or bh <= 0 or min(x1, y1) < -1e-6 or max(x2, y2) > 1+1e-6:
                raise ValueError(f'Caja fuera de rango: {ruta.name}, {indice}')
            caja = [max(0, math.floor(x1*w)), max(0, math.floor(y1*h)),
                    min(w, math.ceil(x2*w)), min(h, math.ceil(y2*h))]
            texto = a['plate']
            falta_texto = texto is None or str(texto).strip().lower() in {'', 'none', 'null'}
            sin_texto += int(falta_texto)
            con_texto += int(not falta_texto)
            revisada = revision.get((ruta.name, indice))
            estado = revisada['estado'] if revisada else None
            if revisada and revisada['rectangle'] != rect:
                raise ValueError(f'La caja cambió desde la revisión: {ruta.name}, {indice}')
            if estado in {'ilegible', 'parcial', 'no_es_matricula'}:
                exclusiones.append(dict(imagen=ruta.name, indice=indice, motivo=estado))
                continue
            if falta_texto:
                pendientes.append(dict(imagen=ruta.name, indice=indice, texto_referencia=''))
                exclusiones.append(dict(imagen=ruta.name, indice=indice, motivo='sin_transcripcion_completa'))
                continue
            if not isinstance(texto, str) or not normalizar(texto):
                raise ValueError(f'Texto inválido: {ruta.name}, {indice}')
            if estado == 'transcrita' and normalizar(revisada.get('texto') or '') != normalizar(texto):
                raise ValueError(f'El texto no coincide con la revisión: {ruta.name}, {indice}')
            candidatos.append(dict(imagen=ruta.name, indice=indice, caja=caja,
                                   referencia=normalizar(texto), referencia_original=texto,
                                   revision_estado=estado or 'transcripcion_preexistente',
                                   ruta=str(ruta.resolve())))
    csv_out(args.salida/'transcripciones_pendientes.csv', pendientes,
            ['imagen', 'indice', 'texto_referencia'])
    csv_out(args.salida/'exclusiones.csv', exclusiones, ['imagen', 'indice', 'motivo'])
    random.Random(args.semilla).shuffle(candidatos)
    muestra = candidatos[:args.limite] if args.limite else candidatos
    if not muestra:
        raise ValueError('No hay matrículas con transcripción en test.')
    for i, fila in enumerate(muestra):
        recorte = args.salida/'recortes'/f'{i:05d}.png'
        with Image.open(fila['ruta']) as im:
            im.convert('RGB').crop(fila['caja']).save(recorte)
        fila['recorte'] = str(recorte.resolve())
        fila['sha256'] = hashlib.sha256(recorte.read_bytes()).hexdigest()
    inventario = dict(imagenes_test=len(imagenes), placas_con_texto=con_texto,
                      placas_sin_texto=sin_texto, placas_evaluables=len(candidatos),
                      placas_excluidas=len(exclusiones), transcripciones_pendientes=len(pendientes),
                      placas_evaluadas=len(muestra),
                      imagenes_evaluadas=len({f["imagen"] for f in muestra}), semilla=args.semilla)
    json_out(args.salida/'muestra.json', muestra)
    json_out(args.salida/'inventario.json', inventario)
    return inventario


def preparar_seleccion(args):
    """Evalúa únicamente los recortes y etiquetas aprobados en el revisor."""
    origen = args.seleccion.resolve()
    if not origen.is_file():
        raise ValueError("Primero selecciona recortes legibles con python revisar_recortes.py.")
    elegidas = json.loads(origen.read_text(encoding='utf-8-sig'))
    if not isinstance(elegidas, list) or not elegidas:
        raise ValueError("No hay recortes legibles seleccionados. Marca Q y confirma el texto con Enter.")
    vistas = set()
    for fila in elegidas:
        clave = (fila['imagen'], fila['indice'])
        if clave in vistas:
            raise ValueError('La selección contiene una matrícula duplicada.')
        vistas.add(clave)
        if fila.get('revision_manual') is not True:
            raise ValueError('La selección contiene recortes sin aprobación manual.')
        referencia = fila.get('referencia', '')
        if not isinstance(referencia, str) or not re.fullmatch(r'[A-Z0-9]+', referencia):
            raise ValueError('Todas las muestras deben tener una transcripción completa.')
        recorte = Path(fila['recorte'])
        etiqueta = json.loads(Path(fila['etiqueta']).read_text(encoding='utf-8-sig'))
        huella = hashlib.sha256(recorte.read_bytes()).hexdigest()
        if huella != fila['sha256'] or huella != etiqueta.get('sha256'):
            raise ValueError('Un recorte ha cambiado después de ser seleccionado.')
        if etiqueta.get('plate') != referencia or etiqueta.get('revision_manual') is not True:
            raise ValueError('La etiqueta no coincide con la selección manual.')
    muestra = elegidas[:args.limite] if args.limite else elegidas
    json_out(args.salida/'muestra.json', muestra)
    inventario = dict(origen=str(origen), criterio='seleccion_manual_legible',
                      placas_disponibles=len(elegidas), placas_evaluadas=len(muestra),
                      imagenes_evaluadas=len({f['imagen'] for f in muestra}))
    json_out(args.salida/'inventario.json', inventario)
    return inventario


def versiones():
    resultado = {}
    for p in ['torch', 'torchvision', 'transformers', 'paddleocr', 'paddlepaddle-gpu',
              'paddlepaddle', 'Pillow', 'numpy', 'matplotlib']:
        try:
            resultado[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return resultado


def worker(args):
    from PIL import Image
    import torch  # Antes de Paddle: mismo orden de DLL que el cuaderno.
    inicio_carga = time.perf_counter()
    revision = None
    if args.motor == 'paddle':
        os.environ.setdefault('FLAGS_allocator_strategy', 'auto_growth')
        import paddle
        from paddleocr import TextRecognition
        if args.dispositivo == 'gpu' and (not paddle.is_compiled_with_cuda()
                                         or not paddle.device.cuda.device_count()):
            raise RuntimeError('Paddle no tiene CUDA disponible.')
        lector = TextRecognition(model_name='en_PP-OCRv5_mobile_rec',
                                 device='gpu:0' if args.dispositivo == 'gpu' else 'cpu',
                                 enable_mkldnn=False)
        dtype = 'float32 (configuración Paddle)'

        def sincronizar():
            if args.dispositivo == 'gpu':
                paddle.device.synchronize()

        def reconocer(im):
            import numpy as np
            resultado = list(lector.predict(input=np.asarray(im)[:, :, ::-1].copy(),
                                            batch_size=1))[0]
            return str(resultado['rec_text'])
    else:
        from transformers import AutoProcessor, AutoModelForVision2Seq
        device = 'cuda:0' if args.dispositivo == 'gpu' else 'cpu'
        if args.dispositivo == 'gpu' and not torch.cuda.is_available():
            raise RuntimeError('PyTorch no tiene CUDA disponible.')
        tipo = torch.float16 if args.dispositivo == 'gpu' else torch.float32
        procesador = AutoProcessor.from_pretrained(args.modelo)
        modelo = AutoModelForVision2Seq.from_pretrained(
            args.modelo, torch_dtype=tipo, attn_implementation='eager').to(device).eval()
        revision = getattr(modelo.config, '_commit_hash', None)
        dtype = str(tipo)

        def sincronizar():
            if args.dispositivo == 'gpu':
                torch.cuda.synchronize()

        def reconocer(im):
            mensajes = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'text', 'text': PROMPT}]}]
            prompt = procesador.apply_chat_template(mensajes, add_generation_prompt=True)
            entradas = procesador(text=prompt, images=[im], return_tensors='pt')
            entradas = entradas.to(device, dtype=tipo)
            with torch.inference_mode():
                salida = modelo.generate(**entradas, max_new_tokens=32,
                                         do_sample=False, use_cache=True)
            nuevos = salida[:, entradas['input_ids'].shape[1]:]
            return procesador.batch_decode(nuevos, skip_special_tokens=True)[0].strip()

    sincronizar()
    carga_s = time.perf_counter()-inicio_carga
    muestra = json.loads((args.salida/'muestra.json').read_text(encoding='utf-8'))
    imagenes = []
    for fila in muestra:
        if hashlib.sha256(Path(fila['recorte']).read_bytes()).hexdigest() != fila['sha256']:
            raise ValueError('El recorte cambió después de preparar la evaluación.')
        with Image.open(fila['recorte']) as im:
            imagenes.append(im.convert('RGB'))
    for _ in range(args.calentamiento):
        reconocer(imagenes[0])
    sincronizar()
    filas = []
    for i, (fila, im) in enumerate(zip(muestra, imagenes)):
        tiempos, textos = [], []
        for _ in range(args.repeticiones):
            sincronizar()
            inicio = time.perf_counter()
            texto = reconocer(im)
            sincronizar()
            tiempos.append(1000*(time.perf_counter()-inicio))
            textos.append(texto)
        pred = normalizar(textos[0])
        filas.append(dict(imagen=fila['imagen'], indice=fila['indice'],
                          referencia=fila['referencia'], texto_original=textos[0],
                          prediccion=pred, acierto=int(pred == fila['referencia']),
                          ms=statistics.median(tiempos), tiempos_ms=tiempos,
                          salida_estable=len(set(textos)) == 1))
        if (i+1) % 25 == 0 or i+1 == len(muestra):
            print(f'{args.motor}: {i+1}/{len(muestra)}', flush=True)
            json_out(args.salida/f'{args.motor}.parcial.json', filas)
    gpu = None
    if args.dispositivo == 'gpu':
        props = torch.cuda.get_device_properties(0)
        gpu = dict(nombre=props.name, vram_gib=props.total_memory/1024**3)
    json_out(args.salida/f'{args.motor}.json', dict(
        motor=args.motor, modelo='en_PP-OCRv5_mobile_rec' if args.motor == 'paddle' else args.modelo,
        revision=revision, dtype=dtype, carga_s=carga_s, gpu=gpu,
        versiones=versiones(), repeticiones=args.repeticiones, calentamiento=args.calentamiento,
        dispositivo=args.dispositivo, manifiesto_sha256=hashlib.sha256((args.salida/'muestra.json').read_bytes()).hexdigest(), resultados=filas))


def generar_graficas(args):
    """Genera las gráficas de acierto completo y tiempo medio, sin editar el README."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nombres = ['PaddleOCR', 'SmolVLM-256M']
    datos = [json.loads((args.salida/f'{m}.json').read_text(encoding='utf-8'))
             for m in ['paddle', 'smolvlm']]
    # Comparar las muestras realmente medidas; el manifiesto de una ejecución
    # posterior no interviene al regenerar gráficas de resultados guardados.
    esperadas = [(f['imagen'], f['indice'], f['referencia'])
                 for f in datos[0]['resultados']]
    for d in datos[1:]:
        recibidas = [(f['imagen'], f['indice'], f['referencia']) for f in d['resultados']]
        if recibidas != esperadas:
            raise ValueError('Los motores no evaluaron exactamente la misma muestra.')
        for clave in ['repeticiones', 'calentamiento', 'dispositivo']:
            if d.get(clave) != datos[0].get(clave):
                raise ValueError('Los protocolos medidos difieren: '+clave)

    resumen = []
    for nombre, d in zip(nombres, datos):
        filas = d['resultados']
        tiempos = [f['ms'] for f in filas]
        resumen.append(dict(motor=nombre, n=len(filas),
                            aciertos=sum(f['acierto'] for f in filas),
                            exactitud=sum(f['acierto'] for f in filas)/len(filas),
                            media_ms=statistics.mean(tiempos),
                            mediana_ms=statistics.median(tiempos)))
        # Compatibilidad con resultados anteriores: no exportar la antigua distancia.
        campos = [k for k in filas[0] if k not in {'tiempos_ms', 'distancia'}]
        csv_out(args.salida/f"detalle_{d['motor']}.csv",
                [{k: f[k] for k in campos} for f in filas], campos)
    csv_out(args.salida/'metricas.csv', resumen, list(resumen[0]))

    colores = ['#00798c', '#d1495b']
    especificaciones = [
        ('acierto.png', [100*r['exactitud'] for r in resumen],
         'Tasa de aciertos de matrícula completa', 'Aciertos (%)', True),
        ('tiempos.png', [r['media_ms'] for r in resumen],
         'Tiempo medio de inferencia por matrícula', 'Milisegundos', False),
    ]
    publicas = ROOT/'images'/'comparativa_ocr'
    publicas.mkdir(parents=True, exist_ok=True)
    for nombre, valores, titulo, eje_y, porcentaje in especificaciones:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        barras = ax.bar(nombres, valores, color=colores, width=0.55)
        ax.set(title=titulo, ylabel=eje_y)
        ax.set_ylim(0, 105 if porcentaje else max(max(valores)*1.15, 1))
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.2)
        for barra, valor in zip(barras, valores):
            etiqueta = f'{valor:.2f}%' if porcentaje else f'{valor:.2f} ms'
            ax.annotate(etiqueta, (barra.get_x()+barra.get_width()/2, valor),
                        xytext=(0, 5), textcoords='offset points', ha='center')
        fig.tight_layout()
        fig.savefig(args.salida/nombre, dpi=160)
        plt.close(fig)
        shutil.copy2(args.salida/nombre, publicas/nombre)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=ROOT/'dataset_plates')
    p.add_argument('--salida',type=Path,default=ROOT/'reports'/'comparativa_ocr')
    p.add_argument('--modelo',default=str(ROOT/'models/SmolVLM-256M-Instruct')
                   if (ROOT/'models/SmolVLM-256M-Instruct/model.safetensors').is_file() else MODEL)
    p.add_argument('--revision',type=Path,default=ROOT/'reports/transcripciones/revision.json',
                   help='Registro manual para excluir ilegibles, parciales y cajas incorrectas')
    p.add_argument('--limite',type=int,default=0,help='0: todas las muestras seleccionadas; >0: limitar la cantidad')
    p.add_argument('--seleccion',type=Path,default=ROOT/'reports/ocr_legibles/muestra.json',
                   help='Recortes y etiquetas aprobados en revisar_recortes.py')
    p.add_argument('--usar-dataset',action='store_true',
                   help='Modo anterior: seleccionar del dataset sin exigir aprobación manual')
    p.add_argument('--semilla',type=int,default=42)
    p.add_argument('--repeticiones',type=int,default=3)
    p.add_argument('--calentamiento',type=int,default=3)
    p.add_argument('--dispositivo',choices=['gpu','cpu'],default='gpu')
    p.add_argument('--solo-graficas', action='store_true', help='Generar gráficas desde resultados guardados, sin ejecutar los modelos')
    p.add_argument('--motor',choices=['paddle','smolvlm'],help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.limite < 0 or args.repeticiones < 1 or args.calentamiento < 1:
        p.error('Límite >= 0, repeticiones >= 1 y calentamiento >= 1.')
    args.salida = args.salida.resolve()
    args.salida.mkdir(parents=True,exist_ok=True)
    (args.salida/'recortes').mkdir(exist_ok=True)
    if args.motor:
        worker(args)
        return
    if args.solo_graficas:
        generar_graficas(args)
        return
    inventario = preparar(args) if args.usar_dataset else preparar_seleccion(args)
    print(json.dumps(inventario),flush=True)
    for motor in ['paddle','smolvlm']:
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--motor',motor,
            '--salida',str(args.salida),'--modelo',args.modelo,'--dispositivo',args.dispositivo,
            '--repeticiones',str(args.repeticiones),'--calentamiento',str(args.calentamiento)],
            check=True,cwd=ROOT)
    generar_graficas(args)
    print('Gráficas generadas en images/comparativa_ocr.',flush=True)


if __name__ == '__main__':
    main()
