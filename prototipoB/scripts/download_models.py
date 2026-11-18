"""Descarga los modelos fijados en el manifiesto y verifica SHA-256."""

from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
    for model in manifest["models"]:
        destination = ROOT / model["path"]
        if destination.exists() and sha256(destination.read_bytes()).hexdigest() == model["sha256"]:
            print(f"Verificado: {model['name']}", flush=True)
            continue
        print(f"Descargando: {model['name']}", flush=True)
        with urlopen(model["url"], timeout=60) as response:
            data = response.read()
        if len(data) != model["bytes"] or sha256(data).hexdigest() != model["sha256"]:
            raise RuntimeError(f"Tamano o SHA-256 incorrecto: {model['name']}. No se sustituye el fichero local.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".task.download")
        temporary.write_bytes(data)
        temporary.replace(destination)
        print(f"Guardado y verificado: {model['path']}", flush=True)


if __name__ == "__main__":
    main()
