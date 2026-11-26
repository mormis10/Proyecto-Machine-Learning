import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
BASE_URL = "https://datahub.io/football/english-premier-league"
FECHA = "2026-10-04"    

# Archivos que no vienen del espejo de DataHub o que se bajaron en otra fecha
EXCEPCIONES = {
    "season-2627.csv": dict(
        url="https://www.football-data.co.uk/mmz4281/2627/E0.csv",  # confirma esta URL
        fecha="2026-11-19",                                          # fecha real de descarga
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with open(RAW / "MANIFEST.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["archivo", "url", "fecha_descarga", "sha256", "filas"])
        for p in sorted(RAW.glob("season-*.csv")):
            with open(p, encoding="latin-1") as fh:
                filas = sum(1 for _ in fh) - 1
            extra = EXCEPCIONES.get(p.name, {})
            url = extra.get("url", f"{BASE_URL}/{p.name}")
            fecha = extra.get("fecha", FECHA)
            w.writerow([p.name, url, fecha, sha256(p), filas])


if __name__ == "__main__":
    main()