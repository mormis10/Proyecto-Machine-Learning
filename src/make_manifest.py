import csv, hashlib
from pathlib import Path

RAW = Path("data/raw")
BASE_URL = "hhttps://datahub.io/football/english-premier-league" 
FECHA = "2026-10-04"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

with open(RAW / "MANIFEST.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["archivo", "url", "fecha_descarga", "sha256", "filas"])
    for p in sorted(RAW.glob("season-*.csv")):
        filas = sum(1 for _ in open(p, encoding="latin-1")) - 1
        w.writerow([p.name, f"{BASE_URL}/{p.name}", FECHA, sha256(p), filas])