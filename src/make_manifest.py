import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
BASE_URL = "https://datahub.io/football/english-premier-league"
FECHA = "2026-10-04"    


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
            w.writerow([p.name, f"{BASE_URL}/{p.name}", FECHA, sha256(p), filas])


if __name__ == "__main__":
    main()