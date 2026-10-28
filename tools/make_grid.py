"""Rechnet das (α, β)-Raster der Gleichgewichtszahl (E1) neu und schreibt data/grid.json samt Erzeuger-Hash.

Aufruf (im Demo-Ordner, mit dem Test-venv):  python tools/make_grid.py
Ändert sich eine der Dateien glo_grid.py, glo_regime.py, glo_linear.py oder glo_core.py, stimmt der Hash nicht mehr (tests/test_measurements.py schlägt an, die App zeigt
dann keine Karte); danach diesen Aufruf wiederholen. Dauer: rund eine halbe Minute, ohne Zufallszahlen."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import glo_experiments as X  # noqa: E402


def main():
    start = time.time()
    payload = X.write_grid()
    r, l = payload["regime"]["multi"], payload["linear"]["multi"]
    print(f"geschrieben: {X.GRID_FILE}  ({len(r)} x {len(r[0])} Zellen je Modell, {time.time() - start:.0f} s, Hash {X.generator_hash()})")
    print(f"Modell R: {sum(v > 0 for row in r for v in row)} (alpha, beta)-Zellen mit mehreren Gleichgewichten; Modell L: {sum(v > 0 for row in l for v in row)}")


if __name__ == "__main__":
    main()
