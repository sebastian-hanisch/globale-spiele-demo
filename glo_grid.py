"""Das (α, β)-Raster der Gleichgewichtszahl für beide Modelle (Experiment E1), mit der Wurzelsuche gezählt, nicht mit dem geschlossenen Band.

Je (α, β)-Zelle wird über ein Raster von (y, c) (Modell R: 5 × 5 = 25 Zellen) bzw. y (Modell L: 9 Werte) gezählt, wie viele davon drei Gleichgewichte haben.
Vorgerechnet (tools/make_grid.py → data/grid.json, mit Hash der Dateien, aus deren Code die Zahlen stammen); die App zeigt die Datei nur, wenn der Hash passt."""

from __future__ import annotations

import math

import glo_linear as L
import glo_regime as R

ALPHAS = tuple(round(0.5 * (60.0 ** (i / 28.0)), 4) for i in range(29))        # 0,5 … 30 (geometrisch, 29 Werte)
BETAS = tuple(round(0.1 * (1000.0 ** (i / 30.0)), 4) for i in range(31))        # 0,1 … 100 (geometrisch, 31 Werte)
R_YS = (0.2, 0.35, 0.5, 0.65, 0.8)
R_CS = (0.2, 0.35, 0.5, 0.65, 0.8)
L_YS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def multi_cells_regime(alpha: float, beta: float) -> int:
    """Zahl der (y, c)-Zellen des Rasters (von 25), in denen das Modell R mehr als ein Gleichgewicht hat."""
    return sum(1 for y in R_YS for c in R_CS if R.roots_regime(alpha, beta, y, c).count > 1)


def multi_cells_linear(alpha: float, beta: float) -> int:
    """Zahl der y-Werte des Rasters (von 9), in denen das Modell L mehr als ein Gleichgewicht hat."""
    return sum(1 for y in L_YS if L.roots_linear(alpha, beta, y).count > 1)


def compute() -> dict:
    return {
        "alphas": list(ALPHAS), "betas": list(BETAS),
        "regime": {"ys": list(R_YS), "cs": list(R_CS), "cells": len(R_YS) * len(R_CS), "multi": [[multi_cells_regime(a, b) for b in BETAS] for a in ALPHAS]},
        "linear": {"ys": list(L_YS), "cells": len(L_YS), "multi": [[multi_cells_linear(a, b) for b in BETAS] for a in ALPHAS]},
    }


def formula_bound(alpha: float) -> float:
    """α²/(2π): oberhalb dieser Signalgenauigkeit ist im Modell R jedes (y, c) eindeutig."""
    return alpha * alpha / (2.0 * math.pi)
