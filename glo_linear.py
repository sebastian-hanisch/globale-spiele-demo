"""Modell L (linear): Mitmachen bringt (Anteil der Mitmachenden − θ). Gleichgewichte in Schwellenstrategien, alle Wurzeln exakt statt auf einem Gitter.

Eigene Herleitung (durch Zählung und Simulation gestützt, nicht gegen Literatur geprüft): Zustand θ ~ N(y, 1/α), Signal x = θ + e, e ~ N(0, 1/β).
Wer die Schwelle x* spielt, macht mit, wenn sein Signal ≤ x* ist. Mit dem Anteil m = (αy + βx*)/(α+β) bei Indifferenz gilt

    Φ(κ(m − y)) = m,    κ = α·√((α+β)/(β(α+2β))),    x* = (m(α+β) − αy)/β.

g(m) = Φ(κ(m−y)) − m ist auf drei Stücken monoton (g' = κφ(κ(m−y)) − 1 hat höchstens zwei Nullstellen y ± z₀/κ, z₀ = √(2 ln(κ/√(2π))));
auf jedem Stück sucht eine Bisektion die Wurzel. Wurzeln dicht an 0 oder 1 (kleines β) liegen weit außerhalb jedes festen x-Gitters, daher wird in m (unteres Ende)
bzw. in u = 1 − m (oberes Ende) mit geometrischer Mitte gesucht; unter 10^-300 meldet die Rechnung `underflow`."""

from __future__ import annotations

import math
from dataclasses import dataclass

from glo_core import SQ2PI, bisect, ncdf, npdf

M_FLOOR = 1e-300      # kleinster Anteil, der noch gesucht wird; darunter Unterlauf-Flag


def kappa(alpha: float, beta: float) -> float:
    """κ = α·√((α+β)/(β(α+2β)))."""
    return alpha * math.sqrt((alpha + beta) / (beta * (alpha + 2.0 * beta)))


def x_star(m: float, alpha: float, beta: float, y: float) -> float:
    """Schwelle zum Anteil m bei Indifferenz: x* = (m(α+β) − αy)/β."""
    return (m * (alpha + beta) - alpha * y) / beta


def share_of_x(x: float, alpha: float, beta: float, y: float) -> float:
    """Umkehrung von `x_star`: m = (αy + βx)/(α+β), das Erwartungsniveau des Zustands bei Signal x."""
    return (alpha * y + beta * x) / (alpha + beta)


def _g_m(m: float, kap: float, y: float) -> float:
    return ncdf(kap * (m - y)) - m


def _g_u(u: float, kap: float, y: float) -> float:
    """Dieselbe Funktion in u = 1 − m, genau am oberen Ende: g = u − Φ(κ(y − 1 + u))."""
    return u - ncdf(kap * (y - 1.0 + u))


def g_values(ms, alpha: float, beta: float, y: float):
    """g(m) für ein Array von m (zum Zeichnen)."""
    import numpy as np
    ms = np.asarray(ms, dtype=float)
    return ncdf(kappa(alpha, beta) * (ms - y)) - ms


def slope(m: float, kap: float, y: float) -> float:
    """g'(m) = κ·φ(κ(m − y)) − 1; negativ heißt: die Beste-Antwort-Iteration zieht zu dieser Wurzel hin (stabil)."""
    return kap * npdf(kap * (m - y)) - 1.0


@dataclass(frozen=True)
class LinearEquilibria:
    alpha: float
    beta: float
    y: float
    kappa: float
    m: tuple[float, ...]          # Anteil bei Indifferenz, aufsteigend
    x: tuple[float, ...]          # Schwellen x*, aufsteigend
    stable: tuple[bool, ...]      # g' < 0
    underflow: bool               # mindestens eine Wurzel liegt unter 10^-300 an einem Rand (als 0 bzw. 1 gemeldet)

    @property
    def count(self) -> int:
        return len(self.m)


def _turning_points(kap: float, y: float) -> list[float]:
    """Stellen in (0, 1), an denen g' = 0 ist (nur für κ > √(2π))."""
    if kap <= SQ2PI:
        return []
    z0 = math.sqrt(2.0 * math.log(kap / SQ2PI))
    return [t for t in (y - z0 / kap, y + z0 / kap) if 0.0 < t < 1.0]


def _piece_roots(kap: float, y: float, lo: float, hi: float) -> tuple[list[float], bool]:
    """Wurzeln von g auf dem monotonen Stück [lo, hi] (liegt ganz in [0, ½] oder [½, 1]) als Werte von m, dazu das Unterlauf-Flag."""
    lower = hi <= 0.5
    if lower:
        h, tl, th = (lambda t: _g_m(t, kap, y)), lo, hi
    else:
        h, tl, th = (lambda t: _g_u(t, kap, y)), 1.0 - hi, 1.0 - lo
    fl, fh = h(tl), h(th)
    roots: list[float] = []
    underflow = False
    if tl == 0.0:
        # am äußersten Rand (m = 0 bzw. u = 0) hat h das Vorzeichen des Randwerts; kippt es bis 10^-300, liegt die Wurzel darunter
        f_floor = h(M_FLOOR)
        if fl == 0.0 or (f_floor > 0) != (fl > 0):
            underflow = True
            roots.append(0.0 if lower else 1.0)       # Wurzel < 10^-300: nicht darstellbar
            return roots, underflow
        tl, fl = M_FLOOR, f_floor
    if fl == 0.0:
        roots.append(tl if lower else 1.0 - tl)
    elif fh == 0.0:
        roots.append(th if lower else 1.0 - th)
    elif (fl > 0) != (fh > 0):
        t = bisect(h, tl, th, geometric=True)
        roots.append(t if lower else 1.0 - t)
    return roots, underflow


def roots_linear(alpha: float, beta: float, y: float) -> LinearEquilibria:
    """Alle Schwellen-Gleichgewichte des Modells L zu (α, β, y)."""
    kap = kappa(alpha, beta)
    edges = sorted({0.0, 0.5, 1.0, *_turning_points(kap, y)})
    found: list[float] = []
    underflow = False
    for lo, hi in zip(edges, edges[1:]):
        r, u = _piece_roots(kap, y, lo, hi)
        found += r
        underflow = underflow or u
    found.sort()
    merged: list[float] = []
    for m in found:                                   # Wurzel genau an einer Stückgrenze wird von beiden Stücken gemeldet
        if not merged or abs(m - merged[-1]) > 1e-13:
            merged.append(m)
    xs = tuple(x_star(m, alpha, beta, y) for m in merged)
    stable = tuple(slope(m, kap, y) < 0 for m in merged)
    return LinearEquilibria(alpha, beta, y, kap, tuple(merged), xs, stable, underflow)


# ---------------------------------------------------------------- geschlossene Formen
def critical_beta(alpha: float) -> float:
    """Kritisches β bei y = ½: ab hier (β ≥ β_krit) gibt es nur ein Gleichgewicht, darunter drei. Positive Wurzel von 4πβ² + (2πα − α²)β − α³ = 0 (κ² = 2π)."""
    b = 2.0 * math.pi * alpha - alpha * alpha
    return (-b + math.sqrt(b * b + 16.0 * math.pi * alpha ** 3)) / (8.0 * math.pi)


def fold_edge(kap: float) -> float | None:
    """Obere Kante Y(κ) des Prior-Bereichs mit drei Gleichgewichten (Faltenkurve), Y = Φ(z₀) − z₀φ(z₀); None für κ ≤ √(2π). Der Bereich ist (1 − Y, Y)."""
    if kap <= SQ2PI:
        return None
    z0 = math.sqrt(2.0 * math.log(kap / SQ2PI))
    return ncdf(z0) - z0 * npdf(z0)


def count_closed(alpha: float, beta: float, y: float) -> int:
    """Zahl der Gleichgewichte aus der Faltenkurve: 3, wenn κ > √(2π) und 1 − Y < y < Y, sonst 1."""
    top = fold_edge(kappa(alpha, beta))
    return 3 if top is not None and 1.0 - top < y < top else 1


def formula_holds(alpha: float, beta: float) -> bool:
    """Die Faustformel β ≥ α²/(2π): im Modell L hinreichend für Eindeutigkeit, nicht notwendig."""
    return beta >= alpha * alpha / (2.0 * math.pi)
