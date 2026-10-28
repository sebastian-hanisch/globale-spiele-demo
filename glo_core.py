"""Normalverteilung ohne scipy: Verteilungsfunktion, Dichte und Umkehrfunktion (math.erfc, Bisektion).

Skalare und Arrays werden gleich behandelt. `ncdf` ist auch in den Rändern genau (erfc statt 1 − Φ), das brauchen die Randwurzeln des Modells L
(Anteile von 10^-10 und kleiner)."""

from __future__ import annotations

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)        # Schwelle der Eindeutigkeit: ρ = α/√β bzw. κ muss darüber liegen, damit es mehrere Gleichgewichte geben kann
_erfc = np.frompyfunc(math.erfc, 1, 1)


def ncdf(z):
    """Φ(z). Skalar → float, Array → Array."""
    if np.ndim(z) == 0:
        return 0.5 * math.erfc(-float(z) / SQRT2)
    z = np.asarray(z, dtype=float)
    return 0.5 * np.asarray(_erfc(-z / SQRT2), dtype=float)


def npdf(z):
    """φ(z)."""
    if np.ndim(z) == 0:
        z = float(z)
        return math.exp(-0.5 * z * z) / SQ2PI
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / SQ2PI


def nppf(p: float) -> float:
    """Φ⁻¹(p) für 0 < p < 1 durch Bisektion auf [-40, 40] (die Verteilungsfunktion ist streng wachsend)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p muss im offenen Intervall (0, 1) liegen")
    lo, hi = -40.0, 40.0
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if ncdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def bisect(f, a: float, b: float, geometric: bool = False, iterations: int = 200) -> float:
    """Nullstelle einer stetigen Funktion mit Vorzeichenwechsel auf [a, b] (f(a) und f(b) verschiedene Vorzeichen).

    `geometric`: Mitte als geometrisches Mittel (für Wurzeln nahe 0, a > 0), solange b/a > 4; sonst arithmetisch."""
    fa = f(a)
    for _ in range(iterations):
        mid = math.sqrt(a * b) if geometric and a > 0 and b > 0 and b / a > 4 else 0.5 * (a + b)
        if mid <= a or mid >= b:
            break
        fm = f(mid)
        if (fm > 0) == (fa > 0):
            a, fa = mid, fm
        else:
            b = mid
    return 0.5 * (a + b)
