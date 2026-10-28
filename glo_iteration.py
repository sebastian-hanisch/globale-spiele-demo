"""Beste-Antwort-Iteration der Schwellen (Runden der iterierten Dominanz) im Kontinuum, beide Modelle. Eine Modellannahme, keine Aussage über Menschen.

Modell R: spielen alle anderen die Schwelle x, startet das Projekt für θ ≤ θ̂(x) mit θ̂ = Φ(√β(x − θ̂)); die beste Antwort ist die Schwelle
x' = ((α+β)(θ̂ − z_c/√(α+β)) − αy)/β  (Indifferenz bei P(θ ≤ θ̂ | x') = c). Modell L: m' = Φ(κ(m − y)).
Start „kaum einer macht mit“ (x₀ weit links) und „fast alle“ (x₀ weit rechts). Gemessen wird, nach wie vielen Runden die Iterierte bis auf `tol` an der kleinsten
bzw. größten Wurzel ist. Ein Abbruch über die Schrittweite täuscht nahe der Grenze (kleine Schritte, großer Rest) und wird nicht benutzt.

Stabilität (eigene Herleitung, durch Iteration von beiden Seiten gestützt): für die Antwortfunktion B gilt B'(x*) < 1 genau dann, wenn G'(w) < 0 (Modell R), bzw.
κφ(κ(m − y)) < 1 ⟺ g'(m) < 0 (Modell L). Eine Wurzel ist also stabil genau dann, wenn die reduzierte Gleichung dort fällt; bei drei Wurzeln ist die mittlere instabil,
bei einer Wurzel ist sie stabil."""

from __future__ import annotations

import math
from dataclasses import dataclass

from glo_core import SQ2PI, SQRT2, ncdf, npdf, nppf

START_LOW, START_HIGH = -50.0, 50.0     # „kaum einer“ / „fast alle“ als Startschwelle im Modell R
DEFAULT_TOL = 1e-6
ROUNDS_CAP = 200_000                    # Obergrenze der Runden in der App (die engste gemessene Zelle brauchte 35 325)


def theta_hat(x: float, beta: float, start: float | None = None) -> float:
    """Kritischer Zustand zur Schwelle x: der Fixpunkt θ = Φ(√β(x − θ)) in (0, 1) (eindeutig, die Differenz fällt streng), Newton mit Klammer [0, 1]."""
    sb = math.sqrt(beta)
    lo, hi = 0.0, 1.0
    th = 0.5 if start is None else min(max(start, 1e-12), 1.0 - 1e-12)
    erfc, exp = math.erfc, math.exp                    # math direkt: diese Schleife läuft bis zu 10^5-mal je Aufruf der App
    for _ in range(100):
        arg = sb * (x - th)
        h = 0.5 * erfc(-arg / SQRT2) - th
        if h > 0:
            lo = th
        else:
            hi = th
        nxt = th + h / (sb * exp(-0.5 * arg * arg) / SQ2PI + 1.0)
        if not (lo <= nxt <= hi):
            nxt = 0.5 * (lo + hi)
        if abs(nxt - th) < 1e-16:
            return nxt
        th = nxt
    return th


def best_reply_regime(x: float, alpha: float, beta: float, y: float, c: float, zc: float | None = None, start: float | None = None) -> tuple[float, float]:
    """(x', θ̂): die beste Antwort auf die Schwelle x im Modell R und der dabei benutzte kritische Zustand."""
    zc = nppf(c) if zc is None else zc
    th = theta_hat(x, beta, start)
    return ((alpha + beta) * (th - zc / math.sqrt(alpha + beta)) - alpha * y) / beta, th


def best_reply_slope_regime(w: float, alpha: float, beta: float) -> float:
    """B'(x*) der Antwortfunktion im Modell R an der Wurzel w: B' = (α+β)/β · s/(1+s) mit s = √β·φ(w) (aus θ̂' = s/(1+s)); B' < 1 genau dann, wenn G'(w) < 0."""
    s = math.sqrt(beta) * npdf(w)
    return (alpha + beta) / beta * s / (1.0 + s)


def best_reply_linear(m: float, kap: float, y: float) -> float:
    """Beste Antwort im Modell L: m' = Φ(κ(m − y))."""
    return ncdf(kap * (m - y))


@dataclass(frozen=True)
class IterationResult:
    rounds: int          # Runden bis |Iterierte − Ziel| < tol (bzw. bis zur Obergrenze)
    reached: bool        # False: Obergrenze erreicht, das Ziel nicht
    last: float          # letzte Iterierte
    monotone: bool       # alle Schritte in dieselbe Richtung


def iterate_regime(alpha: float, beta: float, y: float, c: float, x0: float, target: float, tol: float = DEFAULT_TOL, cap: int = ROUNDS_CAP) -> IterationResult:
    """Beste-Antwort-Runden im Modell R von x₀ bis auf `tol` an `target`."""
    zc = nppf(c)
    x, th = x0, None
    k, mono, prev = 0, True, None
    while k < cap and abs(x - target) >= tol:
        xn, th = best_reply_regime(x, alpha, beta, y, c, zc, th)
        step = xn - x
        if prev is not None and step * prev < -1e-12:
            mono = False
        prev, x, k = step, xn, k + 1
    return IterationResult(k, abs(x - target) < tol, x, mono)


def iterate_linear(alpha: float, beta: float, y: float, m0: float, target: float, tol: float = 1e-9, cap: int = ROUNDS_CAP) -> IterationResult:
    """Beste-Antwort-Runden im Modell L von m₀ (Anteil bei Indifferenz) bis auf `tol` an `target`."""
    from glo_linear import kappa
    kap = kappa(alpha, beta)
    m, k = m0, 0
    mono, prev = True, None
    while k < cap and abs(m - target) >= tol:
        mn = best_reply_linear(m, kap, y)
        step = mn - m
        if prev is not None and step * prev < -1e-15:
            mono = False
        prev, m, k = step, mn, k + 1
    return IterationResult(k, abs(m - target) < tol, m, mono)


def trajectory_regime(alpha: float, beta: float, y: float, c: float, x0: float, rounds: int) -> list[float]:
    """Die ersten `rounds` Iterierten x₀, x₁, … (zum Zeichnen)."""
    zc = nppf(c)
    out, x, th = [x0], x0, None
    for _ in range(rounds):
        x, th = best_reply_regime(x, alpha, beta, y, c, zc, th)
        out.append(x)
    return out
