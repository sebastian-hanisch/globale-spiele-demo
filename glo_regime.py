"""Modell R (Regimewechsel, Hauptmodell): das Projekt startet, wenn der Anteil der Mitmachenden mindestens θ erreicht; Mitmachen bringt 1 bei Start und kostet immer c.

Eigene Herleitung (durch Zählung und Simulation gestützt, nicht gegen Literatur geprüft): Zustand θ ~ N(y, 1/α), Signal x = θ + e, e ~ N(0, 1/β), Schwellenstrategie
„mitmachen genau dann, wenn x ≤ x*“. Der Anteil Mitmachender ist l(θ) = Φ(√β(x* − θ)); der Start gelingt für θ ≤ θ̂ mit l(θ̂) = θ̂. Mit w = Φ⁻¹(θ̂) ist x* = θ̂ + w/√β.
Ein Betrieb mit Signal x* ist indifferent, wenn P(θ ≤ θ̂ | x*) = c, und das gibt mit z_c = Φ⁻¹(c)

    α·Φ(w) − √β·w = α·y + z_c·√(α+β),      in ρ = α/√β und d = (αy + z_c√(α+β))/√β:   G(w) = ρΦ(w) − w = d.

G' = ρφ(w) − 1 hat für ρ > √(2π) genau zwei Nullstellen ±w₀, w₀ = √(2 ln(ρ/√(2π))); G fällt außerhalb, steigt dazwischen. Drei Lösungen genau dann, wenn
G(−w₀) < d < G(w₀). Die Wurzeln werden auf den drei monotonen Stücken durch Bisektion gefunden, kein Gitter."""

from __future__ import annotations

import math
from dataclasses import dataclass

from glo_core import SQ2PI, bisect, ncdf, npdf, nppf


def rho_of(alpha: float, beta: float) -> float:
    """ρ = α/√β."""
    return alpha / math.sqrt(beta)


def d_value(alpha: float, beta: float, y: float, c: float) -> float:
    """d = (αy + z_c·√(α+β))/√β mit z_c = Φ⁻¹(c): die rechte Seite der reduzierten Gleichung G(w) = d."""
    return (alpha * y + nppf(c) * math.sqrt(alpha + beta)) / math.sqrt(beta)


def G(w, rho: float):
    """G(w) = ρΦ(w) − w (Skalar oder Array)."""
    return rho * ncdf(w) - w


def g_slope(w: float, rho: float) -> float:
    """G'(w) = ρφ(w) − 1; negativ heißt stabil (siehe `glo_iteration`)."""
    return rho * npdf(w) - 1.0


def w0_of(rho: float) -> float | None:
    """w₀ = √(2 ln(ρ/√(2π))), die Stelle, an der G' verschwindet; None für ρ ≤ √(2π) (dann fällt G überall: ein Gleichgewicht)."""
    if rho <= SQ2PI:
        return None
    return math.sqrt(2.0 * math.log(rho / SQ2PI))


def d_band(alpha: float, beta: float) -> tuple[float, float] | None:
    """Intervall von d, in dem es drei Gleichgewichte gibt: (G(−w₀), G(w₀)); None für ρ ≤ √(2π)."""
    rho = rho_of(alpha, beta)
    w0 = w0_of(rho)
    if w0 is None:
        return None
    return float(G(-w0, rho)), float(G(w0, rho))


def theta_x(w: float, beta: float) -> tuple[float, float]:
    """(θ̂, x*) zu w: θ̂ = Φ(w), x* = θ̂ + w/√β. Beide wachsen mit w."""
    theta = ncdf(w)
    return theta, theta + w / math.sqrt(beta)


@dataclass(frozen=True)
class RegimeEquilibria:
    alpha: float
    beta: float
    y: float
    c: float
    rho: float
    d: float
    w: tuple[float, ...]          # aufsteigend (und damit θ̂ und x* aufsteigend)
    theta: tuple[float, ...]      # kritischer Zustand θ̂: das Projekt startet für θ ≤ θ̂
    x: tuple[float, ...]          # Schwelle x*
    stable: tuple[bool, ...]      # G' < 0

    @property
    def count(self) -> int:
        return len(self.w)


def roots_regime(alpha: float, beta: float, y: float, c: float) -> RegimeEquilibria:
    """Alle Schwellen-Gleichgewichte des Modells R zu (α, β, y, c)."""
    rho = rho_of(alpha, beta)
    d = d_value(alpha, beta, y, c)
    big = abs(d) + rho + 2.0            # G(−big) > d > G(big): alle Lösungen liegen in [−big, big]
    w0 = w0_of(rho)
    edges = [-big, big] if w0 is None else [-big, -w0, w0, big]
    f = lambda w: float(G(w, rho)) - d  # noqa: E731
    found: list[float] = []
    for lo, hi in zip(edges, edges[1:]):
        flo, fhi = f(lo), f(hi)
        if flo == 0.0:
            found.append(lo)
        elif fhi == 0.0:
            found.append(hi)
        elif (flo > 0) != (fhi > 0):
            found.append(bisect(f, lo, hi))
    found.sort()
    merged: list[float] = []
    for w in found:
        if not merged or abs(w - merged[-1]) > 1e-12:
            merged.append(w)
    th_x = [theta_x(w, beta) for w in merged]
    return RegimeEquilibria(alpha, beta, y, c, rho, d, tuple(merged), tuple(t for t, _ in th_x), tuple(x for _, x in th_x),
                            tuple(g_slope(w, rho) < 0 for w in merged))


def count_closed(alpha: float, beta: float, y: float, c: float) -> int:
    """Zahl der Gleichgewichte aus dem geschlossenen Band: 3, wenn d zwischen G(−w₀) und G(w₀) liegt, sonst 1."""
    band = d_band(alpha, beta)
    if band is None:
        return 1
    return 3 if band[0] < d_value(alpha, beta, y, c) < band[1] else 1


def c_band(alpha: float, beta: float, y: float) -> tuple[float, float] | None:
    """Aufwand-Bereich (c_unten, c_oben) mit drei Gleichgewichten bei festem (α, β, y); None, wenn es nie drei gibt (ρ ≤ √(2π)).

    d(c) wächst mit c; die Bandkanten werden über z_c = (d√β − αy)/√(α+β) auf c = Φ(z_c) zurückgerechnet."""
    band = d_band(alpha, beta)
    if band is None:
        return None
    lo, hi = (ncdf((d * math.sqrt(beta) - alpha * y) / math.sqrt(alpha + beta)) for d in band)
    return lo, hi


def formula_holds(alpha: float, beta: float) -> bool:
    """β ≥ α²/(2π) ⟺ ρ ≤ √(2π): im Modell R genau die Bedingung, unter der es für jedes (y, c) nur ein Gleichgewicht gibt."""
    return beta >= alpha * alpha / (2.0 * math.pi)


def p_known_state_in_unit(alpha: float, y: float) -> float:
    """P(θ ∈ (0, 1)) unter dem Prior: nur dort gibt es bei bekanntem θ zwei Gleichgewichte (alle / keiner)."""
    return ncdf(math.sqrt(alpha) * (1.0 - y)) - ncdf(math.sqrt(alpha) * (0.0 - y))


def p_start(alpha: float, y: float, theta_hat: float) -> float:
    """P(θ ≤ θ̂): Wahrscheinlichkeit, dass das Projekt startet (θ̂ aus dem Gleichgewicht)."""
    return ncdf(math.sqrt(alpha) * (theta_hat - y))


def p_lost(alpha: float, y: float, theta_hat: float) -> float:
    """P(θ̂ < θ < 1): Zustände, in denen bei bekanntem θ „alle machen mit“ ein Gleichgewicht wäre, das Projekt aber nicht startet."""
    return ncdf(math.sqrt(alpha) * (1.0 - y)) - ncdf(math.sqrt(alpha) * (theta_hat - y))
