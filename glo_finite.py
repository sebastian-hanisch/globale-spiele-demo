"""Endlich viele Spieler im Modell R (N Betriebe statt Kontinuum).

Die N − 1 anderen spielen die Schwelle x (mitmachen genau dann, wenn ihr Signal ≤ x ist); ihre Zahl K ~ Bin(N−1, p(θ)) mit p(θ) = Φ(√β(x − θ)). Das Projekt startet, wenn
(1 + K)/N ≥ θ, der eigene Beitrag zählt mit. Ein Betrieb mit Signal s ist indifferent, wenn die Startwahrscheinlichkeit

    P_N(s; x) = ∫ P((1+K)/N ≥ θ | θ) · N(θ; m(s), 1/(α+β)) dθ,    m(s) = (αy + βs)/(α+β),

gleich c ist; ein Gleichgewicht ist eine Lösung von P_N(x; x) = c. Die Binomialzahl ist in θ eine Treppe: auf (j/N, (j+1)/N] müssen genau j andere mitmachen.
Darum wird stückweise integriert (Gauß–Legendre je Stück, 8, 4 oder 2 Knoten je nach N); eine Gitterintegration über θ erzeugt an den Sprungstellen
Quadraturrauschen und Scheingleichgewichte (Wurzelpaare im Abstand 0,005; so in der Vormessung gesehen).

Im Modell L ist der Erwartungsnutzen des Mitmachens für jedes N gleich dem Kontinuumswert (E[K/(N−1) | θ] = Φ(√β(x − θ)) exakt): dort braucht es dieses Modul nicht."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from glo_core import nppf, ncdf, npdf

N_MIN, N_MAX = 2, 1000


def needed_others(theta: float, n: int) -> int:
    """Wie viele der N − 1 anderen mindestens mitmachen müssen, damit (1 + K)/N ≥ θ gilt: ⌈Nθ − 1⌉, höchstens 0 für θ ≤ 1/N; für θ > 1 gibt es keine Zahl (n, nie erfüllbar)."""
    if theta <= 1.0 / n:
        return 0
    if theta > 1.0:
        return n
    return max(0, math.ceil(n * theta - 1.0 - 1e-12))


def binom_tail(j: int, trials: int, p: float) -> float:
    """P(K ≥ j) für K ~ Bin(trials, p), skalar, über Log-Wahrscheinlichkeiten (stabil auch für trials = 999)."""
    if j <= 0:
        return 1.0
    if j > trials:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for k in range(j, trials + 1):
        lc = math.lgamma(trials + 1) - math.lgamma(k + 1) - math.lgamma(trials - k + 1)
        total += math.exp(lc + k * lp + (trials - k) * lq)
    return min(1.0, total)


def success_given_theta(theta: float, p: float, n: int) -> float:
    """Wahrscheinlichkeit, dass das Projekt bei Zustand θ startet, wenn jeder der N − 1 anderen mit Wahrscheinlichkeit p mitmacht."""
    if theta > 1.0:
        return 0.0
    return binom_tail(needed_others(theta, n), n - 1, p)


@lru_cache(maxsize=32)
def _log_coeff(n: int):
    ks = np.arange(n)
    return math.lgamma(n) - np.array([math.lgamma(i + 1) + math.lgamma(n - i) for i in ks])


@lru_cache(maxsize=4)
def _gauss(nodes: int):
    return np.polynomial.legendre.leggauss(nodes)


def nodes_per_piece(n: int) -> int:
    """Gauß–Legendre-Knoten je Stück (j/N, (j+1)/N]: die Integranden sind dort glatt; bei großem N genügen wenige."""
    return 8 if n <= 60 else 4 if n <= 250 else 2


def success_prob(s: float, x: float, n: int, alpha: float, beta: float, y: float) -> float:
    """P_N(s; x): Wahrscheinlichkeit, dass das Projekt startet, für einen Betrieb mit Signal s, wenn die N − 1 anderen die Schwelle x spielen."""
    m = (alpha * y + beta * s) / (alpha + beta)
    sd = 1.0 / math.sqrt(alpha + beta)
    total = float(ncdf((1.0 / n - m) / sd))                       # θ ≤ 1/N: Start sicher
    nn = nodes_per_piece(n)
    gx, gw = _gauss(nn)
    j = np.arange(1, n)
    h = 1.0 / n
    t = ((j / n)[:, None] + h * (gx[None, :] + 1.0) / 2.0).ravel()
    wq = np.broadcast_to(gw[None, :] * h / 2.0, (n - 1, nn)).ravel()
    jj = np.repeat(j, nn)
    arg = math.sqrt(beta) * (x - t)
    p = np.clip(ncdf(arg), 1e-300, 1.0)
    q = np.clip(ncdf(-arg), 1e-300, 1.0)
    ks = np.arange(n)
    logpmf = _log_coeff(n)[None, :] + ks[None, :] * np.log(p)[:, None] + (n - 1 - ks)[None, :] * np.log(q)[:, None]
    surv = np.cumsum(np.exp(logpmf)[:, ::-1], axis=1)[:, ::-1]
    tail = surv[np.arange(len(t)), jj]
    return total + float(np.sum(wq * tail * npdf((t - m) / sd) / sd))


def search_window(n: int, alpha: float, beta: float, y: float, c: float) -> tuple[float, float]:
    """Intervall, in dem alle Gleichgewichte liegen müssen: links davon ist P_N ≥ P(θ ≤ 1/N) > c, rechts davon P_N ≤ P(θ ≤ 1) < c."""
    zc = nppf(c)
    sq = math.sqrt(alpha + beta)
    lo = ((alpha + beta) * (1.0 / n - zc / sq) - alpha * y) / beta
    hi = ((alpha + beta) * (1.0 - zc / sq) - alpha * y) / beta
    return lo, hi


@dataclass(frozen=True)
class FiniteEquilibria:
    n: int
    x: tuple[float, ...]          # Schwellen, aufsteigend
    stable: tuple[bool, ...]      # P_N(x;x) − c wechselt dort von + nach −
    window: tuple[float, float]
    scan_points: int              # Zahl der Rasterpunkte der Zählung (Paare dicht an Berührpunkten können fehlen)

    @property
    def count(self) -> int:
        return len(self.x)


def _illinois(f, a: float, b: float, fa: float, fb: float, iterations: int = 40, tol: float = 1e-10) -> float:
    side = 0
    for _ in range(iterations):
        c = (a * fb - b * fa) / (fb - fa)
        fc = f(c)
        if abs(fc) < 1e-13 or abs(b - a) < tol:
            return c
        if (fc > 0) == (fb > 0):
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a, fa = c, fc
            if side == 1:
                fb *= 0.5
            side = 1
    return c


def scan_points_for(n: int) -> int:
    """Rasterpunkte der Wurzelsuche: dichter, solange eine Auswertung billig ist."""
    return 241 if n <= 100 else 121 if n <= 250 else 81


def roots_finite(n: int, alpha: float, beta: float, y: float, c: float, scan: int | None = None) -> FiniteEquilibria:
    """Gleichgewichte bei N Betrieben: Vorzeichenwechsel von P_N(x;x) − c auf einem Raster im Suchfenster, verfeinert (Illinois-Verfahren)."""
    lo, hi = search_window(n, alpha, beta, y, c)
    scan = scan_points_for(n) if scan is None else scan
    xs = np.linspace(lo, hi, scan)
    f = lambda x: success_prob(x, x, n, alpha, beta, y) - c  # noqa: E731
    vals = [f(float(x)) for x in xs]
    roots, stable = [], []
    for i in range(scan - 1):
        a, b, fa, fb = float(xs[i]), float(xs[i + 1]), vals[i], vals[i + 1]
        if fa == 0.0:
            roots.append(a)
            stable.append(i == 0 or vals[i - 1] > 0)
        elif fa * fb < 0:
            roots.append(_illinois(f, a, b, fa, fb))
            stable.append(fa > 0)
    return FiniteEquilibria(n, tuple(roots), tuple(stable), (lo, hi), scan)


def best_reply(x: float, n: int, alpha: float, beta: float, y: float, c: float) -> float:
    """Beste Antwort bei N Betrieben: die Schwelle s, bei der ein Betrieb indifferent ist, wenn die anderen x spielen (P_N(s; x) = c, in s fallend), Bisektion im Suchfenster."""
    lo, hi = search_window(n, alpha, beta, y, c)
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if success_prob(mid, x, n, alpha, beta, y) - c > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
