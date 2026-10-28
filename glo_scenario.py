"""Szenario: die Einstellungen der App und die gezeichnete Spielrunde (Zustand, private Signale, wer mitmacht, Start ja/nein). Der Zufall kommt aus SplitMix64 (glo_rng)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import glo_constants as C
from glo_core import ncdf
from glo_rng import SplitMix64


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Alle Regler der Seitenleiste. Nur mit Schlüsselwörtern anlegen: ein eingefügtes Feld darf nie still Werte verschieben."""
    model: str = C.MODEL_R
    y: float = C.DEFAULT_Y
    alpha: float = C.DEFAULT_ALPHA
    beta: float = C.DEFAULT_BETA
    c: float = C.DEFAULT_C
    players: int = 0                  # 0 = Kontinuum
    seed: int = C.DEFAULT_SEED

    @property
    def is_regime(self) -> bool:
        return self.model == C.MODEL_R

    @property
    def finite(self) -> bool:
        """Endlich viele Betriebe werden nur im Modell R gerechnet; im Modell L ändert N nichts (Erwartungsnutzen gleich dem Kontinuumswert)."""
        return self.is_regime and self.players >= C.N_FINITE_MIN


def settings_of_preset(name: str) -> Settings:
    return Settings(**C.PRESETS[name])


def normal_from_uniforms(u1: float, u2: float) -> float:
    """Box–Muller: z = √(−2 ln(1 − u₁))·cos(2π u₂) aus zwei Gleichverteilten auf [0, 1)."""
    return math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


def _normal(rng: SplitMix64) -> float:
    u1 = rng.uniform()
    return normal_from_uniforms(u1, rng.uniform())


@dataclass(frozen=True)
class Round:
    theta: float                 # der gezogene Zustand
    signals: np.ndarray          # private Signale der gezeichneten Betriebe
    joins: np.ndarray            # wer mitmacht (Signal ≤ x*)
    joined: int                  # Zahl der Mitmachenden unter den gezeichneten
    n: int                       # Zahl der gezeichneten Betriebe
    share: float                 # Anteil der Mitmachenden: bei endlichem N gezählt, im Kontinuum exakt Φ(√β(x* − θ))
    exact: bool                  # True: share ist der exakte Kontinuumswert
    started: bool | None         # Modell R: startet das Projekt? (Modell L: None)
    payoff_join: float           # was ein Mitmachender bekommt (R: 1[Start] − c, L: Anteil der anderen − θ)
    x_star: float


def draw_round(s: Settings, x_star: float, theta_hat: float | None) -> Round:
    """Eine Spielrunde zum Zufalls-Seed der Einstellungen. `theta_hat`: kritischer Zustand des gespielten Gleichgewichts (Modell R im Kontinuum); sonst None.

    Reihenfolge der Ziehungen: erst θ, dann die Signale der Betriebe 1…n. Endliches N: Start, wenn alle Mitmachenden zusammen mindestens den Anteil θ stellen."""
    rng = SplitMix64(s.seed)
    theta = s.y + _normal(rng) / math.sqrt(s.alpha)
    n = s.players if s.finite else C.DRAW_CONTINUUM
    signals = np.array([theta + _normal(rng) / math.sqrt(s.beta) for _ in range(n)])
    joins = signals <= x_star
    joined = int(joins.sum())
    if not s.is_regime:                                  # Modell L: Mitmachen bringt (Anteil − θ), der Anteil ist im Kontinuum exakt
        share, exact, started = ncdf(math.sqrt(s.beta) * (x_star - theta)), True, None
        payoff = share - theta
    elif s.finite:                                       # Modell R, N Betriebe: alle Mitmachenden zusammen zählen
        share, exact = joined / n, False
        started = share >= theta
        payoff = (1.0 if started else 0.0) - s.c
    else:                                                # Modell R, Kontinuum: Start genau für θ ≤ θ̂
        if theta_hat is None:
            raise ValueError("Modell R im Kontinuum braucht den kritischen Zustand θ̂")
        share, exact = ncdf(math.sqrt(s.beta) * (x_star - theta)), True
        started = theta <= theta_hat
        payoff = (1.0 if started else 0.0) - s.c
    return Round(theta, signals, joins, joined, n, share, exact, started, payoff, x_star)
