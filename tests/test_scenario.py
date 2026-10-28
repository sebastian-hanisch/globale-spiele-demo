"""Szenario: Einstellungen (nur mit Schlüsselwörtern), Box–Muller von Hand, gezeichnete Spielrunde (Zufalls-Seed, Start, Auszahlung)."""

import math

import numpy as np
import pytest

import glo_constants as C
import glo_scenario as S


def test_settings_are_keyword_only_and_frozen():
    with pytest.raises(TypeError):
        S.Settings("Regimewechsel (Modell R)", 0.5)           # positional: ein eingefügtes Feld würde still Werte verschieben
    s = S.Settings()
    with pytest.raises(Exception):
        s.alpha = 1.0                                          # frozen


def test_finite_only_in_the_regime_model_with_at_least_two_players():
    assert not S.Settings().finite and not S.Settings(players=1).finite
    assert S.Settings(players=2).finite and S.Settings(players=1000).finite
    assert not S.Settings(model=C.MODEL_L, players=5).finite                # Modell L: N ändert nichts
    assert S.Settings().is_regime and not S.Settings(model=C.MODEL_L).is_regime


def test_every_preset_builds_a_settings_object():
    for name in C.PRESETS:
        assert isinstance(S.settings_of_preset(name), S.Settings)


def test_box_muller_hand_values():
    """u₁ = 1 − e^(−½) gibt −2 ln(1 − u₁) = 1; mit u₂ = 0: z = +1, u₂ = ½: z = −1, u₂ = ¼: z = 0."""
    u1 = 1 - math.exp(-0.5)
    assert S.normal_from_uniforms(u1, 0.0) == pytest.approx(1.0, abs=1e-14)
    assert S.normal_from_uniforms(u1, 0.5) == pytest.approx(-1.0, abs=1e-14)
    assert S.normal_from_uniforms(u1, 0.25) == pytest.approx(0.0, abs=1e-12)
    assert S.normal_from_uniforms(0.0, 0.3) == pytest.approx(0.0, abs=1e-15)       # u₁ = 0: Radius 0


def test_normals_from_the_stream_have_mean_zero_and_unit_variance():
    from glo_rng import SplitMix64
    rng = SplitMix64(5)
    z = np.array([S._normal(rng) for _ in range(20_000)])
    assert abs(z.mean()) < 4 / math.sqrt(len(z)) and abs(z.var() - 1.0) < 4 * math.sqrt(2 / len(z))


def test_round_is_deterministic_in_the_seed_and_differs_between_seeds():
    s = S.Settings(seed=7)
    a, b = S.draw_round(s, 0.5, 0.5), S.draw_round(s, 0.5, 0.5)
    c = S.draw_round(S.Settings(seed=8), 0.5, 0.5)
    assert a.theta == b.theta and np.array_equal(a.signals, b.signals) and a.theta != c.theta


def test_continuum_round_regime_start_is_theta_below_theta_hat():
    s = S.Settings(seed=7)
    r = S.draw_round(s, 0.5, 0.5)
    assert r.n == C.DRAW_CONTINUUM and r.exact and r.started == (r.theta <= 0.5) and r.joined == int((r.signals <= 0.5).sum())
    assert r.payoff_join == pytest.approx((1.0 if r.started else 0.0) - s.c)
    assert r.share == pytest.approx(0.5 * math.erfc(-math.sqrt(s.beta) * (0.5 - r.theta) / math.sqrt(2.0)), abs=1e-15)
    with pytest.raises(ValueError):
        S.draw_round(s, 0.5, None)                                          # Modell R im Kontinuum braucht θ̂


def test_finite_round_counts_all_joiners_for_the_start():
    """Alle machen mit (x* sehr groß): Anteil 1 ≥ θ, sobald θ ≤ 1; niemand (x* sehr klein): Anteil 0, Start nur für θ ≤ 0."""
    s = S.Settings(players=5, seed=11)
    everyone = S.draw_round(s, 1e9, None)
    nobody = S.draw_round(s, -1e9, None)
    assert everyone.joined == 5 and everyone.share == 1.0 and everyone.started == (everyone.theta <= 1.0) and not everyone.exact
    assert nobody.joined == 0 and nobody.share == 0.0 and nobody.started == (nobody.theta <= 0.0)
    assert everyone.n == 5 and len(everyone.signals) == 5


def test_linear_round_pays_share_minus_theta():
    s = S.Settings(model=C.MODEL_L, alpha=4.0, beta=1.0, seed=3)
    r = S.draw_round(s, 0.5, None)
    assert r.started is None and r.exact and r.payoff_join == pytest.approx(r.share - r.theta, abs=1e-15)
