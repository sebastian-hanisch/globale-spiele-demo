"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): kleinste Instanzen, inklusive Grenzen, Reglerränder, Entartungen."""

import math

import numpy as np
import pytest

import glo_constants as C
import glo_email as EM
import glo_evaluation as V
import glo_experiments as X
import glo_finite as F
import glo_iteration as I
import glo_linear as L
import glo_regime as R
import glo_scenario as S
from glo_core import SQ2PI


def test_two_players_is_the_smallest_finite_game():
    """N = 2: der eigene Beitrag zählt, K ~ Bin(1, p). Ein Betrieb mit sehr niedrigem Signal hält θ ≤ ½ für sicher: Start fast sicher."""
    assert F.success_prob(-5.0, 0.0, 2, 8.0, 4.0, 0.5) == pytest.approx(1.0, abs=1e-9)
    assert F.success_prob(5.0, 0.0, 2, 8.0, 4.0, 0.5) == pytest.approx(0.0, abs=1e-3)      # Posterior-Mittel 2: θ ≤ 1 hat Wahrscheinlichkeit Φ(−3,46) = 2,7·10^-4
    assert F.roots_finite(2, 8.0, 4.0, 0.5, 0.5).count == 1
    assert F.nodes_per_piece(2) == 8 and F.search_window(2, 8.0, 4.0, 0.5, 0.5)[0] < F.search_window(2, 8.0, 4.0, 0.5, 0.5)[1]


def test_formula_boundary_is_inclusive():
    """β = α²/(2π) genau: die Formel gilt (≥), ρ = √(2π) gibt keine Wendestellen, ein Gleichgewicht."""
    a = 8.0
    b = a * a / (2 * math.pi)
    assert R.formula_holds(a, b) and L.formula_holds(a, b)
    assert R.w0_of(R.rho_of(a, b)) in (None, pytest.approx(0.0, abs=1e-6)) or R.rho_of(a, b) <= SQ2PI * (1 + 1e-12)
    assert R.roots_regime(a, b, 0.5, 0.5).count == 1


def test_the_attack_threshold_is_inclusive():
    """Ein Spieler greift an, wenn die Überzeugung mindestens q ist: bei q = (1−ε)/(2−ε) genau greift der Typ 1 noch an."""
    g = EM.build(0.1, 0.5, 40)
    assert EM.greatest_equilibrium(g, EM.q_star(0.1)).attacks
    assert not EM.greatest_equilibrium(g, EM.q_star(0.1) * (1 + 1e-6)).attacks


@pytest.mark.parametrize("a,b,y", [(C.ALPHA_MIN, C.BETA_MIN, C.Y_MIN), (C.ALPHA_MIN, C.BETA_MAX, C.Y_MAX), (C.ALPHA_MAX, C.BETA_MIN, C.Y_MIN), (C.ALPHA_MAX, C.BETA_MAX, C.Y_MAX),
                                   (C.ALPHA_MAX, C.BETA_MIN, 0.5), (C.ALPHA_MIN, C.BETA_MIN, 0.5)])
def test_linear_model_at_the_slider_corners(a, b, y):
    e = L.roots_linear(a, b, y)
    assert e.count in (1, 3) and all(math.isfinite(x) for x in e.x) and list(e.x) == sorted(e.x)
    assert e.count == L.count_closed(a, b, y) or e.underflow


@pytest.mark.parametrize("a,b,y,c", [(C.ALPHA_MIN, C.BETA_MIN, C.Y_MIN, C.C_MIN), (C.ALPHA_MAX, C.BETA_MAX, C.Y_MAX, C.C_MAX), (C.ALPHA_MAX, C.BETA_MIN, 0.5, 0.5), (C.ALPHA_MIN, C.BETA_MAX, 0.5, 0.95),
                                     (C.ALPHA_MAX, C.BETA_MIN, C.Y_MAX, C.C_MIN)])
def test_regime_model_at_the_slider_corners(a, b, y, c):
    e = R.roots_regime(a, b, y, c)
    assert e.count in (1, 3) and all(math.isfinite(x) for x in e.x) and all(0.0 <= t <= 1.0 for t in e.theta)
    assert e.count == R.count_closed(a, b, y, c)


def test_finite_n_at_the_corners_of_the_search_window():
    for n, a, b, y, c in ((2, C.ALPHA_MAX, C.BETA_MIN, 0.5, 0.5), (2, C.ALPHA_MIN, C.BETA_MAX, C.Y_MAX, C.C_MAX), (5, C.ALPHA_MAX, C.BETA_MAX, C.Y_MIN, C.C_MIN)):
        e = F.roots_finite(n, a, b, y, c, scan=41)
        assert e.count >= 1 and all(math.isfinite(x) for x in e.x)


def test_iteration_at_extreme_starts_and_extreme_parameters():
    e = R.roots_regime(C.ALPHA_MAX, C.BETA_MAX, 0.5, 0.5)
    assert e.count == 3
    for x0, target in ((I.START_LOW, e.x[0]), (I.START_HIGH, e.x[-1])):
        r = I.iterate_regime(C.ALPHA_MAX, C.BETA_MAX, 0.5, 0.5, x0, target, cap=5000)
        assert r.reached and r.rounds < 5000
    e = R.roots_regime(C.ALPHA_MIN, C.BETA_MIN, C.Y_MIN, C.C_MIN)
    assert I.iterate_regime(C.ALPHA_MIN, C.BETA_MIN, C.Y_MIN, C.C_MIN, I.START_HIGH, e.x[0], cap=20000).rounds >= 1


def test_start_at_the_equilibrium_needs_zero_rounds():
    e = R.roots_regime(8.0, 4.0, 0.5, 0.5)
    assert I.iterate_regime(8.0, 4.0, 0.5, 0.5, e.x[0], e.x[0]).rounds == 0
    assert I.iterate_linear(4.0, 1.0, 0.5, 0.5, 0.5).rounds == 0


def test_email_with_no_message_level_and_with_certain_loss():
    g0 = EM.build(0.1, 0.5, 0)
    assert len(g0.prob) == 3 and g0.prob.sum() == pytest.approx(1.0)
    assert not EM.greatest_equilibrium(g0, 0.5).attacks
    sim = EM.simulate_protocol(1.0, 0.5, 1000, 3)                    # ε = 1: die erste Nachricht geht immer verloren
    assert (sim["k"][sim["is_g"]] == 1).all() and (sim["k"][~sim["is_g"]] == 0).all()
    sim = EM.simulate_protocol(0.3, 1.0, 500, 3)
    assert sim["is_g"].all() and (sim["k"] >= 1).all()
    sim = EM.simulate_protocol(0.3, 0.0, 500, 3)
    assert not sim["is_g"].any() and (sim["k"] == 0).all()


def test_email_extreme_slider_values():
    for eps in (C.EMAIL_EPS_MIN, C.EMAIL_EPS_MAX):
        for loss in (C.EMAIL_L_MIN, C.EMAIL_L_MAX):
            sm = X.e6_summary(eps, loss)
            assert sm["common_states"] <= sm["g_states"] == 801 and 0 < sm["q"] < 1


def test_every_settings_corner_runs_through_the_main_view():
    for model in C.MODELS:
        for players in (0, 2):
            s = S.Settings(model=model, y=C.Y_MIN, alpha=C.ALPHA_MAX, beta=C.BETA_MIN, c=C.C_MAX, players=players)
            v = V.analyse(s)
            assert v.count >= 1
            for verdict in (V.verdict_count(v), V.verdict_formula(v), V.verdict_no_uncertainty(v)):
                assert verdict[0] in ("success", "warning", "info", "error") and verdict[1]


def test_known_state_probability_for_extreme_priors():
    assert R.p_known_state_in_unit(C.ALPHA_MAX, C.Y_MAX) == pytest.approx(0.7081, abs=1e-4)           # Φ(√30·0,1) − Φ(−√30·0,9)
    assert 0.0 < R.p_known_state_in_unit(C.ALPHA_MIN, 0.5) < 1.0
    assert np.isfinite(R.p_lost(C.ALPHA_MAX, C.Y_MIN, 0.0))
