"""Modell L (linear): Hand-Mini-Instanz, Wurzeln auf allen drei Stücken, Randwurzeln, Unterlauf, geschlossene Formen gegen die Wurzelzählung (mit Abstand zu den Berührpunkten)."""

import math

import pytest

import glo_linear as L
from glo_core import SQ2PI, ncdf


def test_kappa_hand_values(lin2):
    assert L.kappa(lin2["alpha"], lin2["beta"]) == pytest.approx(lin2["kappa"], abs=1e-15)
    assert L.kappa(4.0, 1.0) == pytest.approx(4.0 * math.sqrt(5.0 / 6.0), abs=1e-15)            # 3,6515


def test_x_star_and_share_of_x_are_inverse(lin2):
    a, b, y = lin2["alpha"], lin2["beta"], lin2["y"]
    assert L.x_star(0.5, a, b, y) == pytest.approx(lin2["x_mid"], abs=1e-15)
    for m in (0.0, 0.2, 0.9, 1.0):
        assert L.share_of_x(L.x_star(m, a, b, y), a, b, y) == pytest.approx(m, abs=1e-14)


def test_mini_instance_has_the_hand_root_and_it_is_the_only_one(lin2):
    e = L.roots_linear(lin2["alpha"], lin2["beta"], lin2["y"])
    assert e.count == 1 and e.m[0] == pytest.approx(0.5, abs=1e-12) and e.x[0] == pytest.approx(0.5, abs=1e-12) and e.stable == (True,) and not e.underflow


def test_critical_beta_of_the_mini_instance_and_the_count_flips_there(lin2):
    assert L.critical_beta(2.0) == pytest.approx(lin2["beta_crit"], abs=5e-5)
    crit = L.critical_beta(2.0)
    assert L.roots_linear(2.0, crit * 1.02, 0.5).count == 1
    assert L.roots_linear(2.0, crit * 0.98, 0.5).count == 3


def test_three_roots_below_the_critical_beta_with_unstable_middle(lin2):
    e = L.roots_linear(2.0, 0.4, 0.5)
    assert e.count == 3 and e.stable == (True, False, True)
    assert e.m[1] == pytest.approx(0.5, abs=1e-12)                              # bei y = ½ ist m = ½ immer eine Wurzel
    assert e.m[0] + e.m[2] == pytest.approx(1.0, abs=1e-12)                      # und die beiden äußeren liegen symmetrisch
    assert e.x[0] < e.x[1] < e.x[2]


@pytest.mark.parametrize("alpha,beta", [(1.0, 5.0), (4.0, 1.0), (10.0, 0.5), (3.0, 20.0)])
def test_y_one_half_always_has_the_root_one_half(alpha, beta):
    assert any(abs(m - 0.5) < 1e-10 for m in L.roots_linear(alpha, beta, 0.5).m)


def test_edge_roots_at_small_beta_lie_far_outside_any_fixed_grid():
    """α = 4, β = 0,1, y = ½ (V3-Zelle): m ≈ 2,07·10^-10 und 1 − 2,07·10^-10, x* ≈ −20 und 21. Ein x-Gitter y ± 6 sähe sie nicht."""
    e = L.roots_linear(4.0, 0.1, 0.5)
    assert e.count == 3 and not e.underflow
    assert 1e-11 < e.m[0] < 1e-8 and 1e-11 < 1.0 - e.m[2] < 1e-8
    assert e.x[0] == pytest.approx(-20.0, abs=1e-6) and e.x[2] == pytest.approx(21.0, abs=1e-6) and e.x[1] == pytest.approx(0.5, abs=1e-9)


def test_underflow_flag_when_the_edge_root_is_below_1e_300():
    """κ·min(y, 1−y) > 37: Φ(−κy) liegt unter 10^-300, die Wurzel ist nicht darstellbar, gemeldet als 0 bzw. 1 mit Flag; x* ist trotzdem endlich."""
    a, b = 100.0, 1.0
    assert L.kappa(a, b) * 0.5 > 37.0
    e = L.roots_linear(a, b, 0.5)
    assert e.underflow and e.count == 3 and e.m[0] == 0.0 and e.m[2] == 1.0
    assert e.x[0] == pytest.approx(-a * 0.5 / b, rel=1e-12) and math.isfinite(e.x[2])


def test_no_underflow_flag_in_ordinary_cells():
    assert not any(L.roots_linear(a, b, y).underflow for a, b, y in ((4, 1, 0.5), (2, 0.4, 0.3), (10, 10, 0.5), (1, 100, 0.3)))


def test_far_prior_has_one_edge_root():
    """α = 4, β = 1, y = 0,3: nur ein Gleichgewicht, fast alle machen mit (m ≈ 0,994); eine Wurzel am Rand 1."""
    e = L.roots_linear(4.0, 1.0, 0.3)
    assert e.count == 1 and 0.99 < e.m[0] < 1.0 and e.x[0] > 3.0 and e.stable == (True,)


def test_root_satisfies_the_reduced_equation_in_every_cell():
    for a, b, y in ((4, 1, 0.5), (2, 0.4, 0.5), (4, 0.1, 0.5), (6, 2, 0.6), (10, 2, 0.4), (1, 100, 0.3), (20, 5, 0.5)):
        e = L.roots_linear(a, b, y)
        for m in e.m:
            assert ncdf(e.kappa * (m - y)) == pytest.approx(m, rel=1e-9, abs=1e-14), (a, b, y, m)


def test_closed_form_count_agrees_with_the_root_count_away_from_the_fold():
    """Faltenkurve gegen Wurzelzählung auf einem Raster; Zellen innerhalb 10^-4 der Kante oder von κ = √(2π) werden nicht gezählt (Berührpunkte)."""
    tested = skipped = 0
    for a in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        for b in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0):
            kap = L.kappa(a, b)
            top = L.fold_edge(kap)
            for y in (0.2, 0.35, 0.5, 0.65, 0.8):
                if abs(kap - SQ2PI) < 1e-4 or (top is not None and (abs(y - top) < 1e-4 or abs(y - (1 - top)) < 1e-4)):
                    skipped += 1
                    continue
                assert L.roots_linear(a, b, y).count == L.count_closed(a, b, y), (a, b, y)
                tested += 1
    assert tested > 200 and skipped < 10


def test_fold_edge_is_none_up_to_the_sqrt_2pi_and_half_just_above():
    assert L.fold_edge(SQ2PI) is None and L.fold_edge(1.0) is None
    assert L.fold_edge(SQ2PI * 1.0001) == pytest.approx(0.5, abs=1e-3)
    assert L.fold_edge(10.0) > L.fold_edge(5.0) > 0.5        # der Bereich wächst mit κ


def test_formula_is_sufficient_but_not_necessary():
    """β ≥ α²/(2π) hinreichend: kein Raster-y hat mehrere Wurzeln. Nicht notwendig: (α, β) = (4, 2) liegt unter der Formel (4²/(2π) = 2,546 > 2) und ist bei y = ½ trotzdem eindeutig."""
    assert L.formula_holds(4.0, 3.0) and not L.formula_holds(4.0, 2.0)
    for a in (1.0, 4.0, 10.0):
        b = a * a / (2 * math.pi) * 1.01
        assert all(L.roots_linear(a, b, y).count == 1 for y in (0.1, 0.3, 0.5, 0.7, 0.9))
    assert L.roots_linear(4.0, 2.0, 0.5).count == 1 and not L.formula_holds(4.0, 2.0)


def test_slope_sign_matches_stability_and_g_values_vanish_at_roots():
    e = L.roots_linear(2.0, 0.4, 0.5)
    g = L.g_values(list(e.m), 2.0, 0.4, 0.5)
    assert all(abs(v) < 1e-12 for v in g)
    assert [L.slope(m, e.kappa, 0.5) < 0 for m in e.m] == list(e.stable)
