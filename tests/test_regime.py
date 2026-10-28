"""Modell R (Regimewechsel): Hand-Mini-Instanzen, Band in d und in c, geschlossene Zählung gegen Wurzelsuche (Abstand zu den Berührpunkten), Stabilität, Kennzahlen."""

import math

import pytest

import glo_regime as R
from glo_core import SQ2PI, ncdf


def test_unique_hand_instance(reg44):
    r = reg44
    assert R.rho_of(r["alpha"], r["beta"]) == r["rho"] and R.d_value(r["alpha"], r["beta"], r["y"], r["c"]) == pytest.approx(r["d"], abs=1e-12)
    assert r["rho"] < SQ2PI and R.d_band(r["alpha"], r["beta"]) is None and R.w0_of(r["rho"]) is None
    e = R.roots_regime(r["alpha"], r["beta"], r["y"], r["c"])
    assert e.count == 1 and e.w[0] == pytest.approx(r["w"], abs=1e-12) and e.theta[0] == pytest.approx(r["theta"], abs=1e-12) and e.x[0] == pytest.approx(r["x"], abs=1e-12)
    assert e.stable == (True,)


def test_three_equilibria_hand_instance(reg84):
    r = reg84
    assert R.w0_of(r["rho"]) == pytest.approx(r["w0"], abs=2e-4)
    lo, hi = R.d_band(r["alpha"], r["beta"])
    assert (lo, hi) == pytest.approx(r["band"], abs=5e-4) and lo < r["d"] < hi
    e = R.roots_regime(r["alpha"], r["beta"], r["y"], r["c"])
    assert e.count == 3
    assert e.w[1] == pytest.approx(0.0, abs=1e-12) and e.theta[1] == pytest.approx(0.5, abs=1e-12) and e.x[1] == pytest.approx(0.5, abs=1e-12)     # G(0) = ρ/2 = 2 = d
    assert e.w[0] == pytest.approx(-e.w[2], abs=1e-12) and e.theta[0] + e.theta[2] == pytest.approx(1.0, abs=1e-12)       # Symmetrie bei y = c = ½
    assert e.x[0] < e.x[1] < e.x[2]
    assert e.stable == (True, False, True)


def test_c_band_hand_edges(reg84):
    r = reg84
    lo, hi = R.c_band(r["alpha"], r["beta"], r["y"])
    assert lo == pytest.approx(r["c_lower"], abs=5e-4) and hi == pytest.approx(r["c_upper"], abs=5e-4) and lo + hi == pytest.approx(1.0, abs=1e-12)
    assert R.c_band(4.0, 4.0, 0.5) is None                         # ρ = 2 < √(2π): nie mehrfach
    assert R.roots_regime(8.0, 4.0, 0.5, hi - 0.01).count == 3 and R.roots_regime(8.0, 4.0, 0.5, hi + 0.01).count == 1
    assert R.roots_regime(8.0, 4.0, 0.5, lo + 0.01).count == 3 and R.roots_regime(8.0, 4.0, 0.5, lo - 0.01).count == 1


def test_g_and_slope_at_known_points():
    assert float(R.G(0.0, 4.0)) == 2.0
    assert R.g_slope(0.0, 4.0) == pytest.approx(4.0 * 0.3989422804014327 - 1.0, abs=1e-14)       # ρφ(0) − 1 = 0,5958 > 0: G steigt im mittleren Stück
    assert R.g_slope(3.0, 4.0) < 0 and R.g_slope(-3.0, 4.0) < 0


def test_theta_x_is_increasing_in_w():
    ws = [-3.0, -1.0, 0.0, 0.5, 2.0]
    th = [R.theta_x(w, 4.0)[0] for w in ws]
    xs = [R.theta_x(w, 4.0)[1] for w in ws]
    assert th == sorted(th) and xs == sorted(xs) and R.theta_x(0.0, 4.0) == (0.5, 0.5)


def test_closed_form_count_agrees_with_the_root_count_away_from_the_band_edges():
    tested = skipped = 0
    for a in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        for b in (0.1, 0.5, 1.0, 4.0, 16.0, 64.0):
            band = R.d_band(a, b)
            for y in (0.2, 0.5, 0.8):
                for c in (0.2, 0.35, 0.5, 0.65, 0.8):
                    d = R.d_value(a, b, y, c)
                    if band is not None and (abs(d - band[0]) < 1e-5 or abs(d - band[1]) < 1e-5):
                        skipped += 1
                        continue
                    assert R.roots_regime(a, b, y, c).count == R.count_closed(a, b, y, c), (a, b, y, c)
                    tested += 1
    assert tested > 500 and skipped < 5


def test_formula_is_the_uniqueness_bound_over_all_y_and_c():
    """β ≥ α²/(2π) ⟺ ρ ≤ √(2π): dann gibt es für jedes (y, c) genau eine Wurzel; knapp darunter gibt es (y, c) mit drei."""
    for a in (1.0, 4.0, 8.0, 16.0):
        b = a * a / (2 * math.pi)
        assert R.formula_holds(a, b * 1.001) and not R.formula_holds(a, b * 0.999)
        assert all(R.roots_regime(a, b * 1.001, y, c).count == 1 for y in (0.2, 0.5, 0.8) for c in (0.1, 0.3, 0.5, 0.7, 0.9))
        # unterhalb: das Band in c ist nicht leer (bei y = ½ symmetrisch um ½), also gibt es ein c mit drei Wurzeln
        lo, hi = R.c_band(a, b * 0.9, 0.5)
        assert R.roots_regime(a, b * 0.9, 0.5, 0.5 * (lo + hi)).count == 3


def test_equilibrium_satisfies_the_x_form_of_the_indifference():
    """Rückprobe auf dem anderen Weg: bei x* ist P(θ ≤ θ̂ | x*) = c und l(θ̂) = θ̂."""
    for a, b, y, c in ((8, 4, 0.5, 0.5), (8, 4, 0.3, 0.65), (1, 16, 0.5, 0.25), (4, 4, 0.7, 0.4)):
        e = R.roots_regime(a, b, y, c)
        for th, x in zip(e.theta, e.x):
            m = (a * y + b * x) / (a + b)
            assert ncdf(math.sqrt(a + b) * (th - m)) == pytest.approx(c, abs=1e-9)
            assert ncdf(math.sqrt(b) * (x - th)) == pytest.approx(th, abs=1e-9)


def test_known_state_probability_hand_values():
    assert R.p_known_state_in_unit(1.0, 0.5) == pytest.approx(0.3829, abs=5e-5)         # Φ(½) − Φ(−½)
    assert R.p_known_state_in_unit(4.0, 0.5) == pytest.approx(0.6827, abs=5e-5)         # Φ(1) − Φ(−1)


def test_start_and_lost_probabilities_add_up():
    e = R.roots_regime(4.0, 64.0, 0.5, 0.75)
    th = e.theta[0]
    unit = R.p_known_state_in_unit(4.0, 0.5)
    assert R.p_start(4.0, 0.5, th) < ncdf(math.sqrt(4.0) * 0.5)             # P(Start) < P(θ < 1)
    assert R.p_lost(4.0, 0.5, th) + R.p_start(4.0, 0.5, th) == pytest.approx(ncdf(2.0 * 0.5), abs=1e-14)
    assert 0.0 < R.p_lost(4.0, 0.5, th) < unit + 1e-12
