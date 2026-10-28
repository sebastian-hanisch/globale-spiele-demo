"""Endlich viele Spieler (Modell R): Hand-Mini-Instanzen der Binomialzahl, Startwahrscheinlichkeit, Suchfenster, Zählung der Gleichgewichte mit Abstand zu den Berührpunkten, Stabilität."""

import math

import numpy as np
import pytest

import glo_finite as F
import glo_iteration as I
import glo_regime as R
from glo_core import ncdf

CELL = dict(alpha=8.0, beta=4.0, y=0.5)


def f_values(n, c, points=401, cell=CELL):
    lo, hi = F.search_window(n, cell["alpha"], cell["beta"], cell["y"], c)
    xs = np.linspace(lo, hi, points)
    return xs, np.array([F.success_prob(float(x), float(x), n, cell["alpha"], cell["beta"], cell["y"]) - c for x in xs])


def test_needed_others_hand_values():
    assert F.needed_others(0.7, 2) == 1 and F.needed_others(0.5, 3) == 1                # (1 + K)/2 ≥ 0,7 ⟺ K ≥ 0,4; (1 + K)/3 ≥ ½ ⟺ K ≥ 0,5
    assert F.needed_others(1 / 3, 3) == 0 and F.needed_others(0.3, 3) == 0 and F.needed_others(-1.0, 5) == 0    # θ ≤ 1/N: der eigene Beitrag genügt
    assert F.needed_others(1.0, 3) == 2 and F.needed_others(1.2, 3) == 3                 # θ = 1: alle anderen; θ > 1: nie erfüllbar


def test_binom_tail_hand_values():
    assert F.binom_tail(1, 2, 0.5) == pytest.approx(0.75, abs=1e-15) and F.binom_tail(2, 2, 0.5) == pytest.approx(0.25, abs=1e-15)
    assert F.binom_tail(0, 2, 0.5) == 1.0 and F.binom_tail(3, 2, 0.5) == 0.0
    assert F.binom_tail(2, 3, 0.3) == pytest.approx(3 * 0.3 ** 2 * 0.7 + 0.3 ** 3, abs=1e-15)         # P(K ≥ 2), K ~ Bin(3, 0,3) = 0,216
    assert F.binom_tail(1, 5, 0.0) == 0.0 and F.binom_tail(5, 5, 1.0) == 1.0


def test_success_given_theta_hand_instances(finite_hand):
    for key in ("n2", "n3", "sure"):
        n, theta, p, want = finite_hand[key]
        assert F.success_given_theta(theta, p, n) == pytest.approx(want, abs=1e-14), key
    assert F.success_given_theta(1.5, 0.99, 4) == 0.0


def test_success_given_theta_is_monotone_in_p_and_theta():
    ps = [F.success_given_theta(0.6, p, 10) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    ts = [F.success_given_theta(t, 0.5, 10) for t in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert ps == sorted(ps) and ts == sorted(ts, reverse=True)


def test_success_prob_is_a_probability_and_falls_in_the_signal_of_a_fixed_threshold():
    vals = [F.success_prob(s, 0.5, 5, 8.0, 4.0, 0.5) for s in np.linspace(-1.5, 3.0, 10)]
    assert all(0.0 <= v <= 1.0 for v in vals) and vals == sorted(vals, reverse=True)           # hohes Signal: θ vermutlich hoch, Start unwahrscheinlicher


def test_success_prob_rises_with_the_threshold_of_the_others():
    vals = [F.success_prob(0.5, x, 10, 8.0, 4.0, 0.5) for x in (-1.0, 0.0, 0.5, 1.0, 2.0)]
    assert vals == sorted(vals)


def test_large_n_comes_close_to_the_continuum():
    """N = 1000 gegen die Kontinuumsformel P = Φ(√(α+β)(θ̂ − m)): Unterschied höchstens 0,01 (Größenordnung 1/N)."""
    for x in (0.0, 0.5, 1.0):
        m = (8 * 0.5 + 4 * x) / 12
        cont = ncdf(math.sqrt(12) * (I.theta_hat(x, 4.0) - m))
        assert abs(F.success_prob(x, x, 1000, 8.0, 4.0, 0.5) - cont) < 0.01, x


def test_search_window_brackets_all_roots_in_several_cells():
    for n in (2, 5, 25):
        for c in (0.3, 0.5, 0.7):
            lo, hi = F.search_window(n, 8.0, 4.0, 0.5, c)
            assert F.success_prob(lo, lo, n, 8.0, 4.0, 0.5) - c > 0 > F.success_prob(hi, hi, n, 8.0, 4.0, 0.5) - c, (n, c)


@pytest.mark.parametrize("n,count", [(2, 1), (3, 1), (5, 1), (10, 3), (25, 3), (100, 3)])
def test_equilibrium_count_in_the_three_equilibria_cell(n, count):
    """Zelle α = 8, β = 4, y = c = ½ (Kontinuum: drei). Das Raster ist weit von jedem Berührpunkt: abseits der Wurzeln ist |P − c| mindestens 5·10^-4."""
    e = F.roots_finite(n, 8.0, 4.0, 0.5, 0.5)
    assert e.count == count and e.scan_points == F.scan_points_for(n)
    xs, f = f_values(n, 0.5, 201)
    near = np.zeros(len(f), bool)
    for i in np.where(np.sign(f[1:]) * np.sign(f[:-1]) < 0)[0]:
        near[max(0, i - 3):i + 5] = True
    assert np.min(np.abs(f[~near])) > 5e-4
    if count == 3:
        assert e.stable == (True, False, True)
    else:
        assert e.stable == (True,)


@pytest.mark.parametrize("n,count", [(5, 3), (10, 3), (25, 1), (100, 1)])
def test_just_outside_the_band_few_players_can_add_equilibria(n, count):
    """c = 0,6 liegt außerhalb des Kontinuumsbandes (Rand 0,5837; dort eins): bei N = 5 und 10 gibt es drei."""
    assert R.roots_regime(8.0, 4.0, 0.5, 0.6).count == 1
    assert F.roots_finite(n, 8.0, 4.0, 0.5, 0.6).count == count


def test_few_players_remove_equilibria_in_the_three_equilibria_cell():
    assert R.roots_regime(8.0, 4.0, 0.5, 0.5).count == 3 and F.roots_finite(3, 8.0, 4.0, 0.5, 0.5).count == 1


def test_unique_cell_stays_unique_and_thresholds_fall_toward_the_continuum():
    """ρ = 2 (eindeutig): für jedes N eine Wurzel, oberhalb des Kontinuumswerts ½, mit wachsendem N fallend (1,409 bei N = 2 bis 0,5049 bei 1 000, Vormessung)."""
    xs = []
    for n in (2, 5, 10, 100):
        e = F.roots_finite(n, 4.0, 4.0, 0.5, 0.5)
        assert e.count == 1
        xs.append(e.x[0])
    assert xs == sorted(xs, reverse=True) and xs[-1] > 0.5 and 1.3 < xs[0] < 1.5


def test_worst_case_slider_maximum_runs_through():
    """N = 1 000 am Reglermaximum (grobes Raster, ohne Zeit-Assertion): drei Gleichgewichte, mittleres nahe ½."""
    e = F.roots_finite(1000, 8.0, 4.0, 0.5, 0.5, scan=41)
    assert e.count == 3 and 0.45 < e.x[1] < 0.5 and e.stable == (True, False, True)


def test_roots_are_roots_and_ordered():
    e = F.roots_finite(10, 8.0, 4.0, 0.5, 0.5)
    assert list(e.x) == sorted(e.x)
    for x in e.x:
        assert F.success_prob(x, x, 10, 8.0, 4.0, 0.5) == pytest.approx(0.5, abs=1e-9)


def test_best_reply_fixed_at_the_roots_and_unstable_in_the_middle():
    """Stabilität bei N = 10 (eigene Herleitung: stabil genau dann, wenn P_N(x;x) − c dort fällt): die beste Antwort auf x* ist x*; eine kleine Abweichung wächst in der Mitte und schrumpft außen."""
    e = F.roots_finite(10, 8.0, 4.0, 0.5, 0.5)
    for x in e.x:
        assert F.best_reply(x, 10, 8.0, 4.0, 0.5, 0.5) == pytest.approx(x, abs=1e-6)
    delta = 0.02
    low, mid, high = e.x
    assert abs(F.best_reply(low + delta, 10, 8.0, 4.0, 0.5, 0.5) - low) < delta
    assert abs(F.best_reply(high - delta, 10, 8.0, 4.0, 0.5, 0.5) - high) < delta
    assert F.best_reply(mid + delta, 10, 8.0, 4.0, 0.5, 0.5) - mid > delta
    assert mid - F.best_reply(mid - delta, 10, 8.0, 4.0, 0.5, 0.5) > delta
