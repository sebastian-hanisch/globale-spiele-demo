"""Beste-Antwort-Iteration (Kontinuum): Fixpunkte, Konvergenz zu den äußeren Wurzeln, Stabilität der mittleren Wurzel (H11), Verlangsamung an der Grenze. Rundenzahlen nur als Bänder."""

import math

import pytest

import glo_iteration as I
import glo_linear as L
import glo_regime as R
from glo_core import npdf, ncdf


def test_theta_hat_is_the_fixed_point_and_monotone_in_x():
    for beta in (0.5, 4.0, 100.0):
        previous = -1.0
        for x in (-5.0, -1.0, 0.0, 0.5, 1.0, 3.0, 20.0):
            th = I.theta_hat(x, beta)
            assert 0.0 <= th <= 1.0 and th == pytest.approx(ncdf(math.sqrt(beta) * (x - th)), abs=1e-12)
            assert th >= previous
            previous = th
    assert I.theta_hat(-50.0, 4.0) < 1e-100 and I.theta_hat(50.0, 4.0) == pytest.approx(1.0, abs=1e-12)


def test_theta_hat_warm_start_gives_the_same_value():
    assert I.theta_hat(0.7, 4.0, start=0.2) == pytest.approx(I.theta_hat(0.7, 4.0), abs=1e-14)
    assert I.theta_hat(0.7, 4.0, start=0.9999999) == pytest.approx(I.theta_hat(0.7, 4.0), abs=1e-14)


def test_best_reply_fixes_every_equilibrium_including_the_middle_one(reg84):
    r = reg84
    e = R.roots_regime(r["alpha"], r["beta"], r["y"], r["c"])
    for x in e.x:
        xn, th = I.best_reply_regime(x, r["alpha"], r["beta"], r["y"], r["c"])
        assert xn == pytest.approx(x, abs=1e-10)


def test_best_reply_hand_value(reg84):
    """x = ½, β = 4: θ̂ = ½ (Φ(2(½ − ½)) = ½); x' = ((α+β)(θ̂ − 0) − αy)/β = (12·½ − 4)/4 = ½."""
    xn, th = I.best_reply_regime(0.5, 8.0, 4.0, 0.5, 0.5)
    assert th == pytest.approx(0.5, abs=1e-14) and xn == pytest.approx(0.5, abs=1e-14)


def test_iteration_from_both_ends_reaches_the_outer_equilibria(reg84):
    r = reg84
    e = R.roots_regime(r["alpha"], r["beta"], r["y"], r["c"])
    low = I.iterate_regime(r["alpha"], r["beta"], r["y"], r["c"], I.START_LOW, e.x[0])
    high = I.iterate_regime(r["alpha"], r["beta"], r["y"], r["c"], I.START_HIGH, e.x[2])
    assert low.reached and high.reached and low.monotone and high.monotone
    assert abs(low.last - e.x[0]) < 1e-6 and abs(high.last - e.x[2]) < 1e-6
    assert 8 <= low.rounds <= 20 and 8 <= high.rounds <= 20            # 12 gemessen; Band wegen der Toleranz-Kante


def test_h11_middle_equilibrium_is_unstable_in_the_regime_model(reg84):
    """H11, erster Weg: Iteration knapp unter / über der mittleren Wurzel läuft von ihr weg, zur unteren bzw. oberen Wurzel."""
    r = reg84
    e = R.roots_regime(r["alpha"], r["beta"], r["y"], r["c"])
    for delta, target in ((-1e-3, e.x[0]), (1e-3, e.x[2])):
        res = I.iterate_regime(r["alpha"], r["beta"], r["y"], r["c"], e.x[1] + delta, target)
        assert res.reached and res.monotone and (res.last - e.x[1]) * delta > 0


def _derivative_of_best_reply(x, a, b, y, c, h=1e-5):
    up = I.best_reply_regime(x + h, a, b, y, c)[0]
    down = I.best_reply_regime(x - h, a, b, y, c)[0]
    return (up - down) / (2 * h)


def test_h11_slope_of_the_best_reply_matches_the_sign_of_g_prime(reg84):
    """H11, zweiter Weg: B'(x*) aus der zentralen Differenz gegen die geschlossene Form B' = (α+β)/β · s/(1+s), s = √β φ(w); B' < 1 ⟺ G'(w) < 0."""
    a, b, y, c = reg84["alpha"], reg84["beta"], reg84["y"], reg84["c"]
    e = R.roots_regime(a, b, y, c)
    for w, x, stable in zip(e.w, e.x, e.stable):
        s = math.sqrt(b) * npdf(w)
        closed = (a + b) / b * s / (1 + s)
        numeric = _derivative_of_best_reply(x, a, b, y, c)
        assert numeric == pytest.approx(closed, rel=1e-5)
        assert (numeric < 1) == stable == (R.g_slope(w, e.rho) < 0)


def test_unique_equilibria_are_stable_in_many_cells():
    for a, b, y, c in ((4, 4, 0.5, 0.5), (1, 16, 0.3, 0.7), (8, 4, 0.5, 0.65), (16, 4, 0.5, 0.1), (2, 0.5, 0.8, 0.5)):
        e = R.roots_regime(a, b, y, c)
        if e.count == 1:
            assert e.stable == (True,), (a, b, y, c)


def test_rounds_grow_toward_the_uniqueness_boundary():
    """α = 8, y = c = ½: Grenze β = 64/(2π) = 10,186. Je näher β der Grenze von unten kommt, desto länger die Iteration; hinter der Grenze fällt sie wieder.
    Gemessen 88 / 1 362 / 35 325 für β = 8 / 10 / 10,2 und 163 für β = 16 (Bänder, nicht exakt)."""
    def rounds(beta):
        e = R.roots_regime(8.0, beta, 0.5, 0.5)
        return I.iterate_regime(8.0, beta, 0.5, 0.5, I.START_LOW, e.x[0]).rounds
    r8, r10, r102, r16 = rounds(8.0), rounds(10.0), rounds(10.2), rounds(16.0)
    assert r8 < r10 < r102 and r10 > 5 * r8 and r102 > 10 * r10 and r16 < r102 / 50
    assert 20_000 < r102 < 60_000


def test_iteration_reports_the_cap_instead_of_hanging():
    res = I.iterate_regime(8.0, 10.2, 0.5, 0.5, I.START_LOW, R.roots_regime(8.0, 10.2, 0.5, 0.5).x[0], cap=1000)
    assert not res.reached and res.rounds == 1000


def test_linear_iteration_reaches_the_outer_roots_and_slows_at_the_boundary():
    e = L.roots_linear(4.0, 1.0, 0.5)
    lo = I.iterate_linear(4.0, 1.0, 0.5, 0.0, e.m[0])
    hi = I.iterate_linear(4.0, 1.0, 0.5, 1.0, e.m[2])
    assert lo.reached and hi.reached and lo.monotone and hi.monotone
    crit = L.critical_beta(4.0)
    def rounds(beta):
        ee = L.roots_linear(4.0, beta, 0.5)
        return I.iterate_linear(4.0, beta, 0.5, 0.0, ee.m[0]).rounds
    assert rounds(crit * 0.99) > 10 * rounds(1.0) and rounds(crit * 1.05) < rounds(crit * 0.99)


def test_h11_middle_equilibrium_is_unstable_in_the_linear_model():
    e = L.roots_linear(4.0, 1.0, 0.5)
    for delta, target in ((-1e-3, e.m[0]), (1e-3, e.m[2])):
        res = I.iterate_linear(4.0, 1.0, 0.5, e.m[1] + delta, target)
        assert res.reached and (res.last - e.m[1]) * delta > 0
    kap = e.kappa
    assert [kap * npdf(kap * (m - 0.5)) < 1 for m in e.m] == list(e.stable) == [True, False, True]


def test_trajectory_starts_at_x0_and_is_monotone():
    path = I.trajectory_regime(8.0, 4.0, 0.5, 0.5, -50.0, 15)
    assert path[0] == -50.0 and len(path) == 16
    assert all(b >= a for a, b in zip(path, path[1:]))
