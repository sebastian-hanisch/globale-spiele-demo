"""Jede Zahl aus der README, nachgerechnet über dieselben Funktionen wie in der App (DEMO-PLAYBOOK §4).

Plattformregeln: Zahlen aus Wurzelsuche und geschlossenen Formen mit Rundung auf die README-Stellen (die Rechnung ist libm-genau bis auf Ulp); Rundenzahlen der Iteration nur als Bänder
(2 % oder mindestens 2 Runden); Zählungen endlicher N mit dem Abstandstest der Rastertests in `test_finite.py`; Simulationen mit festem SplitMix64-Strom (Ganzzahlarithmetik, plattformgleich)
und Marge 4 Standardfehler. Nichts hier hängt an einem Löser (kein LAPACK, kein scipy-Wurzelfinder)."""

import math

import numpy as np
import pytest

import glo_email as EM
import glo_experiments as X
import glo_finite as F
import glo_grid as GRID
import glo_iteration as I
import glo_linear as L
import glo_regime as R
from glo_core import SQ2PI, ncdf


@pytest.fixture(scope="module")
def grid():
    return X.load_grid()


def r3(x):
    return round(x, 3)


def r4(x):
    return round(x, 4)


def band(value, stated, rel=0.02, minimum=2):
    return abs(value - stated) <= max(minimum, rel * stated)


# --- Einleitung ---------------------------------------------------------------------------------------------------------------------------------
def test_without_uncertainty_the_two_equilibria_cover_38_and_68_percent():
    """„θ in (0, 1) mit Wahrscheinlichkeit 0,383 bei α = 1 und 0,683 bei α = 4 (y = ½)“."""
    assert r3(R.p_known_state_in_unit(1.0, 0.5)) == 0.383 and r3(R.p_known_state_in_unit(4.0, 0.5)) == 0.683


# --- H1, H2 -------------------------------------------------------------------------------------------------------------------------------------
def test_h1_raster_counts(grid):
    s = X.grid_summary(grid)
    assert (len(grid["alphas"]), len(grid["betas"])) == (29, 31)
    assert s["regime"]["above"] == {"cells": 11975, "multi": 0} and s["regime"]["below"] == {"cells": 10500, "multi": 5206} and s["regime"]["total"] == 22475
    assert s["linear"]["above"] == {"cells": 4311, "multi": 0} and s["linear"]["below"] == {"cells": 3780, "multi": 1846} and s["linear"]["total"] == 8091
    assert round(100 * s["regime"]["below_share"], 1) == 49.6 and round(100 * s["linear"]["below_share"], 1) == 48.8


def test_h2_regime_formula_is_the_uniqueness_bound_over_all_y_and_c(grid):
    assert X.grid_below_formula_without_multi(grid, "regime") == 0
    n_below = sum(1 for a in GRID.ALPHAS for b in GRID.BETAS if b < GRID.formula_bound(a))
    assert sum(1 for row in grid["regime"]["multi"] for v in row if v > 0) == 420 == n_below      # genau die 420 (α, β)-Zellen unterhalb der Formel haben für mindestens ein (y, c) drei


def test_h2_for_a_fixed_cell_the_formula_is_only_sufficient(grid):
    """Unterhalb der Formel hat im Raster knapp die Hälfte der (y, c)-Zellen drei, die andere Hälfte eines: 5 206 von 10 500 und 5 294 eindeutig."""
    s = X.grid_summary(grid)["regime"]["below"]
    assert s["cells"] - s["multi"] == 5294


# --- H3, H4 -------------------------------------------------------------------------------------------------------------------------------------
def test_h3_critical_beta_against_the_formula():
    expected = {0.5: (0.0372, 0.935), 1: (0.1416, 0.890), 2: (0.5268, 0.827), 4: (1.9224, 0.755), 8: (6.953, 0.683), 16: (25.27, 0.620), 30: (82.62, 0.577)}
    for a, (crit, ratio) in expected.items():
        got = L.critical_beta(float(a))
        assert round(got, 4 if a <= 4 else 3 if a == 8 else 2) == crit, a
        assert round(got / GRID.formula_bound(a), 3) == ratio, a


def test_h3_closed_form_equals_the_counted_boundary():
    for a in (1.0, 4.0, 8.0):
        crit = L.critical_beta(a)
        lo, hi = crit / 4, crit * 4
        for _ in range(60):
            mid = math.sqrt(lo * hi)
            if L.roots_linear(a, mid, 0.5).count == 1:
                hi = mid
            else:
                lo = mid
        assert hi == pytest.approx(crit, rel=1e-5), a


def _v3_count(alpha, beta, y=0.5, grid=400001, span=6.0):
    """Die Zählung der Vormessung V3: Vorzeichenwechsel auf einem x-Gitter y ± 6 (Nullen als +1). Nur Prüfling der Korrektur, nicht Teil der App."""
    from scipy.special import ndtr
    xs = np.linspace(y - span, y + span, grid)
    m = (alpha * y + beta * xs) / (alpha + beta)
    g = ndtr(math.sqrt(beta) * (xs - m) / math.sqrt(1 + beta / (alpha + beta))) - m
    sg = np.sign(g)
    sg[sg == 0] = 1
    return int(np.sum(sg[1:] != sg[:-1]))


def test_h4_v3_grid_artifact():
    pytest.importorskip("scipy")
    cells = [(a, b) for a in (1, 2, 4, 6, 10, 20) for b in (0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 400)]
    exact = {c: L.roots_linear(float(c[0]), float(c[1]), 0.5).count for c in cells}
    v3 = {c: _v3_count(float(c[0]), float(c[1])) for c in cells}
    assert len(cells) == 60 and sum(n == 3 for n in exact.values()) == 23 and sum(n == 3 for n in v3.values()) == 14
    assert sorted(c for c in cells if v3[c] != exact[c]) == [(2, 0.1), (4, 0.1), (6, 0.1), (6, 0.5), (10, 0.1), (10, 0.5), (20, 0.1), (20, 0.5), (20, 1)]
    assert all(v3[c] == 1 and exact[c] == 3 for c in cells if v3[c] != exact[c])
    claim = {c: c[1] >= c[0] ** 2 / (2 * math.pi) for c in cells}
    assert sum(claim[c] and exact[c] == 1 for c in cells) == 34 and sum(claim[c] and exact[c] != 1 for c in cells) == 0
    assert sorted(c for c in cells if not claim[c] and exact[c] == 1) == [(4, 2), (6, 5), (20, 50)] and sum((not claim[c]) and exact[c] != 1 for c in cells) == 23
    assert all((L.kappa(float(a), float(b)) <= SQ2PI) == (exact[(a, b)] == 1) for a, b in cells)          # κ² ≤ 2π trifft 60 von 60


# --- H5 -----------------------------------------------------------------------------------------------------------------------------------------
def test_h5_prior_fold_interval_and_rim():
    mismatches = total = 0
    for a in (1.0, 4.0, 10.0):
        for b in (0.1, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 100, 400):
            for y in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                total += 1
                mismatches += L.roots_linear(a, float(b), y).count != L.count_closed(a, float(b), y)
    assert (total, mismatches) == (297, 0)
    multi = lambda b: [y for y in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) if L.roots_linear(4.0, b, y).count == 3]    # noqa: E731
    assert multi(0.1) == [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] and multi(0.25) == [0.3, 0.4, 0.5, 0.6, 0.7] and multi(0.5) == [0.4, 0.5, 0.6] and multi(1.0) == [0.5]
    rim = L.roots_linear(4.0, 1.0, 0.3)
    assert rim.count == 1 and r4(rim.m[0]) == 0.9944 and round(rim.x[0], 2) == 3.77
    assert round(L.roots_linear(1.0, 100.0, 0.3).x[0], 4) == 0.5079


def test_h5_edge_roots_of_the_v3_cell():
    e = L.roots_linear(4.0, 0.1, 0.5)
    assert f"{e.m[0]:.2e}" == "2.07e-10" and round(e.x[0], 3) == -20.0 and round(e.x[2], 3) == 21.0


# --- H6, H7 -------------------------------------------------------------------------------------------------------------------------------------
def test_h6_c_band_edges():
    for (a, b, y), (lo, hi) in {(8, 4, 0.5): (0.4163, 0.5837), (8, 4, 0.3): (0.5989, 0.7496), (8, 4, 0.7): (0.2504, 0.4011), (16, 4, 0.5): (0.1896, 0.8104)}.items():
        got = R.c_band(float(a), float(b), y)
        assert (r4(got[0]), r4(got[1])) == (lo, hi), (a, b, y)
    assert R.c_band(4.0, 4.0, 0.5) is None


def test_h6_theta_hat_approaches_one_minus_c_only_for_large_beta():
    seq = lambda c: [r4(R.roots_regime(1.0, b, 0.5, c).theta[0]) for b in (1, 16, 256, 10000)]    # noqa: E731
    assert seq(0.25) == [0.9144, 0.7778, 0.7555, 0.7508] and seq(0.75) == [0.0856, 0.2222, 0.2445, 0.2492] and seq(0.5) == [0.5, 0.5, 0.5, 0.5]


def test_h7_lost_projects():
    def lost(a, c):
        return R.p_lost(a, 0.5, R.roots_regime(a, 1024.0, 0.5, c).theta[0])
    assert [r3(lost(1.0, c)) for c in (0.25, 0.5, 0.75)] == [0.092, 0.191, 0.291]
    assert [r3(lost(4.0, c)) for c in (0.25, 0.5, 0.75)] == [0.142, 0.341, 0.540]
    start = lambda a, c: R.p_start(a, 0.5, R.roots_regime(a, 1024.0, 0.5, c).theta[0])    # noqa: E731
    assert [r3(start(1.0, c)) for c in (0.25, 0.5, 0.75)] == [0.600, 0.500, 0.400] and [r3(start(4.0, c)) for c in (0.25, 0.5, 0.75)] == [0.699, 0.500, 0.301]
    assert r3(ncdf(0.5)) == 0.691 and r3(ncdf(2.0 * 0.5)) == 0.841


# --- H8 -----------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("c,counts,x_one", [(0.5, {2: 1, 3: 1, 5: 1, 10: 3, 25: 3, 100: 3}, {2: 1.989, 3: 1.978, 5: 1.961}), (0.6, {2: 1, 3: 1, 5: 3, 10: 3, 25: 1, 100: 1}, {25: -1.067, 100: -1.163})])
def test_h8_equilibrium_counts_by_n(c, counts, x_one):
    for n, count in counts.items():
        e = F.roots_finite(n, 8.0, 4.0, 0.5, c)
        assert e.count == count, (c, n)
        if n in x_one:
            assert r3(e.x[0]) == x_one[n], (c, n)


def test_h8_middle_thresholds_rise_toward_the_continuum_value():
    mids = {n: r3(F.roots_finite(n, 8.0, 4.0, 0.5, 0.5, scan=61 if n <= 100 else 41).x[1]) for n in (10, 25, 100, 250)}
    assert mids == {10: 0.151, 25: 0.390, 100: 0.474, 250: 0.490}


def test_h8_unique_cell_thresholds_fall_with_n():
    xs = {n: r3(F.roots_finite(n, 4.0, 4.0, 0.5, 0.5, scan=41 if n <= 100 else 21).x[0]) for n in (2, 3, 5, 10, 25, 100, 250)}
    assert xs == {2: 1.409, 3: 1.267, 5: 1.070, 10: 0.864, 25: 0.675, 100: 0.548, 250: 0.520}
    assert r3(F.roots_finite(1000, 4.0, 4.0, 0.5, 0.5, scan=21).x[0]) == 0.505


def test_h8_n_one_thousand_has_three_equilibria_with_the_stated_middle():
    e = F.roots_finite(1000, 8.0, 4.0, 0.5, 0.5, scan=41)
    assert e.count == 3 and r3(e.x[1]) == 0.497 and r3(e.x[0]) == -0.906 and r3(e.x[2]) == 1.910


def test_h8_few_players_cannot_have_more_than_the_hand_values():
    """N = 2, θ = 0,7, p = 0,3: 0,3; N = 3, θ = ½, p = ½: 0,75 (Hand-Mini-Instanzen)."""
    assert F.success_given_theta(0.7, 0.3, 2) == pytest.approx(0.3) and F.success_given_theta(0.5, 0.5, 3) == pytest.approx(0.75)


# --- H9 -----------------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("beta,count,rounds,spread", [(3, 3, 8, 3.58), (4, 3, 12, 2.82), (6, 3, 31, 1.91), (8, 3, 88, 1.23), (10, 3, 1362, 0.34), (10.1, 3, 2900, 0.23),
                                                      (10.2, 1, 35325, 0.0), (10.3, 1, 4820, 0.0), (11, 1, 772, 0.0), (12, 1, 385, 0.0), (16, 1, 163, 0.0), (100, 1, 90, 0.0)])
def test_h9_rounds_in_the_regime_model(beta, count, rounds, spread):
    e = R.roots_regime(8.0, float(beta), 0.5, 0.5)
    assert e.count == count and round(e.x[-1] - e.x[0], 2) == spread
    lo = I.iterate_regime(8.0, float(beta), 0.5, 0.5, I.START_LOW, e.x[0])
    hi = I.iterate_regime(8.0, float(beta), 0.5, 0.5, I.START_HIGH, e.x[-1])
    assert lo.reached and hi.reached and lo.monotone and hi.monotone
    assert band(lo.rounds, rounds, rel=0.02 if beta in (10.2, 10.3, 10.1) else 0.02) and lo.rounds == hi.rounds, (beta, lo.rounds, hi.rounds)


def test_h9_alpha_16_and_the_linear_model():
    for beta, count, rounds in ((40.0, 3, 2065), (41.0, 1, 12738)):
        e = R.roots_regime(16.0, beta, 0.5, 0.5)
        assert e.count == count and band(I.iterate_regime(16.0, beta, 0.5, 0.5, I.START_LOW, e.x[0]).rounds, rounds)
    for beta, count, rounds in ((1.5, 3, 58), (1.9, 3, 1272), (1.92, 3, 11045), (1.93, 1, 7725), (2.0, 1, 815)):
        e = L.roots_linear(4.0, beta, 0.5)
        assert e.count == count and band(I.iterate_linear(4.0, beta, 0.5, 0.0, e.m[0]).rounds, rounds), beta
    assert L.critical_beta(4.0) == pytest.approx(1.9224, abs=5e-5) and GRID.formula_bound(8.0) == pytest.approx(10.186, abs=5e-4)


# --- H10 ----------------------------------------------------------------------------------------------------------------------------------------
def test_h10_attack_threshold_in_36_cells():
    deviations = cells = 0
    for eps in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        qs = EM.q_star(eps)
        g = EM.build(eps, 0.5, 400)
        for q in (0.3, 0.9 * qs, 1.1 * qs, 0.5, 0.6, 0.9):
            cells += 1
            deviations += EM.greatest_equilibrium(g, q).attacks != (q <= qs)
    assert (cells, deviations) == (36, 0)


def test_h10_half_loss_never_attacks_though_many_messages_could_arrive():
    sm = X.e6_summary(0.1, 1.0)
    assert r4(sm["q_star"]) == 0.4737 and not sm["eq"].attacks and sm["g_states"] == 801 and sm["eq"].rounds > 700
    sm = X.e6_summary(0.1, 0.5)
    assert sm["eq"].attacks and (sm["eq"].first1, sm["eq"].first2) == (1, 1) and sm["common_states"] == 800


def test_h10_common_p_belief_boundary_and_one_round_per_message_level():
    for eps, below, above in ((0.05, 0.4867, 0.4877), (0.1, 0.4732, 0.4742), (0.3, 0.4114, 0.4122)):
        g = EM.build(eps, 0.5, 400)
        f_in, r_in = EM.common_p_belief(g, below)
        f_out, r_out = EM.common_p_belief(g, above)
        assert (int(f_in.sum()), r_in, int(f_out.sum()), r_out) == (800, 1, 0, 801), eps
        assert below < EM.q_star(eps) < above and r4(EM.q_star(eps)) in (0.4872, 0.4737, 0.4118)


def test_h10_simulated_protocol_against_the_formulas():
    sim = EM.simulate_protocol(0.1, 0.5, 400_000, 21)
    in_g = int(sim["is_g"].sum())
    for n, formula, simulated in ((5, 0.5905, 0.5905), (10, 0.3487, 0.3471), (20, 0.1216, 0.1199)):
        assert r4(EM.knowledge_level_prob(0.1, n)) == formula
        p_hat = float((sim["k"] >= n + 1).sum() / in_g)
        assert r4(p_hat) == simulated
        assert abs(p_hat - formula) < 4 * math.sqrt(formula * (1 - formula) / in_g), n
    k = sim["k"]
    t1, t2 = (k + 1) // 2, k // 2
    sel = t1 == 1
    p0 = float((t2[sel] == 0).mean())
    assert r4(1 / 1.9) == 0.5263 and abs(p0 - 1 / 1.9) < 4 * math.sqrt(0.25 / sel.sum())


def test_h10_profile_payoffs_equal_the_hand_values():
    """Auszahlung „ab Typ 1“ bei L = M = 1, p_G = ½: Spieler 1 0,4, Spieler 2 ½·0,9 = 0,45."""
    g = EM.build(0.1, 0.5, 400)
    u1 = np.where(g.t1 >= 1, np.where(g.is_g & (g.t2 >= 1), 1.0, -1.0), 0.0)
    u2 = np.where(g.t2 >= 1, np.where(g.is_g & (g.t1 >= 1), 1.0, -1.0), 0.0)
    assert round(float((g.prob * u1).sum()), 3) == 0.4 and round(float((g.prob * u2).sum()), 3) == 0.45


def test_h10_threshold_for_other_priors_is_the_maximum_of_two_terms():
    """Eigene Herleitung: (1−ε)/(2−ε) bei p_G = ½; bei p_G = 0,99 greift schon Typ 0 von Spieler 2 an (λ₀ = 0,99·0,1/(0,01 + 0,099) = 0,9083)."""
    assert r4(EM.first_message_lost(0.1, 0.99)) == 0.9083 and EM.attack_threshold(0.1, 0.99) == EM.first_message_lost(0.1, 0.99)
    assert r4(EM.first_message_lost(0.1, 0.5)) == 0.0909


# --- H11 ----------------------------------------------------------------------------------------------------------------------------------------
def test_h11_slopes_of_the_best_reply():
    e = R.roots_regime(8.0, 4.0, 0.5, 0.5)
    assert [r4(I.best_reply_slope_regime(w, 8.0, 4.0)) for w in e.w] == [0.36, 1.3314, 0.36] and e.stable == (True, False, True)
    lin = L.roots_linear(4.0, 1.0, 0.5)
    slopes = [lin.kappa * math.exp(-0.5 * (lin.kappa * (m - 0.5)) ** 2) / math.sqrt(2 * math.pi) for m in lin.m]
    assert [r4(s) for s in slopes] == [0.3783, 1.4567, 0.3783] and lin.stable == (True, False, True)


def test_h11_unique_roots_are_stable_in_the_whole_raster(grid):
    """In jeder Zelle des Rasters mit einer Wurzel ist sie stabil, in jeder mit drei ist die mittlere instabil und sind die äußeren stabil (Stichprobe: alle (α, β) bei y = c = ½)."""
    for a in GRID.ALPHAS[::4]:
        for b in GRID.BETAS[::3]:
            e = R.roots_regime(a, b, 0.5, 0.5)
            assert e.stable == ((True,) if e.count == 1 else (True, False, True)), (a, b)
            f = L.roots_linear(a, b, 0.5)
            assert f.stable == ((True,) if f.count == 1 else (True, False, True)), (a, b)


def test_h11_trajectories_leave_the_middle_root():
    t = X.e5_trajectories(8.0, 4.0, 0.5, 0.5, rounds=60)
    assert [x["label"] for x in t][2:] == ["knapp unter der mittleren Wurzel", "knapp über der mittleren Wurzel"]
    e = R.roots_regime(8.0, 4.0, 0.5, 0.5)
    assert abs(t[2]["path"][-1] - e.x[0]) < 1e-3 and abs(t[3]["path"][-1] - e.x[2]) < 1e-3 and abs(t[0]["path"][-1] - e.x[0]) < 1e-3


def test_method_numbers_of_the_email_chain():
    """„803 Zustände“ (J = 400: B, 801 Zustände G, Rest) und „bis zu 801 Nachrichten“."""
    g = EM.build(0.1, 0.5, 400)
    assert len(g.prob) == 803 and int(g.is_g[:-1].sum()) == 801 and g.prob.sum() == pytest.approx(1.0, abs=1e-15)
