"""Orakel-Tests (ORACLE-PLAYBOOK §2, Spieltheorie): jede Kernroutine gegen einen ANDEREN Rechenweg.

Orakel:
  O1  Φ gegen scipy.special.ndtr;
  O2  Modell L: Wurzeln gegen scipy.optimize.brentq auf der reduzierten Gleichung in m (eigenes Gitter, beide Ränder) UND gegen die x-Form (V3-Formel, Identität);
  O3  Modell R: die reduzierte Gleichung in w gegen eine unabhängige x-Rechnung (θ̂(x) per Bisektion mit scipy.special.ndtr, dann Indifferenz), Zahl der Wurzeln in 100 Zufallszellen;
  O4  Zahl der Gleichgewichte gegen das geschlossene Band (R) bzw. die Faltenkurve (L);
  O5  endliches N: Startwahrscheinlichkeit gegen scipy.integrate.quad mit scipy.stats.binom und gegen Rejection-Sampling (4 Standardfehler);
  O6  beste Antwort und Stabilität: Fixpunkt θ̂ gegen brentq, mittlere Wurzel instabil gegen die zentrale Differenz;
  O7  E-Mail-Spiel: größtes Gleichgewicht gegen Vollaufzählung aller reinen Profile (J = 3) mit exakter Bruchrechnung, Grenze, gemeinsame p-Überzeugung gegen eine Mengenschleife,
      Protokollsimulation gegen (1−ε)^n mit numpy-Zufall (ein anderer Strom als SplitMix64).
Kein scipy im Kern (requirements-dev.txt)."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

import glo_email as E
import glo_finite as F
import glo_iteration as I
import glo_linear as L
import glo_regime as R
from glo_core import ncdf

sp = pytest.importorskip("scipy")
from scipy.integrate import quad  # noqa: E402
from scipy.optimize import brentq  # noqa: E402
from scipy.special import ndtr  # noqa: E402
from scipy.stats import binom  # noqa: E402


# --------------------------------------------------------------------------------- O1
def test_o1_ncdf_matches_scipy_ndtr():
    z = np.linspace(-30.0, 30.0, 6001)
    ours, ref = ncdf(z), ndtr(z)
    assert np.max(np.abs(ours - ref)) < 1e-15
    assert np.max(np.abs(ours - ref) / np.maximum(ref, 1e-300)) < 1e-12


# --------------------------------------------------------------------------------- O2
def _scipy_roots_linear(a, b, y):
    """Wurzeln von Φ(κ(m−y)) = m mit brentq auf einem eigenen Gitter aus log- und linearem Teil beider Ränder (unabhängig von den Stücken in glo_linear)."""
    kap = L.kappa(a, b)
    fl = lambda t: ndtr(kap * (t - y)) - t        # noqa: E731
    fu = lambda u: u - ndtr(kap * (y - 1 + u))    # noqa: E731
    grid = np.unique(np.concatenate([np.logspace(-300, math.log10(0.5), 3000), np.linspace(0, 0.5, 5001)[1:]]))
    roots = []
    for fn, flip in ((fl, False), (fu, True)):
        v = fn(grid)
        for i in range(len(grid) - 1):
            if v[i] * v[i + 1] < 0:
                r = brentq(fn, grid[i], grid[i + 1], xtol=1e-320, rtol=1e-14)
                roots.append(1 - r if flip else r)
        if v[-1] == 0.0 and not flip:
            roots.append(float(grid[-1]))
    return sorted(set(round(r, 15) for r in roots))


@pytest.mark.parametrize("a,b,y", [(4, 1, 0.5), (4, 1, 0.3), (10, 10, 0.5), (10, 2, 0.4), (6, 2, 0.6), (4, 0.5, 0.5), (20, 5, 0.5), (4, 0.1, 0.5), (1, 100, 0.3), (2, 0.4, 0.5)])
def test_o2_linear_roots_match_brentq_on_the_reduced_equation(a, b, y):
    mine = L.roots_linear(a, b, y)
    ref = _scipy_roots_linear(a, b, y)
    # 0,5 kann im Gitter doppelt vorkommen (untere und obere Hälfte): auf Zahl und Lage prüfen
    ref = [r for i, r in enumerate(ref) if i == 0 or abs(r - ref[i - 1]) > 1e-12]
    assert len(ref) == mine.count, (a, b, y, ref, mine.m)
    for m, r in zip(mine.m, ref):
        tail = min(m, 1 - m)
        assert abs(min(m, 1 - m) - min(r, 1 - r)) <= 1e-9 * max(tail, 1e-300) + 1e-15, (m, r)


def test_o2_x_form_of_v3_equals_the_reduced_form():
    """Identität der beiden Schreibweisen: g_x(x) = Φ(√β(x − m)/√(1 + β/(α+β))) − m gleich g_m(m(x)) = Φ(κ(m − y)) − m, 400 Zufallspunkte."""
    rng = np.random.default_rng(20261028)
    worst = 0.0
    for _ in range(400):
        a = float(np.exp(rng.uniform(math.log(0.2), math.log(30))))
        b = float(np.exp(rng.uniform(math.log(0.05), math.log(500))))
        y = float(rng.uniform(0.1, 0.9))
        x = float(rng.uniform(y - 2, y + 2))
        m = (a * y + b * x) / (a + b)
        gx = ndtr(math.sqrt(b) * (x - m) / math.sqrt(1 + b / (a + b))) - m
        gm = ndtr(L.kappa(a, b) * (m - y)) - m
        worst = max(worst, abs(gx - gm))
    assert worst < 1e-12


# --------------------------------------------------------------------------------- O3, O4
def _theta_hat_vec(x, beta):
    lo, hi = np.zeros_like(x), np.ones_like(x)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        up = ndtr(math.sqrt(beta) * (x - mid)) - mid > 0
        lo, hi = np.where(up, mid, lo), np.where(up, hi, mid)
    return 0.5 * (lo + hi)


def _x_route_count(a, b, y, c, lo=-3.0, hi=4.0, n=8001):
    x = np.linspace(lo, hi, n)
    th = _theta_hat_vec(x, b)
    m = (a * y + b * x) / (a + b)
    p = ndtr(math.sqrt(a + b) * (th - m)) - c
    s = np.sign(p)
    s[s == 0] = 1
    return int(np.sum(s[1:] != s[:-1])), x[np.where(s[1:] != s[:-1])[0]]


def test_o3_regime_roots_match_the_independent_x_route_in_random_cells():
    rng = np.random.default_rng(20261028)
    same = checked = 0
    for _ in range(100):
        a = float(np.exp(rng.uniform(math.log(0.5), math.log(14))))
        b = float(np.exp(rng.uniform(math.log(0.5), math.log(40))))
        y, c = float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9))
        band = R.d_band(a, b)
        d = R.d_value(a, b, y, c)
        if band is not None and min(abs(d - band[0]), abs(d - band[1])) < 0.02:
            continue                                           # Berührpunkt: das Gitter der x-Route kann das Paar verfehlen
        mine = [x for x in R.roots_regime(a, b, y, c).x if -3.0 <= x <= 4.0]
        n_ref, x_ref = _x_route_count(a, b, y, c)
        checked += 1
        if len(mine) == n_ref:
            same += 1
            if mine:                                           # beide Seiten können im Fenster [−3, 4] leer sein (Wurzeln außerhalb)
                assert np.max(np.abs(np.array(mine) - x_ref)) < 3e-3
    assert checked >= 80 and same == checked


def test_o4_counts_match_the_closed_forms_over_a_c_grid_and_the_c_band():
    for a, b, y in ((8, 4, 0.5), (8, 4, 0.3), (8, 4, 0.7), (16, 4, 0.5)):
        lo, hi = R.c_band(a, b, y)
        cs = np.linspace(0.005, 0.995, 199)
        counts = [R.roots_regime(a, b, y, float(c)).count for c in cs]
        inside = [3 if lo + 1e-4 < c < hi - 1e-4 else 1 for c in cs if abs(c - lo) > 1e-4 and abs(c - hi) > 1e-4]
        got = [n for n, c in zip(counts, cs) if abs(c - lo) > 1e-4 and abs(c - hi) > 1e-4]
        assert got == inside, (a, b, y)


# --------------------------------------------------------------------------------- O5
def _scipy_success(s, x, n, a, b, y):
    m = (a * y + b * s) / (a + b)
    sd = 1 / math.sqrt(a + b)
    total = float(ndtr((1 / n - m) / sd))
    for j in range(1, n):
        integrand = lambda t, j=j: binom.sf(j - 1, n - 1, float(ndtr(math.sqrt(b) * (x - t)))) * math.exp(-0.5 * ((t - m) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))  # noqa: E731
        total += quad(integrand, j / n, (j + 1) / n, epsabs=1e-13, epsrel=1e-13)[0]
    return total


@pytest.mark.parametrize("n", [2, 3, 5, 25])
@pytest.mark.parametrize("x", [0.2, 0.5, 0.75])
def test_o5_success_prob_matches_scipy_quad_and_binom(n, x):
    assert F.success_prob(x, x, n, 8.0, 4.0, 0.5) == pytest.approx(_scipy_success(x, x, n, 8.0, 4.0, 0.5), abs=1e-9)


@pytest.mark.parametrize("n,x", [(250, 0.5), (1000, 0.5), (1000, 1.2)])
def test_o5_large_n_with_few_nodes_stays_close_to_scipy(n, x):
    assert F.success_prob(x, x, n, 8.0, 4.0, 0.5) == pytest.approx(_scipy_success(x, x, n, 8.0, 4.0, 0.5), abs=2e-6)


def test_o5_own_signal_and_threshold_enter_separately():
    """P_N(s; x) mit s ≠ x gegen scipy: der Posterior hängt am eigenen Signal, die Binomialzahl an der Schwelle der anderen."""
    assert F.success_prob(0.3, 0.8, 5, 8.0, 4.0, 0.5) == pytest.approx(_scipy_success(0.3, 0.8, 5, 8.0, 4.0, 0.5), abs=1e-9)
    assert F.success_prob(0.3, 0.8, 5, 8.0, 4.0, 0.5) != pytest.approx(F.success_prob(0.8, 0.8, 5, 8.0, 4.0, 0.5), abs=1e-3)


def _simulated_success(x, n, a, b, y, seed, draws=8_000_000, delta=0.01):
    rng = np.random.default_rng(seed)
    hit = tot = 0
    for _ in range(draws // 2_000_000):
        th = y + rng.standard_normal(2_000_000) / math.sqrt(a)
        x1 = th + rng.standard_normal(2_000_000) / math.sqrt(b)
        tha = th[np.abs(x1 - x) < delta]
        e = rng.standard_normal((len(tha), n - 1)) / math.sqrt(b)
        k = ((tha[:, None] + e) <= x).sum(axis=1)
        hit += int(np.sum((1 + k) / n >= tha))
        tot += len(tha)
    return hit / tot, tot


@pytest.mark.parametrize("n,x", [(3, 0.5), (5, 0.2)])
def test_o5_success_prob_matches_a_rejection_sampling_of_the_game(n, x):
    p_hat, tot = _simulated_success(x, n, 8.0, 4.0, 0.5, seed=11)
    p = F.success_prob(x, x, n, 8.0, 4.0, 0.5)
    assert abs(p_hat - p) < 4 * math.sqrt(p * (1 - p) / tot), (p_hat, p, tot)


# --------------------------------------------------------------------------------- O6
def test_o6_theta_hat_matches_brentq_and_the_best_reply_formula():
    rng = np.random.default_rng(7)
    for _ in range(40):
        x = float(rng.uniform(-2, 3))
        b = float(np.exp(rng.uniform(math.log(0.1), math.log(100))))
        ref = brentq(lambda t: ndtr(math.sqrt(b) * (x - t)) - t, 0.0, 1.0, xtol=1e-15, rtol=1e-14)
        assert I.theta_hat(x, b) == pytest.approx(ref, abs=1e-12)


def test_o6_unique_equilibria_attract_from_both_sides_in_many_cells():
    rng = np.random.default_rng(3)
    done = 0
    while done < 12:
        a, b = float(rng.uniform(0.5, 6)), float(rng.uniform(4, 40))
        y, c = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 0.8))
        e = R.roots_regime(a, b, y, c)
        if e.count != 1:
            continue
        done += 1
        for x0 in (I.START_LOW, I.START_HIGH):
            res = I.iterate_regime(a, b, y, c, x0, e.x[0])
            assert res.reached and res.monotone


# --------------------------------------------------------------------------------- O7
def _payoff_exact(game, own, own_type, s_other, q):
    """Bedingter Erwartungsnutzen des Angriffs (M = 1 − q, L = q) mit exakten Brüchen: Summe über alle Zustände, ohne bincount."""
    num = den = Fraction(0)
    for st in range(len(game.prob)):
        t_own = game.t1[st] if own == 1 else game.t2[st]
        if t_own != own_type:
            continue
        t_oth = game.t2[st] if own == 1 else game.t1[st]
        pr = Fraction(game.prob[st]).limit_denominator(10 ** 12)
        den += pr
        num += pr * ((1 - q) if (game.is_g[st] and s_other[t_oth]) else -q)
    return num / den


def _enumerate_equilibria(eps, p_g, j, q):
    game = E.build(eps, p_g, j)
    qf = Fraction(q).limit_denominator(1000)
    n1, n2 = game.n1 - 1, game.n2 - 1             # echte Typen; der Rest-Typ greift nie an
    out = []
    for bits in itertools.product([False, True], repeat=n1 + n2):
        s1 = list(bits[:n1]) + [False]
        s2 = list(bits[n1:]) + [False]
        ok = all((_payoff_exact(game, 1, t, s2, qf) >= 0) == s1[t] or _payoff_exact(game, 1, t, s2, qf) == 0 for t in range(n1))
        ok = ok and all((_payoff_exact(game, 2, t, s1, qf) >= 0) == s2[t] or _payoff_exact(game, 2, t, s1, qf) == 0 for t in range(n2))
        if ok:
            out.append((tuple(s1), tuple(s2)))
    return game, out


@pytest.mark.parametrize("q,count", [(0.3, 2), (0.45, 2), (0.5, 1)])
def test_o7_full_enumeration_j3_matches_the_greatest_equilibrium(q, count):
    """J = 3, ε = 0,1, p_G = ½ (512 reine Profile): q = 0,3 und 0,45 je zwei Gleichgewichte (keiner / Angriff ab Typ 1), q = 0,5 genau eins (keiner)."""
    game, eqs = _enumerate_equilibria(0.1, 0.5, 3, q)
    assert len(eqs) == count
    big = E.greatest_equilibrium(game, q)
    mine = (tuple(bool(v) for v in big.s1), tuple(bool(v) for v in big.s2))
    assert mine in eqs
    # das größte: kein anderes Gleichgewicht greift in einem Typ an, in dem das größte es nicht tut
    for s1, s2 in eqs:
        assert all(m or not o for m, o in zip(mine[0], s1)) and all(m or not o for m, o in zip(mine[1], s2))
    if count == 2:
        assert mine[0] == (False, True, True, True, True, False) and mine[1] == (False, True, True, True, False)
    else:
        assert not any(mine[0]) and not any(mine[1])


def test_o7_exact_payoff_helper_agrees_with_the_float_belief():
    game = E.build(0.1, 0.5, 3)
    s2 = np.array([False, True, True, True, False])
    for t in range(5):
        want = _payoff_exact(game, 1, t, s2, Fraction(1, 2))
        b = E.belief(game, 1, s2)[t]
        assert float(want) == pytest.approx((1 - 0.5) * b - 0.5 * (1 - b), abs=1e-12)


def _common_belief_loop(game, p):
    """Gemeinsame p-Überzeugung mit Mengenschleifen statt bincount."""
    states = [i for i in range(len(game.prob) - 1) if game.is_g[i]]
    f = set(states)
    changed = True
    while changed and f:
        changed = False
        keep = set()
        for st in f:
            ok = True
            for own_t_arr in (game.t1, game.t2):
                same = [i for i in range(len(game.prob)) if own_t_arr[i] == own_t_arr[st]]
                num = sum(game.prob[i] for i in same if i in f)
                den = sum(game.prob[i] for i in same)
                if num / den < p - 1e-12:
                    ok = False
            if ok:
                keep.add(st)
        if keep != f:
            f, changed = keep, True
    return f


@pytest.mark.parametrize("eps", [0.05, 0.1, 0.3])
def test_o7_common_p_belief_matches_a_set_loop_and_the_boundary(eps):
    game = E.build(eps, 0.5, 12)
    qs = E.q_star(eps)
    for p in (0.05, qs * 0.99, qs * 1.01, 0.9):
        mine, _ = E.common_p_belief(game, p)
        assert set(np.where(mine)[0].tolist()) == _common_belief_loop(game, p), (eps, p)
    inside, _ = E.common_p_belief(game, qs * 0.99)
    outside, _ = E.common_p_belief(game, qs * 1.01)
    assert int(inside.sum()) == 2 * 12 and int(outside.sum()) == 0


def test_o7_protocol_simulation_with_numpy_random_matches_the_knowledge_levels():
    """Ein anderer Zufallsstrom als SplitMix64: 4 Seeds × 500 000 Spiele, Nachricht für Nachricht, gegen (1−ε)^n; 4 Standardfehler."""
    eps, tot, hits = 0.1, 0, {5: 0, 10: 0, 20: 0}
    for seed in (21, 22, 23, 24):
        rng = np.random.default_rng(seed)
        n = 500_000
        alive = np.ones(n, bool)                                   # nur die Spiele im Zustand G
        k = np.zeros(n, int)
        while alive.any():
            k += alive
            alive &= rng.random(n) >= eps
        delivered = k - 1
        tot += n
        for m in hits:
            hits[m] += int(np.sum(delivered >= m))
    for m, h in hits.items():
        p = 0.9 ** m
        assert abs(h / tot - p) < 4 * math.sqrt(p * (1 - p) / tot), m


# --------------------------------------------------------------------------------- Modell L: N ändert nichts
@pytest.mark.parametrize("x", [0.0, 0.5, 1.0])
def test_o8_linear_expected_payoff_at_n_three_equals_the_continuum_value(x):
    """Modell L, α = 4, β = 1, y = ½, N = 3: der Erwartungsnutzen des Mitmachens bei Signal x (die zwei anderen spielen Schwelle x) ist Φ(κ(m − y)) − m, simuliert mit Rejection-Sampling (4 Standardfehler)."""
    a, b, y, n = 4.0, 1.0, 0.5, 3
    rng = np.random.default_rng(31)
    vals = []
    for _ in range(4):
        th = y + rng.standard_normal(2_000_000) / math.sqrt(a)
        x1 = th + rng.standard_normal(2_000_000) / math.sqrt(b)
        tha = th[np.abs(x1 - x) < 0.02]
        e = rng.standard_normal((len(tha), n - 1)) / math.sqrt(b)
        k = ((tha[:, None] + e) <= x).sum(axis=1)
        vals.append(k / (n - 1) - tha)
    v = np.concatenate(vals)
    m = (a * y + b * x) / (a + b)
    want = float(ndtr(L.kappa(a, b) * (m - y))) - m
    assert abs(v.mean() - want) < 4 * v.std() / math.sqrt(len(v)), (v.mean(), want)
