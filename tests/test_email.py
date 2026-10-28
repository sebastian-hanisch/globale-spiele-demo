"""E-Mail-Spiel: Zustandsraum, Typen, bedingte Überzeugungen von Hand, größtes Gleichgewicht, Angriffsgrenze, gemeinsame p-Überzeugung, Protokollsimulation."""

import numpy as np
import pytest

import glo_email as E
from glo_rng import SplitMix64


def test_state_space_probabilities_and_types():
    g = E.build(0.1, 0.5, 3)
    assert len(g.prob) == 2 * 3 + 3 and g.prob.sum() == pytest.approx(1.0, abs=1e-15)
    assert g.prob[0] == 0.5 and g.prob[1] == pytest.approx(0.5 * 0.1) and g.prob[2] == pytest.approx(0.5 * 0.1 * 0.9)         # B; k = 1; k = 2
    assert g.t1.tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 5] and g.t2.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]                 # ⌈k/2⌉, ⌊k/2⌋ für k = 1..7; Rest: (J+2, J+1)
    assert g.is_g.tolist() == [False] + [True] * 8 and (g.n1, g.n2) == (6, 5)


def test_message_to_type_mapping_by_hand():
    assert [E.messages_to_types(k) for k in range(0, 6)] == [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)]


def test_hand_conditional_probabilities(mail):
    """P(k = 2 | t₁ = 1) = (1−ε)/(2−ε) = q*; P(t₂ = 0 | t₁ = 1) = 1/(2−ε); die Überzeugung von Typ 1 hängt am Angriff von Typ t₂ = 1."""
    g = E.build(mail["eps"], 0.5, 3)
    s2 = np.array([False, True, True, True, False])              # Spieler 2 greift ab Typ 1 an (n2 = 5: Typen 0..3 und der Rest)
    b1 = E.belief(g, 1, s2)
    assert b1[1] == pytest.approx(mail["q_star"], abs=1e-15) and 1 - b1[1] == pytest.approx(mail["p_t2_zero"], abs=1e-15)
    assert b1[0] == 0.0                                           # Typ 0 von Spieler 1: Zustand B
    assert E.q_star(mail["eps"]) == pytest.approx(mail["q_star"], abs=1e-15)
    assert E.knowledge_level_prob(mail["eps"], 1) == pytest.approx(mail["p_k2"], abs=1e-15)


def test_hand_payoff_of_the_profile_attack_from_type_one(mail):
    """Auszahlung von Spieler 1 im Profil „ab Typ 1“ bei L = M = 1, p_G = ½: ½·(0,9 − 0,1) = 0,4, aus den Zustandswahrscheinlichkeiten gerechnet."""
    g = E.build(mail["eps"], 0.5, 400)                          # J = 400: der abgeschnittene Rest hat Wahrscheinlichkeit 0,9^801 ≈ 10^-37
    s = np.ones(g.n1, bool)
    s[0] = False
    s[-1] = False
    s2 = np.ones(g.n2, bool)
    s2[0] = False
    s2[-1] = False
    attacks1 = s[g.t1]
    other = s2[g.t2]
    u = np.where(attacks1, np.where(g.is_g & other, 1.0, -1.0), 0.0)
    assert float((g.prob * u)[:-1].sum() + 0 * g.prob[-1]) == pytest.approx(mail["payoff1"], abs=1e-12)


def test_best_response_is_threshold_in_the_belief_and_rest_type_never_attacks():
    g = E.build(0.1, 0.5, 3)
    s2 = np.ones(g.n2, bool)
    s2[0] = s2[-1] = False                                         # Spieler 2 greift ab Typ 1 an: Überzeugung von Typ 1 ist q* = 0,4737
    for q, expect in ((0.4, True), (0.5, False)):
        assert E.best_response(g, 1, s2, q)[1] == expect
    assert not E.best_response(g, 1, np.ones(g.n2, bool), 0.0)[-1]


@pytest.mark.parametrize("q,attack", [(0.30, True), (0.45, True), (0.47, True), (0.48, False), (0.50, False)])
def test_greatest_equilibrium_at_eps_one_tenth(q, attack):
    """ε = 0,1, p_G = ½: Grenze (1−ε)/(2−ε) = 0,4737. Bei q = ½ (L = M) greift nie jemand an, obwohl bis zu 801 Nachrichten ankommen könnten."""
    g = E.build(0.1, 0.5, 400)
    eq = E.greatest_equilibrium(g, q)
    assert eq.attacks == attack and E.is_equilibrium(g, eq.s1, eq.s2, q)
    if attack:
        assert eq.first1 == 1 and eq.first2 == 1 and eq.rounds <= 3
    else:
        assert eq.first1 is None and eq.rounds > 700           # die Kette löst sich Stufe für Stufe auf (eine Runde je Nachrichtenstufe)


def test_unraveling_takes_one_round_per_message_level():
    """Ohne Angriff (q = ½) braucht die Iteration eine Runde je Typ plus ein, in J = 20: Bänder um 2J + 2 = 42 (Zählung der Typen beider Spieler)."""
    g = E.build(0.1, 0.5, 20)
    assert 38 <= E.greatest_equilibrium(g, 0.5).rounds <= 46


def test_attack_threshold_formula_against_the_iteration_in_a_grid():
    """Angriff im größten Gleichgewicht genau dann, wenn q ≤ max(λ₀, (1−ε)/(2−ε)); Zellen innerhalb 0,1 % der Grenze entfallen (Gleichheit)."""
    tested = 0
    for eps in (0.05, 0.1, 0.2, 0.3, 0.5):
        for p_g in (0.2, 0.5, 0.8, 0.95):
            thr = E.attack_threshold(eps, p_g)
            g = E.build(eps, p_g, 40)
            for q in (0.05, 0.3, 0.5, 0.7, 0.9, thr * 0.99, thr * 1.01):
                if not 0 < q < 1 or abs(q - thr) < 1e-3 * thr and q not in (thr * 0.99, thr * 1.01):
                    continue
                assert E.greatest_equilibrium(g, q).attacks == (q <= thr), (eps, p_g, q)
                tested += 1
    assert tested > 100


def test_threshold_equals_q_star_at_half_prior_up_to_eps_one_half():
    for eps in (0.02, 0.1, 0.3, 0.5):
        assert E.attack_threshold(eps, 0.5) == pytest.approx(E.q_star(eps), abs=1e-15)
    assert E.first_message_lost(0.1, 0.5) == pytest.approx(0.1 / 1.1, abs=1e-15)
    assert E.attack_threshold(0.1, 0.99) > E.q_star(0.1)          # bei fast sicherem G greift schon Typ 0 von Spieler 2 an


def test_common_p_belief_boundary_and_rounds():
    """Gemeinsame p-Überzeugung von G: bis (1−ε)/(2−ε) in allen Zuständen G außer k = 1 (Spieler 2 ohne Nachricht hält G nur mit λ₀ für wahr), knapp darüber in keinem,
    dann braucht der Operator eine Runde je Nachrichtenstufe (hier J = 50: 101 Zustände)."""
    eps = 0.1
    qs = E.q_star(eps)
    g = E.build(eps, 0.5, 50)
    f_in, r_in = E.common_p_belief(g, qs * 0.999)
    f_out, r_out = E.common_p_belief(g, qs * 1.001)
    assert int(f_in.sum()) == 100 and r_in == 1 and not f_in[1] and f_in[2:-1].all()
    assert int(f_out.sum()) == 0 and r_out == 101


def test_common_p_belief_is_everything_for_small_p_and_for_p_below_lambda0_even_k_equals_one():
    g = E.build(0.1, 0.5, 20)
    f, rounds = E.common_p_belief(g, 0.05)                       # 0,05 < λ₀ = 0,0909: auch k = 1 bleibt
    assert int(f.sum()) == 2 * 20 + 1 and rounds == 0
    assert not f[0] and not f[-1]


def test_knowledge_level_probability():
    assert E.knowledge_level_prob(0.1, 5) == pytest.approx(0.9 ** 5, abs=1e-15) and E.knowledge_level_prob(0.1, 0) == 1.0


def test_protocol_simulation_matches_the_formulas_within_standard_errors():
    """100 000 Spiele, Zufalls-Seed 11, p_G = ½, ε = 0,1: P(k ≥ n+1 | G) = (1−ε)^n und der Anteil G; Marge 4 Standardfehler."""
    games = 100_000
    sim = E.simulate_protocol(0.1, 0.5, games, 11)
    g_share = sim["is_g"].mean()
    assert abs(g_share - 0.5) < 4 * np.sqrt(0.25 / games)
    in_g = sim["is_g"].sum()
    for n in (1, 5, 10):
        p_hat = float((sim["k"] >= n + 1).sum() / in_g)
        p = 0.9 ** n
        assert abs(p_hat - p) < 4 * np.sqrt(p * (1 - p) / in_g), n
    assert (sim["k"][~sim["is_g"]] == 0).all() and (sim["k"][sim["is_g"]] >= 1).all()


def test_protocol_simulation_is_deterministic_in_the_seed():
    a = E.simulate_protocol(0.2, 0.5, 2000, 5)
    b = E.simulate_protocol(0.2, 0.5, 2000, 5)
    c = E.simulate_protocol(0.2, 0.5, 2000, 6)
    assert np.array_equal(a["k"], b["k"]) and not np.array_equal(a["k"], c["k"])


def test_single_game_draw_is_consistent_with_its_message_list():
    rng = SplitMix64(35)
    is_g, k, arrived = E.draw_game(0.3, 0.5, rng)
    if is_g:
        assert k == len(arrived) and arrived[-1] is False and all(arrived[:-1])
    else:
        assert k == 0 and arrived == []
    seen = {E.draw_game(0.3, 0.5, SplitMix64(s))[0] for s in range(20)}
    assert seen == {True, False}
