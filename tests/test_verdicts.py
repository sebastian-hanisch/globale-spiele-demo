"""Verdikte als reine Funktionen (DEMO-PLAYBOOK §12, Punkt 2): Art, jede Zahl im Text und jeder Zweig an handgesetzten Werten; Formate."""

import math


import glo_constants as C
import glo_evaluation as V
import glo_experiments as X
import glo_iteration as I
import glo_scenario as S
from glo_evaluation import EquilibriumRow, MainView


def mv(rows, *, settings=None, cont=None, formula_ok=True, bound=2.546, crit=None, band=None, p_unit=0.6827, underflow=False, scan=None):
    return MainView(settings or S.Settings(), tuple(rows), len(rows) if cont is None else cont, formula_ok, bound, crit, band, p_unit, underflow, scan)


ONE = [EquilibriumRow(0.5, 0.5, True, "einziges Gleichgewicht")]
THREE = [EquilibriumRow(-0.91, 0.03, True, "kaum einer macht mit"), EquilibriumRow(0.5, 0.5, False, "mittleres (instabil)"), EquilibriumRow(1.91, 0.97, True, "fast alle machen mit")]


# --- Formate ---------------------------------------------------------------------------------------------------------------------------------
def test_de_thousands_pct():
    assert V.de(1.5, 1) == "1,5" and V.de(3.14159, 3) == "3,142" and V.de(7, 0) == "7"
    assert V.de(-0.91, 3) == "−0,910" and V.de(-0.0001, 3) == "0,000"                 # echtes Minuszeichen; kein „−0,000“
    assert V.thousands(1234567) == "1.234.567" and V.thousands(999.6) == "1.000" and V.thousands(0) == "0"
    assert V.pct(0.4958) == "49,6 %" and V.pct(0.0, 0) == "0 %" and V.pct(1.0) == "100,0 %"


def test_fmt_x_and_fmt_small():
    assert V.fmt_x(-1.1953) == "−1,195" and V.fmt_x(0.5) == "0,500" and V.fmt_x(2500.0) == "2.500"
    assert V.fmt_small(0.0) == "0" and V.fmt_small(0.05028) == "0,0503" and V.fmt_small(2.07e-10) == "2,07·10^-10"


def test_players_text():
    assert V.players_text(0) == "Kontinuum" and V.players_text(5) == "N = 5"


# --- Zahl der Gleichgewichte -------------------------------------------------------------------------------------------------------------------
def test_verdict_count_one_regime():
    kind, text = V.verdict_count(mv(ONE))
    assert kind == "success" and text.startswith("**Ein einziges Gleichgewicht** (Schwelle x* = 0,500, kritischer Zustand θ̂ = 0,5000).")
    assert "Das Projekt startet für alle Zustände θ ≤ 0,500." in text and "führt von „kaum einer“ und von „fast alle“ dorthin" in text
    finite_one = V.verdict_count(mv([EquilibriumRow(1.4, None, True, "einziges Gleichgewicht")], settings=S.Settings(players=2), cont=1))[1]
    assert "Iteration" not in finite_one                   # die Iteration ist nur im Kontinuum gerechnet


def test_verdict_count_one_linear_has_no_start_sentence():
    kind, text = V.verdict_count(mv([EquilibriumRow(3.77, 0.9944, True, "einziges Gleichgewicht")], settings=S.Settings(model=C.MODEL_L)))
    assert kind == "success" and "Anteil m bei Indifferenz = 0,9944" in text and "Das Projekt startet" not in text and "x* = 3,770" in text


def test_verdict_count_three_names_all_thresholds_and_the_unstable_middle():
    kind, text = V.verdict_count(mv(THREE))
    assert kind == "warning" and "**Drei Gleichgewichte**" in text
    assert "x* = −0,910 (kaum einer macht mit), 0,500 (mittleres) und 1,910 (fast alle machen mit)" in text and "Das mittlere ist instabil" in text and "10^-300" not in text


def test_verdict_count_three_with_underflow_mentions_the_flag():
    assert "Wurzeln unter 10^-300 sind als 0 bzw. 1 gemeldet" in V.verdict_count(mv(THREE, underflow=True))[1]


def test_verdict_count_finite_names_players_and_continuum_count():
    rows = [EquilibriumRow(x, None, st, lab) for x, st, lab in ((-0.51, True, "a"), (0.78, False, "b"), (1.59, True, "c"))]
    kind, text = V.verdict_count(mv(rows, settings=S.Settings(players=5), cont=1))
    assert kind == "warning" and text.startswith("Bei N = 5 (Kontinuum: 1 Gleichgewicht): **Drei Gleichgewichte**")
    kind2, text2 = V.verdict_count(mv(rows[:1], settings=S.Settings(players=10), cont=3))
    assert text2.startswith("Bei N = 10 (Kontinuum: 3 Gleichgewichte): ") and "θ̂" not in text2          # kein kritischer Zustand bei endlichem N


def test_verdict_count_other_counts_are_info_with_the_raster_size():
    rows = [EquilibriumRow(0.1, None, True, "s"), EquilibriumRow(0.7, None, False, "i")]
    kind, text = V.verdict_count(mv(rows, settings=S.Settings(players=3), scan=241))
    assert kind == "info" and "**2 Gleichgewichte** gezählt (davon 1 stabil)" in text and "Raster mit 241 Punkten" in text


# --- Faustformel -------------------------------------------------------------------------------------------------------------------------------
def test_verdict_formula_regime_holds_is_the_uniqueness_bound():
    kind, text = V.verdict_formula(mv(ONE, settings=S.Settings(beta=16.0)))
    assert kind == "success" and text.startswith("β = 16,00 ≥ α²/(2π) = 2,546:") and "genau die Bedingung" in text


def test_verdict_formula_regime_below_names_the_band_and_whether_c_is_inside():
    s = S.Settings(alpha=8.0, beta=4.0, c=0.5)
    kind, text = V.verdict_formula(mv(THREE, settings=s, formula_ok=False, bound=10.186, band=(0.4163, 0.5837)))
    assert kind == "warning" and text.startswith("β = 4,00 < α²/(2π) = 10,186:") and "zwischen 0,4163 und 0,5837" in text and "c = 0,50 liegt darin" in text and "nur hinreichend" in text
    kind2, text2 = V.verdict_formula(mv(ONE, settings=S.Settings(alpha=8.0, beta=4.0, c=0.6), formula_ok=False, bound=10.186, band=(0.4163, 0.5837)))
    assert "c = 0,60 liegt außerhalb" in text2 and "Kontinuum: 1 Gleichgewicht)" in text2


def test_verdict_formula_regime_without_band():
    kind, text = V.verdict_formula(mv(ONE, formula_ok=False, band=None))
    assert kind == "info" and "Kein Band" in text


def test_verdict_formula_linear_sufficient_not_necessary():
    s = S.Settings(model=C.MODEL_L, alpha=4.0, beta=1.0)
    kind, text = V.verdict_formula(mv(THREE, settings=s, formula_ok=False, bound=2.546, crit=1.9224))
    assert kind == "warning" and "die Formel sagt nichts" in text and "β = 1,922, das ist das 0,76-Fache der Formel" in text and "hier gezählt: 3 Gleichgewichte" in text
    kind2, text2 = V.verdict_formula(mv(ONE, settings=s, formula_ok=True, bound=2.546, crit=1.9224))
    assert kind2 == "success" and "hinreichend" in text2 and "nicht notwendig" in text2
    kind3, text3 = V.verdict_formula(mv(ONE, settings=s, formula_ok=False, bound=2.546, crit=1.9224))
    assert kind3 == "info" and "hier gezählt: 1 Gleichgewicht." in text3


def test_verdict_no_uncertainty_states_the_probability_and_the_spread():
    kind, text = V.verdict_no_uncertainty(mv(ONE, settings=S.Settings(alpha=4.0, y=0.5), p_unit=0.6827))
    assert kind == "info" and "Streuung 0,500" in text and "Wahrscheinlichkeit 68,3 %" in text and "Mittel 0,50" in text


def test_model_note_only_for_the_linear_model():
    assert V.verdict_model_note(S.Settings()) is None
    kind, text = V.verdict_model_note(S.Settings(model=C.MODEL_L))
    assert kind == "info" and "ausgegraut" in text


# --- Spielrunde --------------------------------------------------------------------------------------------------------------------------------
def _round(started, share=0.9, exact=True, payoff=0.5):
    import numpy as np
    return S.Round(0.3, np.zeros(10), np.ones(10, bool), 9, 10, share, exact, started, payoff, 0.5)


def test_verdict_round_start_and_no_start():
    kind, text = V.verdict_round(mv(ONE), _round(True))
    assert kind == "success" and "Zustand θ = 0,300. 9 von 10" in text and "exakter Anteil 90,0 % ≥ θ" in text and "das Projekt startet**, ein Mitmachender bekommt 0,50" in text
    kind, text = V.verdict_round(mv(ONE), _round(False, share=0.2, exact=False, payoff=-0.5))
    assert kind == "error" and "gezählter Anteil 20,0 % < θ" in text and "startet nicht" in text and "bekommt −0,50" in text


def test_verdict_round_linear():
    kind, text = V.verdict_round(mv(ONE, settings=S.Settings(model=C.MODEL_L)), _round(None, payoff=0.123))
    assert kind == "info" and "Anteil − θ = 0,123" in text


# --- Experimente --------------------------------------------------------------------------------------------------------------------------------
def test_verdict_e1_names_counts_and_shares():
    summary = {"regime": {"above": {"cells": 100, "multi": 0}, "below": {"cells": 200, "multi": 100}, "below_share": 0.5},
               "linear": {"above": {"cells": 50, "multi": 0}, "below": {"cells": 150, "multi": 30}, "below_share": 0.2}}
    kind, text = V.verdict_e1(summary, 0, 7, 29, 31)
    assert kind == "info" and "Raster 29 × 31" in text and "0 von 100" in text and "100 von 200 (50,0 %)" in text and "30 von 150 (20,0 %)" in text
    assert "0 (α, β)-Zellen unterhalb haben für kein (y, c) mehrere" in text and "7 (α, β)-Zellen unterhalb der Formel sind für alle neun y eindeutig" in text


def test_verdict_e2_regime_band_and_no_band():
    kind, text = V.verdict_e2_regime(8.0, 4.0, 0.5)
    assert "zwischen 0,4163 und 0,5837" in text and "Breite 0,1673" in text and "symmetrisch" in text and "desto niedriger" in text
    assert "symmetrisch" not in V.verdict_e2_regime(8.0, 4.0, 0.3)[1]
    assert "für kein c mehrere Gleichgewichte" in V.verdict_e2_regime(4.0, 4.0, 0.5)[1]


def test_verdict_e2_linear_fold_and_no_fold():
    assert "drei Gleichgewichte für y zwischen" in V.verdict_e2_linear(4.0, 1.0)[1] and "Kritisch bei y = ½: β = 1,922" in V.verdict_e2_linear(4.0, 1.0)[1]
    assert "für kein Prior-Mittel y mehrere Gleichgewichte" in V.verdict_e2_linear(4.0, 16.0)[1]


def test_verdict_e3_limit_gap_and_lost_projects():
    rows = X.e3_rows(1.0, 0.5, 0.25, (1, 16, 256, 10000))
    kind, text = V.verdict_e3(rows, 1.0, 0.5, 0.25)
    assert kind == "info" and "1 − c = 0,750" in text and "Bei β = 10.000 liegt θ̂ bei 0,7508" in text and "(sehr nahe am Grenzwert)" in text and "Verlorene Projekte" in text
    assert "Bei β = 1 liegt θ̂ bei 0,9144, Abstand 0,1644 vom Grenzwert" in text
    half = V.verdict_e3(X.e3_rows(1.0, 0.5, 0.5, (1, 16, 10000)), 1.0, 0.5, 0.5)[1]
    assert "Bei c = ½ ändert das Rauschen den kritischen Zustand nicht" in half and "bei 0,5000 und 0,5000" in half


def test_verdict_e3_without_unique_rows():
    rows = X.e3_rows(8.0, 0.5, 0.5, (1, 2))                 # bei so kleinem β drei Gleichgewichte
    assert all(r["count"] == 3 for r in rows)
    assert "keine der gerechneten Signalgenauigkeiten" in V.verdict_e3(rows, 8.0, 0.5, 0.5)[1]


def test_verdict_e4_counts_and_changes():
    rows = [{"n": 0, "count": 1, "x": (0.5,), "stable": (True,)}, {"n": 2, "count": 1, "x": (1.4,), "stable": (True,)}, {"n": 5, "count": 3, "x": (0.0, 0.5, 1.0), "stable": (True, False, True)}]
    kind, text = V.verdict_e4(rows, 8.0, 4.0, 0.6)
    assert "N = 2: 1, N = 5: 3" in text and "ändert sich die Zahl bei N = 5" in text
    same = [rows[0], rows[1]]
    assert "bleibt bei allen gerechneten N wie im Kontinuum (1)" in V.verdict_e4(same, 8.0, 4.0, 0.6)[1]


def test_verdict_e5_names_both_sides_and_symmetry():
    res = lambda k, reached=True: I.IterationResult(k, reached, 0.0, True)   # noqa: E731
    rows = [{"factor": 0.3, "beta": 3.0, "count": 3, "low": res(8), "high": res(8)}, {"factor": 0.98, "beta": 9.98, "count": 3, "low": res(1244), "high": res(1244)},
            {"factor": 1.02, "beta": 10.39, "count": 1, "low": res(2784), "high": res(2784)}, {"factor": 2.0, "beta": 20.4, "count": 1, "low": res(120), "high": res(120)}]
    kind, text = V.verdict_e5(rows, 8.0)
    assert "von 8 (Faktor 0,30) auf 1.244 (Faktor 0,98)" in text and "von 2.784 (Faktor 1,02) auf 120 (Faktor 2,00)" in text and "Spiegelsymmetrie" in text and "Obergrenze" not in text
    rows[1]["high"] = res(1244, reached=False)
    rows[1]["low"] = res(1200)
    kind, text = V.verdict_e5(rows, 8.0)
    assert "Spiegelsymmetrie" not in text and "Bei 1 Faktoren wurde die Obergrenze von 200.000 Runden erreicht" in text
    assert "Die Faktoren liegen alle auf einer Seite" in V.verdict_e5(rows[:2], 8.0)[1]


def test_verdict_e5_h11_names_the_slopes():
    kind, text = V.verdict_e5_h11(8.0, 4.0, 0.5, 0.5)
    assert "Drei Gleichgewichte" in text and text.count("stabil") >= 3 and "instabil" in text and "B' > 1" in text
    one = V.verdict_e5_h11(4.0, 16.0, 0.5, 0.5)[1]
    assert "Ein Gleichgewicht" in one and "B' = " in one and "stabil" in one


def test_verdict_e6_attack_and_no_attack():
    sm_att = X.e6_summary(0.1, 0.5)                          # q = 1/3
    kind, text = V.verdict_e6(sm_att, 0.1, 0.5)
    assert kind == "success" and "q = L/(L+M) = 0,3333" in text and "Grenze (1−ε)/(2−ε) = 0,4737" in text and "Spieler 1 greift ab Typ 1 an, Spieler 2 ab Typ 1" in text
    sm_no = X.e6_summary(0.1, 1.0)                           # q = ½
    kind, text = V.verdict_e6(sm_no, 0.1, 1.0)
    assert kind == "warning" and "**nie jemand an**" in text and "bis zu 801 Nachrichten" in text and "801 Runden (eine je Nachrichtenstufe)" in text and "0 von 801" in text


def test_verdict_e6_knowledge_margin():
    rows = [{"n": 1, "formula": 0.9, "simulated": 0.9005, "se": 0.001, "games_in_g": 10000}, {"n": 5, "formula": 0.59, "simulated": 0.5902, "se": 0.005, "games_in_g": 10000}]
    kind, text = V.verdict_e6_knowledge(rows)
    assert kind == "success" and "10.000 simulierte Spiele" in text and "n = 1: 0,9000 gegen 0,9005" in text and "Größte Abweichung 0,5 Standardfehler (innerhalb der Marge von 4)" in text
    rows[0]["simulated"] = 0.91
    kind, text = V.verdict_e6_knowledge(rows)
    assert kind == "warning" and "außerhalb" in text and math.isclose(max(abs(r["simulated"] - r["formula"]) / r["se"] for r in rows), 10.0, rel_tol=1e-6)
