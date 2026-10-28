"""Unsicherheit und Gleichgewicht: globale Spiele - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Viertes Stück der Reihe Koordinationsspiele. Betriebe kennen den Zustand (wie viele müssen mitmachen, damit sich ein gemeinsamer Zubringer trägt) nur über ein verrauschtes privates Signal.
Die Demo zählt die Gleichgewichte in Schwellenstrategien, zeigt, wo die Eindeutigkeit endet, was endlich viele Betriebe ändern, wie lange die Beste-Antwort-Iteration braucht
und warum „jeder weiß es“ weniger ist als gemeinsames Wissen (E-Mail-Spiel).

Lauffähig mit: streamlit run app.py
"""

import math

import streamlit as st

import glo_constants as C
import glo_email as EM
import glo_evaluation as V
import glo_experiments as X
import glo_finite as F
import glo_linear as L
import glo_regime as R
import glo_scenario as S
import glo_visualization as Z
from glo_layout import dunkles_design, kopf_css
from glo_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from glo_rng import SplitMix64

st.set_page_config(page_title="Unsicherheit und Gleichgewicht: globale Spiele – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _finite(n, alpha, beta, y, c):
    return F.roots_finite(n, alpha, beta, y, c)


@st.cache_data(show_spinner=False)
def _grid():
    return X.load_grid()


@st.cache_data(show_spinner=False)
def _e4(alpha, beta, y, c):
    return X.e4_rows(alpha, beta, y, c)


@st.cache_data(show_spinner=False)
def _e5(alpha):
    return X.e5_rows(alpha), X.e5_linear_rows(alpha)


@st.cache_data(show_spinner=False)
def _e6(eps, loss):
    return X.e6_summary(eps, loss)


@st.cache_data(show_spinner=False)
def _e6_knowledge(eps):
    return X.e6_knowledge_rows(eps)


def show(verdict):
    kind, text = verdict
    getattr(st, kind)(text)


def table(rows):
    st.dataframe(rows, hide_index=True, width="stretch")


def yes_no(flag: bool) -> str:
    return "ja" if flag else "nein"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in glo_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🌐 Unsicherheit und Gleichgewicht: globale Spiele")
st.markdown(
    """
**Viertes Stück der Reihe Koordinationsspiele.** Das erste Stück (`koordinationsspiel-demo`) zeigt die Gleichgewichte eines Koordinationsspiels, das zweite (`gleichgewichtsauswahl-demo`), welche Konvention auf Dauer gewählt wird,
das dritte (`konventionswechsel-demo`), wie lange ein Wechsel dauert. Hier geht es um die Frage davor: **Betriebe legen ihre Sendungen auf einen gemeinsamen Zubringer zum Hub (mitmachen) oder fahren direkt.**
Der Zubringer trägt sich nur, wenn mindestens der Anteil **θ** der Betriebe mitmacht, und θ kennt niemand genau: jeder Betrieb hat nur eine **private, verrauschte Schätzung** (sein Signal).
**Wird aus „alle machen mit / keiner macht mit“ dadurch ein eindeutiges Gleichgewicht, und wo genau endet diese Eindeutigkeit?** Gezählt wird, nicht behauptet.

Nicht zu verwechseln mit `globaler-schnitt-demo` (dort geht es um den globalen Schnitt in Graphen): „globales Spiel“ heißt hier ein Koordinationsspiel mit Unsicherheit über den Zustand.
**Eigennutz, Schwellenstrategien und die Beste-Antwort-Regel sind Modellannahmen**, keine Aussagen über Menschen oder echte Terminals.

Weiter unten: sechs Tabs mit je einem **🔬 Experiment**, die Abschnitte **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Was ein globales Spiel ist", expanded=True):
    st.markdown(
        r"""
1. **Zustand und Signale.** Der Zustand $\theta$ ist normalverteilt mit Mittel $y$ und Genauigkeit $\alpha$ (Streuung $1/\sqrt{\alpha}$). Jeder Betrieb sieht $x_i = \theta + e_i$ mit unabhängigem Rauschen der Genauigkeit $\beta$.
2. **Schwellenstrategie.** Ein Betrieb macht mit, wenn sein Signal höchstens $x^*$ ist. Welche Schwellen $x^*$ sind ein Gleichgewicht, wenn alle anderen dieselbe Schwelle spielen? Die Antwort ist eine Zahl: **ein, zwei oder drei** Gleichgewichte; die Demo zählt sie, bevor sie auf eine Formel verweist.
3. **Zwei Modelle.** *Modell R (Regimewechsel, Hauptmodell):* das Projekt startet, wenn der Anteil der Mitmachenden mindestens $\theta$ erreicht; Mitmachen bringt 1 bei Start und kostet immer $c$. *Modell L (linear):* Mitmachen bringt (Anteil der Mitmachenden − $\theta$).
4. **Ohne Unsicherheit** (alle kennen $\theta$) gibt es für jedes $\theta$ in $(0, 1)$ zwei Gleichgewichte, „alle“ und „keiner“. Mit Signalen kann es eines geben, aber nicht immer: genau das ist die Frage.
        """
    )

with st.expander("Erklärtabelle: die zwei Auszahlungsformen"):
    st.markdown(
        """
| | **Modell R: Regimewechsel** | **Modell L: linear** |
|---|---|---|
| **Was Mitmachen bringt** | 1 bei Start, sonst 0, minus Aufwand c (immer fällig). Start, wenn der Anteil Mitmachender mindestens θ ist. | Anteil der anderen Mitmachenden minus θ. |
| **Reduzierte Gleichung** | α·Φ(w) − √β·w = α·y + z_c·√(α+β), mit w = Φ⁻¹(θ̂) | Φ(κ(m − y)) = m, κ = α·√((α+β)/(β(α+2β))) |
| **Gesucht** | kritischer Zustand θ̂ (das Projekt startet für θ ≤ θ̂) und Schwelle x* | Anteil m bei Indifferenz und Schwelle x* |
| **Faustformel β ≥ α²/(2π)** | genau die Eindeutigkeitsgrenze über alle (y, c) | hinreichend, nicht notwendig (die wahre Grenze bei y = ½ liegt tiefer) |
| **Endlich viele Betriebe N** | ändert die Schwellen und kann Gleichgewichte hinzufügen oder entfernen (Experiment E4) | kein Effekt: der Erwartungsnutzen ist für jedes N gleich dem Kontinuumswert |
| **Aufwand c** | wirkt (verschiebt das Band mit drei Gleichgewichten) | gibt es nicht (Regler ausgegraut) |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("🧭 Modell")
    model = st.selectbox("Auszahlung", C.MODELS, key="model_select",
                         help="Regimewechsel: das Projekt startet, wenn genug mitmachen. Linear: Mitmachen bringt Anteil minus θ. Die Experimente E3 und E4 rechnen immer das Modell R, E5 beide Modelle.")
    is_regime = model == C.MODEL_R
    st.header("🎚️ Unsicherheit")
    y = st.slider("Prior-Mittel y", *bounds("y_slider"), key="y_slider", step=0.05, help="Mittel des Zustands θ unter dem Prior (θ ist der Mindestanteil, ab dem sich der Zubringer trägt).")
    alpha = st.slider("Prior-Genauigkeit α", *bounds("alpha_slider"), key="alpha_slider", step=0.5, help="Genauigkeit des Priors (1/Varianz). Die Streuung ist 1/√α: je größer α, desto besser kennen alle θ schon vorab.")
    beta = st.slider("Signal-Genauigkeit β", *bounds("beta_slider"), key="beta_slider", step=0.1, help="Genauigkeit des privaten Signals (1/Varianz des Rauschens). Die Streuung ist 1/√β. Ab β ≥ α²/(2π) gibt es im Modell R für jedes (y, c) nur ein Gleichgewicht.")
    st.caption(f"Streuung des Priors 1/√α = {V.de(1 / math.sqrt(alpha), 3)}; Streuung des Signals 1/√β = {V.de(1 / math.sqrt(beta), 3)}.")
    st.header("⚖️ Aufwand und Betriebe")
    c = st.slider("Aufwand c (nur Modell R)", *bounds("c_slider"), key="c_slider", step=0.01, disabled=not is_regime,
                  help="Aufwand der Umstellung, immer fällig. Nur im Modell R wirksam; im Modell L ausgegraut.")
    players = st.selectbox("Zahl der Betriebe (nur Modell R)", C.PLAYERS, key="players_select", disabled=not is_regime,
                           format_func=lambda n: "Kontinuum (unendlich viele)" if n == 0 else str(n),
                           help="Kontinuum oder N Betriebe, von denen jeder die N − 1 anderen zählt. Im Modell L ändert N nichts (Erwartungsnutzen gleich dem Kontinuumswert); dort ausgegraut.")
    st.header("🎲 Spielrunde")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"model_select": model, "y_slider": float(y), "alpha_slider": float(alpha), "beta_slider": float(beta), "c_slider": float(c), "players_select": int(players), "seed_input": int(seed)})

settings = S.Settings(model=model, y=float(y), alpha=float(alpha), beta=float(beta), c=float(c), players=int(players), seed=int(seed))
finite_eq = None
if settings.finite:
    est = X.finite_estimated_seconds(settings.players)
    with st.spinner(f"Zähle die Gleichgewichte bei N = {settings.players} Betrieben (rund {max(1, math.ceil(est))} Sekunden) …"):
        finite_eq = _finite(settings.players, settings.alpha, settings.beta, settings.y, settings.c)
view = V.analyse(settings, finite_eq)
verdicts = []   # Verdikte für den PDF-Export

# --- Hauptansicht --------------------------------------------------------------------------------------------------------------------------
st.markdown("## 📊 Wie viele Gleichgewichte gibt es?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Gleichgewichte (Schwellenstrategien)", str(view.count), help="Gezählt, nicht aus einer Formel gelesen.")
m2.metric("Faustformel β ≥ α²/(2π)", "erfüllt" if view.formula_ok else "nicht erfüllt", help=f"α²/(2π) = {V.de(view.formula_bound, 3)}; hinreichend, nicht notwendig (Modell L), genau die Grenze über alle (y, c) (Modell R).")
m3.metric("stabil / instabil", f"{sum(r.stable for r in view.rows)} / {sum(not r.stable for r in view.rows)}", help="Stabil: die Beste-Antwort-Iteration zieht dorthin; instabil: eine kleine Abweichung wächst.")
m4.metric("P(θ ∈ (0, 1))", V.pct(view.p_unit), help="Ohne Unsicherheit gäbe es nur in diesen Zuständen die zwei Gleichgewichte „alle“ und „keiner“.")

st.markdown("#### ❓ Kernfrage: Wie viele Gleichgewichte gibt es, und trägt die Faustformel?")
v_count = V.verdict_count(view)
show(v_count)
verdicts.append(v_count)
v_formula = V.verdict_formula(view)
show(v_formula)
verdicts.append(v_formula)
note = V.verdict_model_note(settings)
if note is not None:
    show(note)
show(V.verdict_no_uncertainty(view))

col_curve, col_table = st.columns([3, 2])
with col_curve:
    if is_regime:
        st.plotly_chart(Z.build_regime_curve(settings, R.roots_regime(settings.alpha, settings.beta, settings.y, settings.c)), width="stretch", key="curve_chart")
        st.caption("Kontinuum-Kurve: Wie sicher ist der Betrieb am Indifferenzsignal, dass das Projekt startet, wenn alle anderen auf den kritischen Zustand θ̂ setzen? Gleichgewicht, wo die Kurve die rote Linie c schneidet."
                   + (" Die Punkte unten gelten für N Betriebe und liegen daher nicht auf dieser Kurve." if settings.finite else ""))
    else:
        st.plotly_chart(Z.build_linear_curve(settings, L.roots_linear(settings.alpha, settings.beta, settings.y)), width="stretch", key="curve_chart")
        st.caption("Der Anteil Mitmachender, den der indifferente Betrieb erwartet, gegen den Anteil m; Gleichgewicht, wo die Kurve die Diagonale schneidet.")
with col_table:
    level_name = "kritischer Zustand θ̂" if is_regime else "Anteil m"
    table([{"Nr.": i + 1, "Gleichgewicht": r.label, "Schwelle x*": V.fmt_x(r.x), level_name: "–" if r.level is None else V.de(r.level, 4), "stabil": yes_no(r.stable)} for i, r in enumerate(view.rows)])
    if view.scan_points:
        st.caption(f"Bei N Betrieben auf einem Raster mit {view.scan_points} Punkten gezählt: ein Paar dicht an einem Berührpunkt könnte fehlen. Kontinuum zum Vergleich: {view.continuum_count} Gleichgewicht{'e' if view.continuum_count != 1 else ''}.")

st.markdown("**Eine Spielrunde** (Zufalls-Seed der Seitenleiste): Zustand ziehen, private Signale ziehen, jeder entscheidet nach der Schwelle")
labels = [f"{r.label} (x* = {V.fmt_x(r.x)})" for r in view.rows]
pick = st.selectbox("Gespieltes Gleichgewicht", list(range(view.count)), format_func=lambda i: labels[i], key=f"round_eq_{view.count}") if view.count > 1 else 0
row = view.rows[pick]
rnd = S.draw_round(settings, row.x, row.level if (is_regime and not settings.finite) else None)
st.plotly_chart(Z.build_round(rnd), width="stretch", key="round_chart")
v_round = V.verdict_round(view, rnd)
show(v_round)
verdicts.append(v_round)
st.caption("Eine einzelne Runde zeigt, wie ein Spiel aussieht, nicht, wie oft ein Projekt startet." + (f" Im Kontinuum werden {C.DRAW_CONTINUUM} Betriebe gezeichnet; der Anteil selbst ist exakt gerechnet." if (not settings.finite) else ""))

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/grid.json) fehlt oder gehört zu einem anderen Rechencode: `python tools/make_grid.py` ausführen. Das Experiment E1 zeigt dann keine Karte.")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1 Wo gibt es drei?", "2 Prior und Aufwand verschieben das Band", "3 Was kostet die Unsicherheit?", "4 Endlich viele Betriebe", "5 Runden bis zur Eindeutigkeit", "6 Gemeinsames Wissen"])

# --- Tab 1 ---------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader("🔬 Experiment E1: Wo gibt es drei Gleichgewichte?")
    st.caption("Karte der (α, β)-Zellen: Farbe = Anteil der (y, c)-Werte (Modell R, 25 Werte) bzw. y-Werte (Modell L, 9 Werte) des Rasters, für die es mehr als ein Gleichgewicht gibt. Vorgerechnet mit der Wurzelsuche, nicht mit dem geschlossenen Band.")
    e1_model = st.radio("Modell der Karte", C.MODELS, horizontal=True, key="e1_model")
    if grid is not None:
        st.plotly_chart(Z.build_map(grid, e1_model, settings), width="stretch", key="e1_chart")
        v1 = V.verdict_e1(X.grid_summary(grid), X.grid_below_formula_without_multi(grid, "regime"), X.grid_below_formula_without_multi(grid, "linear"), len(grid["alphas"]), len(grid["betas"]))
        show(v1)
        verdicts.append(v1)

# --- Tab 2 ---------------------------------------------------------------------------------------------------------------------------------
with tab2:
    st.subheader("🔬 Experiment E2: Prior und Aufwand verschieben das Band mit drei Gleichgewichten")
    st.caption("Links Modell R: Bereich des Aufwands c mit drei Gleichgewichten, je Prior-Mittel y (α und β aus der Seitenleiste). Rechts Modell L: Bereich des Prior-Mittels y mit drei Gleichgewichten, je β (Faltenkurve; α aus der Seitenleiste). Beides geschlossen gerechnet.")
    left, right = st.columns(2)
    with left:
        ys = [round(0.05 + 0.025 * i, 3) for i in range(37)]
        rows2r = X.e2_regime_band_rows(settings.alpha, settings.beta, ys)
        st.plotly_chart(Z.build_c_band(rows2r, settings), width="stretch", key="e2_chart_r")
        v2r = V.verdict_e2_regime(settings.alpha, settings.beta, settings.y)
        show(v2r)
        verdicts.append(v2r)
    with right:
        betas2 = [round(0.1 * (1000.0 ** (i / 60.0)), 4) for i in range(61)]
        rows2l = X.e2_linear_fold_rows(settings.alpha, betas2)
        st.plotly_chart(Z.build_fold(rows2l, settings), width="stretch", key="e2_chart_l")
        v2l = V.verdict_e2_linear(settings.alpha, settings.beta)
        show(v2l)
        verdicts.append(v2l)

# --- Tab 3 ---------------------------------------------------------------------------------------------------------------------------------
with tab3:
    st.subheader("🔬 Experiment E3: Was kostet die Unsicherheit?")
    st.caption(f"Modell R mit α = {V.de(settings.alpha, 1)}, y = {V.de(settings.y, 2)}, c = {V.de(settings.c, 2)} aus der Seitenleiste (unabhängig vom gewählten Modell). Der kritische Zustand θ̂ wird mit genaueren Signalen mit dem Grenzwert 1 − c verglichen, den ein Betrieb ohne Rauschen wählen würde.")
    rows3 = X.e3_rows(settings.alpha, settings.y, settings.c)
    st.plotly_chart(Z.build_theta_over_beta(rows3, settings.c), width="stretch", key="e3_chart")
    table([{"β": V.thousands(r["beta"]), "Gleichgewichte": r["count"], "θ̂": ", ".join(V.de(t, 4) for t in r["thetas"]), "Grenzwert 1 − c": V.de(r["limit"], 3),
            "P(Start)": "–" if r["p_start"] is None else V.pct(r["p_start"]), "verloren P(θ̂ < θ < 1)": "–" if r["p_lost"] is None else V.pct(r["p_lost"])} for r in rows3])
    v3 = V.verdict_e3(rows3, settings.alpha, settings.y, settings.c)
    show(v3)
    verdicts.append(v3)

# --- Tab 4 ---------------------------------------------------------------------------------------------------------------------------------
with tab4:
    st.subheader("🔬 Experiment E4: Endlich viele Betriebe")
    st.caption(f"Modell R mit α = {V.de(settings.alpha, 1)}, β = {V.de(settings.beta, 1)}, y = {V.de(settings.y, 2)}, c = {V.de(settings.c, 2)} aus der Seitenleiste. Jeder Betrieb zählt die N − 1 anderen; sein eigener Beitrag zählt für den Start mit. "
               "Im Modell L ändert N nichts: E[K/(N−1) | θ] = Φ(√β(x − θ)) für jedes N.")
    sig4 = (settings.alpha, settings.beta, settings.y, settings.c)
    if st.button("▶️ Gleichgewichte für alle Spielerzahlen zählen", key="e4_button", help=f"Zählt für N = {', '.join(str(n) for n in C.E4_PLAYERS)}; rechnet rund {max(1, math.ceil(X.e4_estimated_seconds()))} Sekunden, der größte Teil für N = 1 000."):
        st.session_state["e4_signature"] = sig4
    if st.session_state.get("e4_signature") == sig4:
        with st.spinner(f"Zähle für {len(C.E4_PLAYERS)} Spielerzahlen (rund {max(1, math.ceil(X.e4_estimated_seconds()))} Sekunden) …"):
            rows4 = _e4(*sig4)
        st.plotly_chart(Z.build_players(rows4), width="stretch", key="e4_chart")
        table([{"Betriebe N": "Kontinuum" if r["n"] == 0 else r["n"], "Gleichgewichte": r["count"], "Schwellen x*": ", ".join(V.de(x, 3) for x in r["x"]), "stabil": ", ".join(yes_no(s) for s in r["stable"])} for r in rows4])
        v4 = V.verdict_e4(rows4, settings.alpha, settings.beta, settings.c)
        show(v4)
        verdicts.append(v4)

# --- Tab 5 ---------------------------------------------------------------------------------------------------------------------------------
with tab5:
    st.subheader("🔬 Experiment E5: Runden bis zur Eindeutigkeit")
    st.caption("Beste-Antwort-Iteration (eine Modellannahme): alle spielen die Schwelle x, dann spielen alle die beste Antwort darauf, und so weiter. Start „kaum einer macht mit“ (x₀ = −50) und „fast alle“ (x₀ = +50).")
    st.markdown("**Die Folge zu den Einstellungen der Seitenleiste** (Modell R, Kontinuum)")
    eq5 = R.roots_regime(settings.alpha, settings.beta, settings.y, settings.c)
    trajs = X.e5_trajectories(settings.alpha, settings.beta, settings.y, settings.c)
    st.plotly_chart(Z.build_trajectories(trajs, eq5.x), width="stretch", key="e5_traj_chart")
    v5h = V.verdict_e5_h11(settings.alpha, settings.beta, settings.y, settings.c)
    show(v5h)
    verdicts.append(v5h)
    st.markdown("**Wie lange dauert es nahe der Grenze?** Modell R bei y = c = ½, β als Vielfaches der Grenze α²/(2π); Modell L bei y = ½, β als Vielfaches des kritischen β.")
    e5_alpha = st.selectbox("Prior-Genauigkeit α der Reihe", (2.0, 4.0, 8.0, 16.0), index=2, key="e5_alpha", format_func=lambda a: V.de(a, 0))
    if st.button("▶️ Runden zählen", key="e5_button", help=f"Zählt Runden für {len(C.E5_FACTORS)} Faktoren in zwei Modellen; höchstens {V.thousands(C.E5_CAP)} Runden je Start, rund {max(1, math.ceil(X.e5_estimated_seconds(e5_alpha)))} Sekunden."):
        st.session_state["e5_signature"] = e5_alpha
    if st.session_state.get("e5_signature") == e5_alpha:
        with st.spinner(f"Zähle Runden (rund {max(1, math.ceil(X.e5_estimated_seconds(e5_alpha)))} Sekunden) …"):
            rows5, rows5l = _e5(e5_alpha)
        st.plotly_chart(Z.build_rounds(rows5, rows5l), width="stretch", key="e5_chart")
        table([{"Faktor": V.de(r["factor"], 2), "β": V.de(r["beta"], 3), "Gleichgewichte": r["count"], "Runden (Modell R)": V.thousands(r["low"].rounds) + ("" if r["low"].reached and r["high"].reached else " (abgebrochen)"),
                "Runden (Modell L)": V.thousands(l["low"].rounds) + ("" if l["low"].reached and l["high"].reached else " (abgebrochen)")} for r, l in zip(rows5, rows5l)])
        v5 = V.verdict_e5(rows5, e5_alpha)
        show(v5)
        verdicts.append(v5)

# --- Tab 6 ---------------------------------------------------------------------------------------------------------------------------------
with tab6:
    st.subheader("🔬 Experiment E6: Gemeinsames Wissen (E-Mail-Spiel)")
    st.caption("Eigene Modellform, angelehnt an Rubinstein, nicht gegen die Originalarbeit geprüft. Zustand G (Koordination lohnt: beide angreifen bringt M = 1 je Spieler, einsamer Angriff −L) mit Wahrscheinlichkeit ½, sonst B (Angriff bringt −L). "
               "In G schickt Spieler 1 eine Nachricht, jede Nachricht geht mit Wahrscheinlichkeit ε verloren, jede angekommene wird bestätigt. Der Typ eines Spielers ist die Zahl der Nachrichten, die er gesendet hat; Typen bis 400 (der Rest greift nie an).")
    ec1, ec2 = st.columns(2)
    e6_eps = ec1.slider("Verlustwahrscheinlichkeit ε je Nachricht", C.EMAIL_EPS_MIN, C.EMAIL_EPS_MAX, C.EMAIL_DEFAULT_EPS, step=0.01, key="e6_eps")
    e6_loss = ec2.slider("Verlust L bei einsamem Angriff (Gewinn M = 1)", C.EMAIL_L_MIN, C.EMAIL_L_MAX, C.EMAIL_DEFAULT_L, step=0.05, key="e6_loss")
    sm = _e6(float(e6_eps), float(e6_loss))
    e6m1, e6m2, e6m3, e6m4 = st.columns(4)
    e6m1.metric("Schwelle q = L/(L+M)", V.de(sm["q"], 4))
    e6m2.metric("Grenze (1−ε)/(2−ε)", V.de(sm["q_star"], 4))
    e6m3.metric("Angriff im größten Gleichgewicht", yes_no(sm["eq"].attacks))
    e6m4.metric("G gemeinsame q-Überzeugung in", f"{V.thousands(sm['common_states'])} von {V.thousands(sm['g_states'])}")
    st.plotly_chart(Z.build_email_beliefs(sm), width="stretch", key="e6_chart")
    v6 = V.verdict_e6(sm, float(e6_eps), float(e6_loss))
    show(v6)
    verdicts.append(v6)
    st.markdown("**Ein Spiel Nachricht für Nachricht** (Zufalls-Seed der Seitenleiste)")
    is_g, k, arrived = EM.draw_game(float(e6_eps), C.EMAIL_P_G, SplitMix64(int(seed)))
    if not is_g:
        st.info("Zustand B: es wird keine Nachricht gesendet, beide Spieler haben Typ 0.")
    else:
        t1, t2 = EM.messages_to_types(k)
        eq6 = sm["eq"]
        a1 = bool(eq6.s1[t1]) if t1 < len(eq6.s1) - 1 else False
        a2 = bool(eq6.s2[t2]) if t2 < len(eq6.s2) - 1 else False
        trail = ", ".join(f"Nachricht {i + 1} {'kommt an' if ok else 'geht verloren'}" for i, ok in enumerate(arrived[:12])) + (" …" if len(arrived) > 12 else "")
        st.info(f"Zustand G, {k} gesendete Nachricht{'en' if k != 1 else ''} ({trail}). Typ von Spieler 1: {t1}, von Spieler 2: {t2}. Im größten Gleichgewicht greift Spieler 1 {'an' if a1 else 'nicht an'}, Spieler 2 {'an' if a2 else 'nicht an'}.")
    if st.button("▶️ Formel gegen Simulation rechnen", key="e6_sim_button", help=f"Simuliert {V.thousands(C.EMAIL_SIM_GAMES)} Spiele Nachricht für Nachricht; rechnet unter einer Sekunde."):
        st.session_state["e6_sim_eps"] = float(e6_eps)
    if st.session_state.get("e6_sim_eps") == float(e6_eps):
        with st.spinner("Simuliere das Protokoll …"):
            know = _e6_knowledge(float(e6_eps))
        table([{"Stufe n": r["n"], "(1−ε)^n": V.de(r["formula"], 4), "simuliert": V.de(r["simulated"], 4), "Standardfehler": V.de(r["se"], 4)} for r in know])
        v6k = V.verdict_e6_knowledge(know)
        show(v6k)
        verdicts.append(v6k)

# --- Verwandte Demos -----------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🔗 Verwandte Demos")
st.markdown(
    """
- **`koordinationsspiel-demo`** (Stück 1 der Reihe): die Gleichgewichte und die Schwelle selbst, Risiko- gegen Auszahlungsdominanz; hier ist die Auszahlungsschwelle c das p* der Hirschjagd.
- **`gleichgewichtsauswahl-demo`** (Stück 2): welche Konvention auf Dauer gewählt wird; hier fehlt jede Dynamik außer der Beste-Antwort-Iteration.
- **`konventionswechsel-demo`** (Stück 3): wie lange ein Wechsel dauert; hier gibt es keine Fehlerrate, nur die Frage nach der Zahl der Gleichgewichte.
- **`komplexe-ansteckung-demo`**: Schwellenregeln auf Netzen, aber ohne Auszahlungen und ohne Signale.
- **`korreliert-demo`**: Koordination über ein öffentliches Vermittlersignal; hier sind die Signale privat und verrauscht.
- **`myerson-satterthwaite-demo`**: Bayes-Nash-Gleichgewichte mit privater Information, aber für Handel, nicht für Koordination.
- **`globaler-schnitt-demo`**: der globale Schnitt in Graphen, nicht verwandt (nur der Name ähnelt sich).
"""
)

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Nur Schwellenstrategien** („mitmachen genau dann, wenn das Signal ≤ x*“) | Andere Gleichgewichte (nicht-monotone Strategien) sind **weder ausgeschlossen noch gesucht**; die Zahlen zählen nur Schwellen-Gleichgewichte. Dass es keine anderen gibt, ist nicht gezeigt. | – |
| **Eigennutz und Beste-Antwort-Iteration sind Modellannahmen** | Gemessen ist diese Regel, nicht das Verhalten von Menschen oder Terminals. Die Runden von E5 sind eine Eigenschaft der Antwortfunktion, kein Lernprozess. | `gleichgewichtsauswahl-demo` |
| **Gleiche Signalgenauigkeit für alle, normalverteilter Prior und normalverteiltes Rauschen** | Ungleiche Genauigkeiten, schiefe Priors und dicke Ränder ändern die Zahl der Gleichgewichte und die Grenze; nicht gerechnet. | – |
| **Kontinuum und N Betriebe sind verschiedene Spiele** | Im Modell R ändert N die Schwellen und kann Gleichgewichte hinzufügen oder entfernen (E4); gezählt wird auf einem Raster, ein Paar dicht an einem Berührpunkt könnte fehlen. Im Modell L ändert N nichts. | – |
| **Randwurzeln bei kleinem β** | Bei kleinem β liegen äußere Wurzeln des Modells L bei Anteilen von 10^-10 und weniger: „praktisch nie mitmachen“ bzw. „praktisch immer“. Ein festes x-Gitter sähe sie nicht. | – |
| **E-Mail-Spiel: eigene Modellform** | Angelehnt an Rubinstein, nicht gegen die Originalarbeit geprüft; Typen bis 400 abgeschnitten (der Rest greift nie an); Wahrscheinlichkeit von G fest ½. | – |
| **Strategische Substitute** | Globale Spiele, in denen Mitmachen mit der Zahl der Mitmachenden *weniger* lohnt, sind nicht gerechnet; keine Literatur geprüft. | – |
| **Eindeutigkeit ist kein Realismusbefund** | Ein einziges Gleichgewicht heißt: unter diesen Annahmen gibt es nur eine Schwelle, nicht, dass Menschen sie wählen. | `koordinationsspiel-demo` |
    """
)

st.markdown("---")
with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Prior-Mittel $y$ und -Genauigkeit $\alpha$ des Zustands $\theta$, die Signal-Genauigkeit $\beta$ und (Modell R) der Aufwand $c$; gesucht sind alle Schwellen $x^*$, die ein Gleichgewicht sind, wenn alle anderen dieselbe Schwelle spielen. Die Herleitungen sind eigene Rechnung, durch Zählung, zweite Rechenwege und Simulation gestützt.

| Symbol | Bedeutung |
|---|---|
| $y,\ \alpha$ | Mittel und Genauigkeit des Priors: $\theta \sim \mathcal N(y, 1/\alpha)$ |
| $\beta$ | Genauigkeit des Signals $x_i = \theta + e_i$, $e_i \sim \mathcal N(0, 1/\beta)$ |
| $x^*$ | Schwelle der Strategie „mitmachen, wenn $x_i \le x^*$“ |
| $\hat\theta,\ w$ | kritischer Zustand (Projekt startet für $\theta \le \hat\theta$) und $w = \Phi^{-1}(\hat\theta)$ |
| $c,\ z_c$ | Aufwand und $z_c = \Phi^{-1}(c)$ |
| $\rho,\ d$ | $\rho = \alpha/\sqrt{\beta}$ und $d = (\alpha y + z_c\sqrt{\alpha+\beta})/\sqrt{\beta}$ |
| $m,\ \kappa$ | Modell L: Anteil bei Indifferenz und $\kappa = \alpha\sqrt{(\alpha+\beta)/(\beta(\alpha+2\beta))}$ |
| $N,\ K$ | Zahl der Betriebe und Zahl der mitmachenden anderen |
| $\varepsilon,\ q$ | E-Mail-Spiel: Verlustwahrscheinlichkeit und Schwelle $q = L/(L+M)$ |

**Modell R, Anteil der Mitmachenden** (Kontinuum):

$$\ell(\theta) = \Phi\!\left(\sqrt{\beta}\,(x^* - \theta)\right)$$

**Kritischer Zustand und Schwelle** (das Projekt startet genau für $\theta \le \hat\theta$):

$$\ell(\hat\theta) = \hat\theta, \qquad x^* = \hat\theta + \frac{w}{\sqrt{\beta}}, \qquad w = \Phi^{-1}(\hat\theta)$$

**Indifferenz am Signal $x^*$** mit $m(x) = \frac{\alpha y + \beta x}{\alpha + \beta}$:

$$\Phi\!\left(\sqrt{\alpha+\beta}\,\bigl(\hat\theta - m(x^*)\bigr)\right) = c$$

**Reduzierte Gleichung** in $w$ (alle Gleichgewichte sind ihre Lösungen):

$$\alpha\,\Phi(w) - \sqrt{\beta}\,w = \alpha y + z_c\sqrt{\alpha+\beta}$$

**Gestalt der linken Seite.** Mit $G(w) = \rho\,\Phi(w) - w$ ist

$$G'(w) = \rho\,\varphi(w) - 1$$

Für $\rho > \sqrt{2\pi}$ verschwindet $G'$ bei $\pm w_0$,

$$w_0 = \sqrt{2\ln\!\bigl(\rho/\sqrt{2\pi}\bigr)}$$

und es gibt genau dann **drei** Lösungen, wenn

$$G(-w_0) < d < G(w_0)$$

Für $\rho \le \sqrt{2\pi}$, also $\beta \ge \alpha^2/(2\pi)$, fällt $G$ überall und es gibt für jedes $(y, c)$ genau eine Lösung.

**Stabilität** der Beste-Antwort-Iteration (Steigung $B'$ der Antwortfunktion an der Wurzel, $s = \sqrt{\beta}\,\varphi(w)$):

$$B' = \frac{\alpha+\beta}{\beta}\cdot\frac{s}{1+s}, \qquad B' < 1 \iff G'(w) < 0$$

**Modell L** (Mitmachen bringt Anteil − $\theta$):

$$\Phi\!\bigl(\kappa\,(m - y)\bigr) = m, \qquad x^* = \frac{m(\alpha+\beta) - \alpha y}{\beta}$$

Mehrere Lösungen genau dann, wenn $\kappa > \sqrt{2\pi}$ und $y$ im Bereich $(1 - Y, Y)$ liegt:

$$Y(\kappa) = \Phi(z_0) - z_0\,\varphi(z_0), \qquad z_0 = \sqrt{2\ln\!\bigl(\kappa/\sqrt{2\pi}\bigr)}$$

Bei $y = \tfrac12$ liegt die Grenze bei $\kappa^2 = 2\pi$, also bei der positiven Wurzel von

$$4\pi\beta^2 + (2\pi\alpha - \alpha^2)\beta - \alpha^3 = 0$$

**Endlich viele Betriebe** (Modell R). Die $N-1$ anderen spielen die Schwelle $x$; $K$ zählt die Mitmachenden unter ihnen:

$$K \sim \mathrm{Bin}\!\left(N-1,\ \Phi\!\left(\sqrt{\beta}\,(x - \theta)\right)\right)$$

Die Startwahrscheinlichkeit eines Betriebs mit Signal $s$ (der eigene Beitrag zählt mit) ist

$$P_N(s;x) = \int \Pr\!\left(\frac{1+K}{N} \ge \theta\right) \mathcal N\!\left(\theta;\ m(s),\ \frac{1}{\alpha+\beta}\right) \mathrm{d}\theta$$

und ein Gleichgewicht löst $P_N(x;x) = c$. Im Modell L ist $E[K/(N-1) \mid \theta]$ gleich dem Kontinuumswert, N ändert dort nichts.

**E-Mail-Spiel.** Bei $k$ gesendeten Nachrichten (die $k$-te geht verloren) haben die Spieler die Typen

$$P(k) = p_G\,\varepsilon\,(1-\varepsilon)^{k-1}, \qquad t_1 = \lceil k/2\rceil, \quad t_2 = \lfloor k/2\rfloor$$

Ein Typ greift an, wenn die bedingte Wahrscheinlichkeit „G und der andere greift an“ mindestens $q$ ist. Das größte Gleichgewicht enthält einen Angriff genau dann, wenn (für $p_G = \tfrac12$, $\varepsilon \le \tfrac12$)

$$q \le \frac{1-\varepsilon}{2-\varepsilon}$$

**Gemeinsame $p$-Überzeugung** von $G$ ist der größte Fixpunkt des Operators

$$F \mapsto F \cap B_1^p(F) \cap B_2^p(F)$$

wobei $B_i^p(F)$ die Typen von Spieler $i$ sind, die $F$ mit Wahrscheinlichkeit mindestens $p$ für wahr halten.

**Literatur.** Die Quellen stehen in der README mit Stand der Prüfung; Formeln und Zahlen dieser Demo sind eigene Rechnung.
        """
    )

st.markdown("---")
st.subheader("📄 Ergebnisse als PDF")
if st.button("PDF erstellen", key="pdf_button"):
    from glo_pdf_export import build_pdf
    st.session_state["pdf_bytes"] = build_pdf(view, verdicts)
if st.session_state.get("pdf_bytes"):
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="globale-spiele.pdf", mime="application/pdf", key="pdf_download")

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
