"""Kennzahlen der Hauptansicht und Verdikte (reine Funktionen: Art und Text, die App ruft `getattr(st, Art)(Text)`)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import glo_constants as C
import glo_finite as F
import glo_grid as GRID
import glo_iteration as I
import glo_linear as L
import glo_regime as R
import glo_scenario as S


# --- Formate ----------------------------------------------------------------------------------------------------------------------------------
def de(x: float, nd: int = 2) -> str:
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen (−)."""
    text = f"{x:.{nd}f}"
    if float(text) == 0.0:
        text = text.lstrip("-")                      # „−0,000“ gibt es nicht
    return text.replace(".", ",").replace("-", "−")


def thousands(n: int | float) -> str:
    """Ganze Zahl mit Punkt als Tausendertrenner."""
    return f"{int(round(n)):,}".replace(",", ".")


def pct(x: float, nd: int = 1) -> str:
    """Anteil als Prozentzahl mit Komma und Leerzeichen vor dem Prozentzeichen."""
    return f"{de(100.0 * x, nd)} %"


def fmt_x(x: float) -> str:
    """Eine Schwelle: drei Stellen, bei Beträgen ab 1 000 ohne Nachkommastellen."""
    return thousands(x) if abs(x) >= 1000 else de(x, 3)


def fmt_small(x: float) -> str:
    """Ein Anteil nahe 0 oder 1: wissenschaftlich, wenn unter 10^-3."""
    if x == 0.0:
        return "0"
    if abs(x) < 1e-3:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant.replace('.', ',')}·10^{int(exp)}"
    return de(x, 4)


def players_text(players: int) -> str:
    return "Kontinuum" if players == 0 else f"N = {players}"


# --- Hauptansicht -------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EquilibriumRow:
    x: float                   # Schwelle x*
    level: float | None        # Modell R: kritischer Zustand θ̂; Modell L: Anteil m bei Indifferenz; endliches N: None
    stable: bool
    label: str


@dataclass(frozen=True)
class MainView:
    settings: S.Settings
    rows: tuple[EquilibriumRow, ...]
    continuum_count: int               # Zahl im Kontinuum (bei endlichem N zum Vergleich)
    formula_ok: bool                   # β ≥ α²/(2π)
    formula_bound: float               # α²/(2π)
    critical_beta: float | None        # Modell L: kritisches β bei y = ½ (geschlossen)
    c_band: tuple[float, float] | None  # Modell R: Bereich des Aufwands c mit drei Gleichgewichten (Kontinuum)
    p_unit: float                      # P(θ ∈ (0, 1)): Zustände, in denen θ bekannt zwei Gleichgewichte hätte
    underflow: bool
    scan_points: int | None            # endliches N: Rasterpunkte der Zählung

    @property
    def count(self) -> int:
        return len(self.rows)


def _labels(n: int, stable: list[bool]) -> list[str]:
    if n == 1:
        return ["einziges Gleichgewicht"]
    if n == 3:
        return ["kaum einer macht mit", "mittleres (instabil)", "fast alle machen mit"]
    return [("stabil" if s else "instabil") for s in stable]


def analyse(s: S.Settings, finite: F.FiniteEquilibria | None = None) -> MainView:
    """Gleichgewichte und Kennzahlen zu den Einstellungen. Bei endlichem N rechnet `finite` (teuer, von der App gecacht); fehlt es, wird hier gerechnet."""
    bound = GRID.formula_bound(s.alpha)
    p_unit = R.p_known_state_in_unit(s.alpha, s.y)
    if not s.is_regime:
        eq = L.roots_linear(s.alpha, s.beta, s.y)
        labels = _labels(eq.count, list(eq.stable))
        rows = tuple(EquilibriumRow(x, m, st, lab) for x, m, st, lab in zip(eq.x, eq.m, eq.stable, labels))
        return MainView(s, rows, eq.count, L.formula_holds(s.alpha, s.beta), bound, L.critical_beta(s.alpha), None, p_unit, eq.underflow, None)
    cont = R.roots_regime(s.alpha, s.beta, s.y, s.c)
    band = R.c_band(s.alpha, s.beta, s.y)
    if s.finite:
        fin = finite if finite is not None else F.roots_finite(s.players, s.alpha, s.beta, s.y, s.c)
        labels = _labels(fin.count, list(fin.stable))
        rows = tuple(EquilibriumRow(x, None, st, lab) for x, st, lab in zip(fin.x, fin.stable, labels))
        return MainView(s, rows, cont.count, R.formula_holds(s.alpha, s.beta), bound, None, band, p_unit, False, fin.scan_points)
    labels = _labels(cont.count, list(cont.stable))
    rows = tuple(EquilibriumRow(x, th, st, lab) for x, th, st, lab in zip(cont.x, cont.theta, cont.stable, labels))
    return MainView(s, rows, cont.count, R.formula_holds(s.alpha, s.beta), bound, None, band, p_unit, False, None)


def verdict_count(view: MainView) -> tuple[str, str]:
    """Die Zahl der Gleichgewichte vor jedem Verweis auf die Formel."""
    s = view.settings
    who = "" if not s.finite else f"Bei {players_text(s.players)} (Kontinuum: {view.continuum_count} Gleichgewicht{'e' if view.continuum_count != 1 else ''}): "
    level_name = "kritischer Zustand θ̂" if s.is_regime else "Anteil m bei Indifferenz"
    if view.count == 1:
        row = view.rows[0]
        lv = "" if row.level is None else f", {level_name} = {de(row.level, 4)}"
        tail = ""
        if s.is_regime and not s.finite:
            tail = f" Das Projekt startet für alle Zustände θ ≤ {de(row.level, 3)}."
        kind, head = "success", f"{who}**Ein einziges Gleichgewicht** (Schwelle x* = {fmt_x(row.x)}{lv})."
        return kind, head + tail + " Die Wahl ist eindeutig" + ("" if s.finite else ", und die Beste-Antwort-Iteration (Modellannahme) führt von „kaum einer“ und von „fast alle“ dorthin") + "."
    if view.count == 3:
        lo, mid, hi = view.rows
        return "warning", (f"{who}**Drei Gleichgewichte** mit den Schwellen x* = {fmt_x(lo.x)} (kaum einer macht mit), {fmt_x(mid.x)} (mittleres) und {fmt_x(hi.x)} (fast alle machen mit). "
                           f"Das mittlere ist instabil: schon eine kleine Abweichung wächst, die äußeren beiden sind stabil." + (" Wurzeln unter 10^-300 sind als 0 bzw. 1 gemeldet." if view.underflow else ""))
    stable = sum(r.stable for r in view.rows)
    return "info", f"{who}**{view.count} Gleichgewichte** gezählt (davon {stable} stabil), Schwellen x* = " + ", ".join(fmt_x(r.x) for r in view.rows) + f" (Raster mit {view.scan_points} Punkten)."


def verdict_formula(view: MainView) -> tuple[str, str]:
    """Die Faustformel β ≥ α²/(2π) neben der gezählten Zahl: im Modell R genau die Eindeutigkeitsgrenze über alle (y, c), im Modell L nur hinreichend."""
    s = view.settings
    head = f"β = {de(s.beta, 2)} {'≥' if view.formula_ok else '<'} α²/(2π) = {de(view.formula_bound, 3)}"
    if s.is_regime:
        if view.formula_ok:
            return "success", f"{head}: Im Modell R ist das genau die Bedingung, unter der es für jedes Prior-Mittel y und jeden Aufwand c nur ein Gleichgewicht gibt (Kontinuum)."
        if view.c_band is None:
            return "info", f"{head}. Kein Band mit drei Gleichgewichten."
        lo, hi = view.c_band
        inside = "liegt darin" if lo < s.c < hi else "liegt außerhalb"
        return "warning", (f"{head}: Für manche (y, c) gibt es drei Gleichgewichte. Bei y = {de(s.y, 2)} und diesen Genauigkeiten liegen sie für c zwischen {de(lo, 4)} und {de(hi, 4)}; "
                           f"c = {de(s.c, 2)} {inside} (Kontinuum: {view.continuum_count} Gleichgewicht{'e' if view.continuum_count != 1 else ''}). Für festes (y, c) ist die Formel nur hinreichend.")
    ratio = view.critical_beta / view.formula_bound
    crit = f"Die wahre Grenze bei y = ½ liegt bei β = {de(view.critical_beta, 3)}, das ist das {de(ratio, 2)}-Fache der Formel"
    if view.formula_ok:
        return "success", f"{head}: hinreichend für nur ein Gleichgewicht bei jedem y. {crit}; die Formel ist nicht notwendig."
    return "warning" if view.count > 1 else "info", f"{head}: die Formel sagt nichts. {crit}; hier gezählt: {view.count} Gleichgewicht{'e' if view.count != 1 else ''}."


def verdict_no_uncertainty(view: MainView) -> tuple[str, str]:
    """Ohne Unsicherheit (θ bekannt): zwei Gleichgewichte, wenn θ in (0, 1) liegt; die Lücke ist die Wahrscheinlichkeit dafür."""
    s = view.settings
    return "info", (f"Ohne Unsicherheit (θ allen bekannt) gäbe es für jedes θ in (0, 1) die zwei Gleichgewichte „alle machen mit“ und „keiner macht mit“. "
                    f"Bei diesem Prior (Mittel {de(s.y, 2)}, Streuung {de(1 / math.sqrt(s.alpha), 3)}) liegt θ mit Wahrscheinlichkeit {pct(view.p_unit)} in (0, 1).")


def verdict_round(view: MainView, rnd: S.Round) -> tuple[str, str]:
    """Die gezeichnete Spielrunde in Worten."""
    s = view.settings
    base = f"Gezogener Zustand θ = {de(rnd.theta, 3)}. {rnd.joined} von {rnd.n} gezeichneten Betrieben machen mit (Signal ≤ {fmt_x(rnd.x_star)})"
    if s.is_regime:
        how = "exakter Anteil" if rnd.exact else "gezählter Anteil"
        if rnd.started:
            return "success", f"{base}; {how} {pct(rnd.share)} ≥ θ: **das Projekt startet**, ein Mitmachender bekommt {de(rnd.payoff_join, 2)}."
        return "error", f"{base}; {how} {pct(rnd.share)} < θ: **das Projekt startet nicht**, ein Mitmachender bekommt {de(rnd.payoff_join, 2)}."
    return "info", f"{base}; exakter Anteil der Mitmachenden {pct(rnd.share)}, ein Mitmachender bekommt Anteil − θ = {de(rnd.payoff_join, 3)}."


def verdict_model_note(settings: S.Settings) -> tuple[str, str] | None:
    """Hinweis auf Regler, die im gewählten Modell nichts bewirken (ausgegraut): Modell L kennt weder Aufwand c noch endliches N."""
    if settings.is_regime:
        return None
    return "info", ("Im Modell L bringt Mitmachen (Anteil der anderen − θ): es gibt keinen Aufwand c, und der Erwartungsnutzen ist für jede Spielerzahl N gleich dem Kontinuumswert "
                    "(E[K/(N−1) | θ] = Φ(√β(x − θ)) exakt). Die Regler c und N sind daher ausgegraut.")


# --- Verdikte der Experimente --------------------------------------------------------------------------------------------------------------------
def verdict_e1(summary: dict, without_multi_regime: int, without_multi_linear: int, alphas: int, betas: int) -> tuple[str, str]:
    r, l = summary["regime"], summary["linear"]
    return "info", (
        f"Raster {alphas} × {betas} Zellen (α, β), je Zelle 25 Werte (y, c) im Modell R und 9 Werte y im Modell L. "
        f"**Modell R:** oberhalb der Formel haben {thousands(r['above']['multi'])} von {thousands(r['above']['cells'])} Zellen mehrere Gleichgewichte, unterhalb {thousands(r['below']['multi'])} von {thousands(r['below']['cells'])} ({pct(r['below_share'])}); "
        f"{without_multi_regime} (α, β)-Zellen unterhalb haben für kein (y, c) mehrere, die Formel ist also die Eindeutigkeitsgrenze über alle (y, c). "
        f"**Modell L:** oberhalb {thousands(l['above']['multi'])} von {thousands(l['above']['cells'])}, unterhalb {thousands(l['below']['multi'])} von {thousands(l['below']['cells'])} ({pct(l['below_share'])}); "
        f"{without_multi_linear} (α, β)-Zellen unterhalb der Formel sind für alle neun y eindeutig: hinreichend, nicht notwendig. Der Prozentanteil hängt vom gewählten (y, c)-Raster ab.")


def verdict_e2_regime(alpha: float, beta: float, y: float) -> tuple[str, str]:
    band = R.c_band(alpha, beta, y)
    rho = R.rho_of(alpha, beta)
    if band is None:
        return "info", f"Modell R mit α = {de(alpha, 1)}, β = {de(beta, 1)}: ρ = α/√β = {de(rho, 3)} ≤ √(2π) = {de(math.sqrt(2 * math.pi), 3)}, es gibt für kein c mehrere Gleichgewichte."
    lo, hi = band
    sym = " Bei y = ½ ist das Band symmetrisch um ½." if abs(y - 0.5) < 1e-12 else ""
    return "info", (f"Modell R mit α = {de(alpha, 1)}, β = {de(beta, 1)}, y = {de(y, 2)}: ρ = {de(rho, 3)} > √(2π); drei Gleichgewichte für c zwischen {de(lo, 4)} und {de(hi, 4)} (Breite {de(hi - lo, 4)}).{sym} "
                    f"Ein anderes Prior-Mittel y verschiebt das Band: je größer y, desto niedriger liegt es.")


def verdict_e2_linear(alpha: float, beta: float) -> tuple[str, str]:
    kap = L.kappa(alpha, beta)
    top = L.fold_edge(kap)
    crit = L.critical_beta(alpha)
    if top is None:
        return "info", f"Modell L mit α = {de(alpha, 1)}, β = {de(beta, 1)}: κ = {de(kap, 3)} ≤ √(2π), für kein Prior-Mittel y mehrere Gleichgewichte (kritisch bei y = ½: β = {de(crit, 3)})."
    return "info", (f"Modell L mit α = {de(alpha, 1)}, β = {de(beta, 1)}: κ = {de(kap, 3)} > √(2π); drei Gleichgewichte für y zwischen {de(1 - top, 4)} und {de(top, 4)}. "
                    f"Am Rand des Priors (y außerhalb) gibt es genau eines. Kritisch bei y = ½: β = {de(crit, 3)}.")


def verdict_e3(rows: list[dict], alpha: float, y: float, c: float) -> tuple[str, str]:
    unique = [r for r in rows if r["count"] == 1]
    if not unique:
        return "info", "Bei diesen Einstellungen hat keine der gerechneten Signalgenauigkeiten ein einziges Gleichgewicht; der kritische Zustand ist nicht eindeutig."
    last, first = unique[-1], unique[0]
    gap = abs(last["thetas"][0] - last["limit"])
    lost = f" Verlorene Projekte P(θ̂ < θ < 1): {pct(first['p_lost'])} bei β = {thousands(first['beta'])} und {pct(last['p_lost'])} bei β = {thousands(last['beta'])}." if first["p_lost"] is not None and last["p_lost"] is not None else ""
    multi = [r for r in rows if r["count"] != 1]
    note = f" Bei β = {', '.join(thousands(r['beta']) for r in multi)} gibt es mehrere Gleichgewichte (kein einzelnes θ̂)." if multi else ""
    near = "sehr nahe" if gap < 0.01 else "noch nicht nahe"
    if abs(c - 0.5) < 1e-12:
        effect = f" Bei c = ½ ändert das Rauschen den kritischen Zustand nicht: θ̂ liegt bei β = {thousands(first['beta'])} und bei β = {thousands(last['beta'])} bei {de(first['thetas'][0], 4)} und {de(last['thetas'][0], 4)}."
    else:
        effect = f" Bei β = {thousands(first['beta'])} liegt θ̂ bei {de(first['thetas'][0], 4)}, Abstand {de(abs(first['thetas'][0] - first['limit']), 4)} vom Grenzwert."
    return "info", (f"Grenzwert ohne Rauschen im Signal: θ̂ → 1 − c = {de(1 - c, 3)}. Bei β = {thousands(last['beta'])} liegt θ̂ bei {de(last['thetas'][0], 4)}, Abstand {de(gap, 4)} ({near} am Grenzwert).{effect}{lost}{note}")


def verdict_e4(rows: list[dict], alpha: float, beta: float, c: float) -> tuple[str, str]:
    cont = rows[0]
    parts = ", ".join(f"N = {r['n']}: {r['count']}" for r in rows[1:])
    changed = [r["n"] for r in rows[1:] if r["count"] != cont["count"]]
    if changed:
        what = f"Gegenüber dem Kontinuum ({cont['count']}) ändert sich die Zahl bei N = {', '.join(str(n) for n in changed)}."
    else:
        what = f"Die Zahl bleibt bei allen gerechneten N wie im Kontinuum ({cont['count']})."
    return "info", (f"Zahl der Gleichgewichte nach Spielerzahl (α = {de(alpha, 1)}, β = {de(beta, 1)}, c = {de(c, 2)}): {parts}. {what} "
                    f"Gezählt auf einem Raster mit 81 bis 241 Punkten im Suchfenster: ein Paar dicht an einem Berührpunkt könnte fehlen. Im Modell L ändert N nichts.")


def verdict_e5(rows: list[dict], alpha: float) -> tuple[str, str]:
    below = [r for r in rows if r["count"] == 3]
    above = [r for r in rows if r["count"] == 1]
    if not below or not above:
        return "info", "Die Faktoren liegen alle auf einer Seite der Grenze."
    b0, b1 = below[0], below[-1]
    a0, a1 = above[0], above[-1]
    sym = all(r["low"].rounds == r["high"].rounds for r in rows)
    capped = [r for r in rows if not (r["low"].reached and r["high"].reached)]
    cap_note = f" Bei {len(capped)} Faktoren wurde die Obergrenze von {thousands(C.E5_CAP)} Runden erreicht." if capped else ""
    sym_note = " Von beiden Enden gleich viele Runden: Spiegelsymmetrie bei y = c = ½." if sym else ""
    return "info", (f"Runden bis auf 10^-6 an die Wurzel (α = {de(alpha, 1)}, y = c = ½, Grenze β = {de(rows[0]['beta'] / rows[0]['factor'], 3)}): auf der Seite mit drei Gleichgewichten von {thousands(b0['low'].rounds)} (Faktor {de(b0['factor'], 2)}) "
                    f"auf {thousands(b1['low'].rounds)} (Faktor {de(b1['factor'], 2)}), auf der Seite mit einem von {thousands(a0['low'].rounds)} (Faktor {de(a0['factor'], 2)}) auf {thousands(a1['low'].rounds)} (Faktor {de(a1['factor'], 2)}). "
                    f"In diesen Zeilen verlangsamt sich die Iteration an der Grenze zwischen „eindeutig“ und „mehrfach“.{sym_note}{cap_note}")


def verdict_e5_h11(alpha: float, beta: float, y: float, c: float) -> tuple[str, str]:
    """H11: Stabilität der Wurzeln aus der Steigung B' der Antwortfunktion (Modell R, Kontinuum)."""
    e = R.roots_regime(alpha, beta, y, c)
    slopes = [I.best_reply_slope_regime(w, alpha, beta) for w in e.w]
    if e.count == 1:
        return "info", f"Ein Gleichgewicht: die Steigung der Antwortfunktion ist dort B' = {de(slopes[0], 3)} < 1, die Wurzel ist stabil."
    parts = ", ".join(f"B' = {de(sl, 3)} ({'stabil' if st else 'instabil'})" for sl, st in zip(slopes, e.stable))
    return "info", f"Drei Gleichgewichte, Steigung der Antwortfunktion an den Wurzeln: {parts}. B' > 1 heißt: eine kleine Abweichung wächst; die mittlere Wurzel ist instabil, die äußeren sind es nicht."


def verdict_e6(sm: dict, eps: float, loss: float) -> tuple[str, str]:
    eq = sm["eq"]
    q, qs = sm["q"], sm["q_star"]
    head = f"q = L/(L+M) = {de(q, 4)}, Grenze (1−ε)/(2−ε) = {de(qs, 4)}."
    if eq.attacks:
        return "success", (f"{head} Das größte Gleichgewicht enthält einen Angriff: Spieler 1 greift ab Typ {eq.first1} an, Spieler 2 ab Typ {eq.first2}. "
                           f"G ist in {thousands(sm['common_states'])} von {thousands(sm['g_states'])} Zuständen gemeinsame q-Überzeugung.")
    return "warning", (f"{head} Im größten Gleichgewicht greift **nie jemand an**, obwohl bis zu {thousands(sm['g_states'])} Nachrichten ankommen könnten: Die Kette löst sich in {thousands(eq.rounds)} Runden Stufe für Stufe auf. "
                       f"G ist in {thousands(sm['common_states'])} von {thousands(sm['g_states'])} Zuständen gemeinsame q-Überzeugung, der Operator braucht {thousands(sm['common_rounds'])} Runden (eine je Nachrichtenstufe).")


def verdict_e6_knowledge(rows: list[dict]) -> tuple[str, str]:
    worst = max(abs(r["simulated"] - r["formula"]) / r["se"] for r in rows)
    ok = worst <= 4.0
    body = "; ".join(f"n = {r['n']}: {de(r['formula'], 4)} gegen {de(r['simulated'], 4)}" for r in rows)
    return ("success" if ok else "warning"), (f"Gegenseitiges Wissen der Stufe n, Formel (1−ε)^n gegen {thousands(rows[0]['games_in_g'])} simulierte Spiele im Zustand G: {body}. "
                                              f"Größte Abweichung {de(worst, 1)} Standardfehler ({'innerhalb' if ok else 'außerhalb'} der Marge von 4).")
