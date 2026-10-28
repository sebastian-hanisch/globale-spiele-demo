"""Plotly-Figuren (alle Achsen fest, fixedrange). Nur Darstellung: jede Zahl kommt fertig aus Kern und Auswertung."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go

import glo_constants as C
import glo_grid as GRID
import glo_linear as L
import glo_regime as R
from glo_core import ncdf
from glo_scenario import Round, Settings


def _base(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(title=title, height=height, margin=dict(l=55, r=15, t=45 if title else 15, b=55), legend=dict(orientation="h", y=-0.28, x=0))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _root_markers(fig: go.Figure, xs, ys, stable, names) -> None:
    for st, color, label in ((True, C.COLOR_STABLE, "stabil"), (False, C.COLOR_UNSTABLE, "instabil")):
        idx = [i for i, s in enumerate(stable) if s == st]
        if idx:
            fig.add_trace(go.Scatter(x=[xs[i] for i in idx], y=[ys[i] for i in idx], mode="markers", name=f"Gleichgewicht ({label})", text=[names[i] for i in idx], hoverinfo="text",
                                     marker=dict(size=13, color=color, line=dict(width=1.5, color="white"))))


# --- Hauptansicht ---------------------------------------------------------------------------------------------------------------------------
def build_regime_curve(s: Settings, eq: R.RegimeEquilibria) -> go.Figure:
    """Modell R: Wie sicher ist der indifferente Betrieb, dass das Projekt startet, wenn alle anderen auf den kritischen Zustand θ̂ setzen? Gleichgewicht, wo es genau c ist."""
    w = np.linspace(-4.5, 4.5, 721)
    theta = ncdf(w)
    prob = ncdf((s.alpha * (theta - s.y) - math.sqrt(s.beta) * w) / math.sqrt(s.alpha + s.beta))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=theta, y=prob, mode="lines", name="Sicherheit am Indifferenzsignal", line=dict(color=C.COLOR_LINE, width=2.5)))
    fig.add_hline(y=s.c, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text=f"Aufwand c = {s.c:.2f}".replace(".", ","), annotation_position="top left")
    _root_markers(fig, list(eq.theta), [s.c] * eq.count, list(eq.stable), [f"θ̂ = {t:.4f}".replace(".", ",") for t in eq.theta])
    fig.update_xaxes(title="Kritischer Zustand θ̂", range=[0, 1])
    fig.update_yaxes(title="P(Start | Signal x*(θ̂))", range=[0, 1.02])
    return _base(fig)


def build_linear_curve(s: Settings, eq: L.LinearEquilibria) -> go.Figure:
    """Modell L: der Anteil Mitmachender, den der indifferente Betrieb erwartet, gegen den Anteil m; Gleichgewicht, wo die Kurve die Diagonale schneidet."""
    m = np.linspace(0, 1, 801)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=ncdf(eq.kappa * (m - s.y)), mode="lines", name="Φ(κ(m − y))", line=dict(color=C.COLOR_LINE, width=2.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Diagonale", line=dict(color=C.COLOR_GREY, dash="dash")))
    _root_markers(fig, list(eq.m), list(eq.m), list(eq.stable), [f"m = {v:.4g}".replace(".", ",") for v in eq.m])
    fig.update_xaxes(title="Anteil m bei Indifferenz", range=[0, 1])
    fig.update_yaxes(title="erwarteter Anteil Mitmachender", range=[0, 1.02])
    return _base(fig)


def build_round(rnd: Round) -> go.Figure:
    """Die gezeichnete Spielrunde: Signale der Betriebe, Schwelle x* und der wahre Zustand θ."""
    order = np.argsort(rnd.signals)
    sig, joins = rnd.signals[order], rnd.joins[order]
    fig = go.Figure()
    if rnd.n <= C.DRAW_MAX:
        idx = np.arange(rnd.n)
        for flag, color, name in ((True, C.COLOR_STABLE, "macht mit"), (False, C.COLOR_GREY, "macht nicht mit")):
            sel = joins == flag
            fig.add_trace(go.Scatter(x=sig[sel], y=idx[sel], mode="markers", name=name, marker=dict(size=8, color=color), hovertemplate="Signal %{x:.3f}<extra></extra>"))
        fig.update_yaxes(title="Betrieb (nach Signal geordnet)", range=[-1, rnd.n])
    else:
        lo, hi = float(sig.min()), float(sig.max())
        bins = dict(start=lo, end=hi, size=(hi - lo) / 40)
        fig.add_trace(go.Histogram(x=sig[joins], name="macht mit", marker_color=C.COLOR_STABLE, xbins=bins))
        fig.add_trace(go.Histogram(x=sig[~joins], name="macht nicht mit", marker_color=C.COLOR_GREY, xbins=bins))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title="Zahl der Betriebe")
    fig.add_vline(x=rnd.x_star, line=dict(color=C.COLOR_UNSTABLE, dash="dash"), annotation_text="Schwelle x*", annotation_position="top")
    fig.add_vline(x=rnd.theta, line=dict(color=C.COLOR_RED), annotation_text="Zustand θ", annotation_position="bottom")
    fig.update_xaxes(title="privates Signal")
    return _base(fig, height=320)


# --- E1 -------------------------------------------------------------------------------------------------------------------------------------
def build_map(grid: dict, model: str, s: Settings) -> go.Figure:
    """Karte (β, α): Anteil der (y, c)- bzw. y-Werte des Rasters mit drei Gleichgewichten; Linien: Faustformel β = α²/(2π) und (Modell L) die wahre Grenze bei y = ½."""
    key = "regime" if model == C.MODEL_R else "linear"
    cells = grid[key]["cells"]
    z = np.array(grid[key]["multi"], dtype=float) / cells
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=grid["betas"], y=grid["alphas"], z=z, zmin=0, zmax=1, colorscale=[[0, "#f3f3f3"], [1, C.COLOR_UNSTABLE]], xgap=0, ygap=0,
                             colorbar=dict(title=dict(text="Anteil mit mehreren", side="top"), tickvals=[0, 0.5, 1], len=0.8),
                             hovertemplate="β %{x}, α %{y}: Anteil %{z:.2f}<extra></extra>"))
    a_line = np.geomspace(GRID.ALPHAS[0], GRID.ALPHAS[-1], 80)
    fig.add_trace(go.Scatter(x=[GRID.formula_bound(a) for a in a_line], y=a_line, mode="lines", name="Faustformel β = α²/(2π)", line=dict(color="black", dash="dash")))
    if model == C.MODEL_L:
        fig.add_trace(go.Scatter(x=[L.critical_beta(a) for a in a_line], y=a_line, mode="lines", name="wahre Grenze bei y = ½", line=dict(color=C.COLOR_STABLE, width=2.5)))
    fig.add_trace(go.Scatter(x=[s.beta], y=[s.alpha], mode="markers", name="Ihre Einstellung", marker=dict(size=13, color=C.COLOR_RED, symbol="x", line=dict(width=2, color=C.COLOR_RED))))
    fig.update_xaxes(title="Signal-Genauigkeit β", type="log", range=[math.log10(0.1), math.log10(100)])
    fig.update_yaxes(title="Prior-Genauigkeit α", type="log", range=[math.log10(0.5), math.log10(30)])
    return _base(fig, height=420)


# --- E2 -------------------------------------------------------------------------------------------------------------------------------------
def build_c_band(rows: list[dict], s: Settings) -> go.Figure:
    ys = [r["y"] for r in rows]
    fig = go.Figure()
    if all(r["lo"] is not None for r in rows):
        fig.add_trace(go.Scatter(x=ys, y=[r["hi"] for r in rows], mode="lines", name="obere Kante", line=dict(color=C.COLOR_UNSTABLE)))
        fig.add_trace(go.Scatter(x=ys, y=[r["lo"] for r in rows], mode="lines", name="drei Gleichgewichte dazwischen", line=dict(color=C.COLOR_UNSTABLE), fill="tonexty", fillcolor="rgba(192,118,43,0.25)"))
    fig.add_trace(go.Scatter(x=[s.y], y=[s.c], mode="markers", name="Ihre Einstellung", marker=dict(size=13, color=C.COLOR_RED, symbol="x", line=dict(width=2, color=C.COLOR_RED))))
    fig.update_xaxes(title="Prior-Mittel y", range=[C.Y_MIN - 0.02, C.Y_MAX + 0.02])
    fig.update_yaxes(title="Aufwand c", range=[0, 1])
    return _base(fig, "Modell R: Band im Aufwand c")


def build_fold(rows: list[dict], s: Settings) -> go.Figure:
    betas = [r["beta"] for r in rows]
    fig = go.Figure()
    shown = [r for r in rows if r["hi"] is not None]
    if shown:
        fig.add_trace(go.Scatter(x=[r["beta"] for r in shown], y=[r["hi"] for r in shown], mode="lines", name="obere Kante Y(κ)", line=dict(color=C.COLOR_UNSTABLE)))
        fig.add_trace(go.Scatter(x=[r["beta"] for r in shown], y=[r["lo"] for r in shown], mode="lines", name="drei Gleichgewichte dazwischen", line=dict(color=C.COLOR_UNSTABLE), fill="tonexty", fillcolor="rgba(192,118,43,0.25)"))
    fig.add_trace(go.Scatter(x=[s.beta], y=[s.y], mode="markers", name="Ihre Einstellung", marker=dict(size=13, color=C.COLOR_RED, symbol="x", line=dict(width=2, color=C.COLOR_RED))))
    fig.update_xaxes(title="Signal-Genauigkeit β", type="log", range=[math.log10(min(betas)), math.log10(max(betas))])
    fig.update_yaxes(title="Prior-Mittel y", range=[0, 1])
    return _base(fig, f"Modell L: Band im Prior y (α = {s.alpha:g})".replace(".", ","))


# --- E3 -------------------------------------------------------------------------------------------------------------------------------------
def build_theta_over_beta(rows: list[dict], c: float) -> go.Figure:
    fig = go.Figure()
    xs = [r["beta"] for r in rows for _ in r["thetas"]]
    ys = [t for r in rows for t in r["thetas"]]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name="kritischer Zustand θ̂ (jede Wurzel)", line=dict(color=C.COLOR_LINE), marker=dict(size=8)))
    fig.add_hline(y=1 - c, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text=f"Grenzwert 1 − c = {1 - c:.2f}".replace(".", ","), annotation_position="top left")
    fig.update_xaxes(title="Signal-Genauigkeit β", type="log")
    fig.update_yaxes(title="θ̂", range=[0, 1])
    return _base(fig)


# --- E4 -------------------------------------------------------------------------------------------------------------------------------------
def build_players(rows: list[dict]) -> go.Figure:
    labels = ["Kontinuum" if r["n"] == 0 else str(r["n"]) for r in rows]
    xs, ys, stable, names = [], [], [], []
    for lab, r in zip(labels, rows):
        for x, st in zip(r["x"], r["stable"]):
            xs.append(lab)
            ys.append(x)
            stable.append(st)
            names.append(f"{lab}: x* = {x:.4f}".replace(".", ","))
    fig = go.Figure()
    _root_markers(fig, xs, ys, stable, names)
    fig.update_xaxes(title="Zahl der Betriebe N", type="category", categoryorder="array", categoryarray=labels)
    fig.update_yaxes(title="Schwelle x*")
    return _base(fig)


# --- E5 -------------------------------------------------------------------------------------------------------------------------------------
def build_rounds(rows_regime: list[dict], rows_linear: list[dict] | None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[r["factor"] for r in rows_regime], y=[r["low"].rounds for r in rows_regime], mode="lines+markers", name="Modell R (y = c = ½)", line=dict(color=C.COLOR_LINE),
                             marker=dict(size=8, symbol=["circle" if r["low"].reached else "circle-open" for r in rows_regime])))
    if rows_linear:
        fig.add_trace(go.Scatter(x=[r["factor"] for r in rows_linear], y=[r["low"].rounds for r in rows_linear], mode="lines+markers", name="Modell L (y = ½)", line=dict(color=C.COLOR_STABLE),
                                 marker=dict(size=8, symbol=["circle" if r["low"].reached else "circle-open" for r in rows_linear])))
    fig.add_vline(x=1.0, line=dict(color="black", dash="dash"), annotation_text="Grenze", annotation_position="top")
    fig.update_xaxes(title="β als Vielfaches der Grenze (links: drei Gleichgewichte, rechts: eines)", type="log")
    fig.update_yaxes(title="Runden bis auf 10^-6", type="log")
    return _base(fig)


def build_trajectories(trajs: list[dict], roots: tuple[float, ...]) -> go.Figure:
    fig = go.Figure()
    palette = [C.COLOR_LINE, C.COLOR_STABLE, C.COLOR_UNSTABLE, C.COLOR_RED]
    for i, t in enumerate(trajs):
        fig.add_trace(go.Scatter(x=list(range(len(t["path"]))), y=t["path"], mode="lines", name=t["label"], line=dict(color=palette[i % len(palette)], width=2)))
    for x in roots:
        fig.add_hline(y=x, line=dict(color=C.COLOR_GREY, dash="dot"))
    lo, hi = min(roots) - 0.5, max(roots) + 0.5
    fig.update_xaxes(title="Runde")
    fig.update_yaxes(title="Schwelle x", range=[lo, hi])
    return _base(fig)


# --- E6 -------------------------------------------------------------------------------------------------------------------------------------
def build_email_beliefs(sm: dict) -> go.Figure:
    eq, q = sm["eq"], sm["q"]
    n = C.EMAIL_PLOT_TYPES
    t = np.arange(n + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=eq.belief1[: n + 1], mode="lines+markers", name="Spieler 1: Überzeugung „G und der andere greift an“", line=dict(color=C.COLOR_LINE), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=t, y=eq.belief2[: n + 1], mode="lines+markers", name="Spieler 2", line=dict(color=C.COLOR_STABLE), marker=dict(size=6)))
    fig.add_hline(y=q, line=dict(color=C.COLOR_RED, dash="dash"), annotation_text=f"q = {q:.3f}".replace(".", ","), annotation_position="top right")
    fig.update_xaxes(title=f"Typ (Zahl der gesendeten Nachrichten), die ersten {n}")
    fig.update_yaxes(title="bedingte Wahrscheinlichkeit", range=[0, 1.02])
    return _base(fig)
