"""Experimente E1 bis E6: Tabellen und Reihen aus dem Kern (live) und aus dem vorgerechneten Raster (data/grid.json).

Vorgerechnet ist nur das (α, β)-Raster der Gleichgewichtszahl (E1), mit Erzeuger-Hash (tools/make_grid.py). Alles andere wird live gerechnet (Wurzeln unter 1 ms, endliches N bis rund 4 s,
Beste-Antwort-Iteration, E-Mail-Spiel unter 0,1 s)."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path


import glo_constants as C
import glo_email as EM
import glo_finite as F
import glo_grid as GRID
import glo_iteration as I
import glo_linear as L
import glo_regime as R

ROOT = Path(__file__).resolve().parent
GRID_FILE = ROOT / "data" / "grid.json"
HASHED_FILES = ("glo_grid.py", "glo_regime.py", "glo_linear.py", "glo_core.py")


# --------------------------------------------------------------------------------- vorgerechnetes Raster
def generator_hash() -> str:
    """Hash der Dateien, aus deren Code das Raster stammt (Zeilenenden normalisiert, damit CRLF/LF denselben Hash geben)."""
    h = hashlib.sha256()
    for name in HASHED_FILES:
        text = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        h.update(name.encode("utf-8") + b"\0" + text + b"\0")
    return h.hexdigest()[:16]


def write_grid() -> dict:
    payload = {"generator_hash": generator_hash(), **GRID.compute()}
    GRID_FILE.parent.mkdir(exist_ok=True)
    GRID_FILE.write_text(json.dumps(payload, indent=None, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    return payload


def load_grid() -> dict | None:
    """Das Raster, oder None, wenn die Datei fehlt oder zu einem anderen Rechencode gehört (dann zeigt die App keine Karte)."""
    if not GRID_FILE.exists():
        return None
    data = json.loads(GRID_FILE.read_text(encoding="utf-8"))
    return data if data.get("generator_hash") == generator_hash() else None


def grid_summary(grid: dict) -> dict:
    """Zählungen über alle (α, β, y, c)- bzw. (α, β, y)-Zellen des Rasters: oberhalb und unterhalb der Faustformel β ≥ α²/(2π)."""
    out = {}
    for model in ("regime", "linear"):
        per = grid[model]["cells"]
        above = {"cells": 0, "multi": 0}
        below = {"cells": 0, "multi": 0}
        for i, a in enumerate(grid["alphas"]):
            for j, b in enumerate(grid["betas"]):
                bucket = above if b >= GRID.formula_bound(a) else below
                bucket["cells"] += per
                bucket["multi"] += grid[model]["multi"][i][j]
        out[model] = {"above": above, "below": below, "total": above["cells"] + below["cells"],
                      "below_share": below["multi"] / below["cells"] if below["cells"] else float("nan")}
    return out


def grid_below_formula_without_multi(grid: dict, model: str) -> int:
    """Zahl der (α, β)-Zellen unter der Faustformel, in denen KEIN (y, c) mehrere Gleichgewichte hat. Modell R: 0 (bei y = c = ½ gibt es unterhalb der Formel immer drei)."""
    return sum(1 for i, a in enumerate(grid["alphas"]) for j, b in enumerate(grid["betas"]) if b < GRID.formula_bound(a) and grid[model]["multi"][i][j] == 0)


# --------------------------------------------------------------------------------- E2
def e2_regime_band_rows(alpha: float, beta: float, ys: list[float]) -> list[dict]:
    """Band des Aufwands c mit drei Gleichgewichten, je Prior-Mittel y (Modell R)."""
    rows = []
    for y in ys:
        band = R.c_band(alpha, beta, y)
        rows.append({"y": y, "lo": None if band is None else band[0], "hi": None if band is None else band[1]})
    return rows


def e2_linear_fold_rows(alpha: float, betas: list[float]) -> list[dict]:
    """Obere Kante Y(κ) des Prior-Bereichs mit drei Gleichgewichten (Modell L), je β."""
    rows = []
    for b in betas:
        kap = L.kappa(alpha, b)
        top = L.fold_edge(kap)
        rows.append({"beta": b, "kappa": kap, "lo": None if top is None else 1.0 - top, "hi": top})
    return rows


# --------------------------------------------------------------------------------- E3
def e3_rows(alpha: float, y: float, c: float, betas=C.RAT_BETAS) -> list[dict]:
    """Kritischer Zustand θ̂ gegen die Signalgenauigkeit β (Modell R): Grenzwert 1 − c ohne Rauschen im Signal; verlorene Projekte P(θ̂ < θ < 1)."""
    rows = []
    for b in betas:
        e = R.roots_regime(alpha, b, y, c)
        row = {"beta": b, "count": e.count, "thetas": e.theta, "limit": 1.0 - c, "xs": e.x}
        if e.count == 1:
            row["p_start"] = R.p_start(alpha, y, e.theta[0])
            row["p_lost"] = R.p_lost(alpha, y, e.theta[0])
        else:
            row["p_start"] = row["p_lost"] = None
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------- E4
def e4_rows(alpha: float, beta: float, y: float, c: float, players=C.E4_PLAYERS) -> list[dict]:
    """Gleichgewichte bei N Betrieben (Modell R) gegen das Kontinuum."""
    cont = R.roots_regime(alpha, beta, y, c)
    rows = [{"n": 0, "count": cont.count, "x": cont.x, "stable": cont.stable, "seconds": 0.0}]
    for n in players:
        start = time.perf_counter()
        e = F.roots_finite(n, alpha, beta, y, c)
        rows.append({"n": n, "count": e.count, "x": e.x, "stable": e.stable, "seconds": time.perf_counter() - start})
    return rows


def finite_seconds_per_eval(n: int) -> float:
    """Grobe Zeit einer Auswertung der Startwahrscheinlichkeit (Entwicklungsrechner: N = 100 rund 1 ms, N = 250 rund 5 ms, N = 1 000 rund 40 ms), nur für die Zeitangabe an den Knöpfen."""
    return 5e-4 if n <= 10 else (1e-7 * n * n if n <= 250 else 4e-8 * n * n)


def finite_estimated_seconds(n: int) -> float:
    """Grobe Zeitschätzung der Wurzelsuche bei N Betrieben: Rasterpunkte plus rund 40 Auswertungen zum Verfeinern."""
    return finite_seconds_per_eval(n) * (F.scan_points_for(n) + 40)


def e4_estimated_seconds(players=C.E4_PLAYERS) -> float:
    """Grobe Zeitschätzung der E4-Tabelle für den Knopf."""
    return sum(finite_estimated_seconds(n) for n in players)


# --------------------------------------------------------------------------------- E5
def e5_rows(alpha: float, factors=C.E5_FACTORS, cap: int = C.E5_CAP) -> list[dict]:
    """Runden der Beste-Antwort-Iteration (Modell R, y = c = ½) bei β als Vielfaches der Grenze α²/(2π): von „kaum einer“ und von „fast alle“."""
    bound = GRID.formula_bound(alpha)
    rows = []
    for f in factors:
        b = bound * f
        e = R.roots_regime(alpha, b, 0.5, 0.5)
        lo = I.iterate_regime(alpha, b, 0.5, 0.5, I.START_LOW, e.x[0], cap=cap)
        hi = I.iterate_regime(alpha, b, 0.5, 0.5, I.START_HIGH, e.x[-1], cap=cap)
        rows.append({"factor": f, "beta": b, "count": e.count, "spread": e.x[-1] - e.x[0], "low": lo, "high": hi})
    return rows


def e5_linear_rows(alpha: float, factors=C.E5_FACTORS, cap: int = C.E5_CAP) -> list[dict]:
    """Dasselbe im Modell L (y = ½) mit β als Vielfaches des kritischen β(α)."""
    crit = L.critical_beta(alpha)
    rows = []
    for f in factors:
        b = crit * f
        e = L.roots_linear(alpha, b, 0.5)
        lo = I.iterate_linear(alpha, b, 0.5, 0.0, e.m[0], cap=cap)
        hi = I.iterate_linear(alpha, b, 0.5, 1.0, e.m[-1], cap=cap)
        rows.append({"factor": f, "beta": b, "count": e.count, "low": lo, "high": hi})
    return rows


def e5_estimated_seconds(alpha: float) -> float:
    """Grobe Zeitschätzung für den Knopf: Summe der Runden der Zeilen aus einem Näherungsansatz (Rundenzahl ∝ 1/|1 − Faktor|), geteilt durch ROUNDS_PER_SECOND."""
    rounds = sum(min(C.E5_CAP, 40.0 / abs(1.0 - f) ** 1.5 + 50) * 2 for f in C.E5_FACTORS)
    return rounds / C.ROUNDS_PER_SECOND + 0.5


def e5_trajectories(alpha: float, beta: float, y: float, c: float, rounds: int = 60) -> list[dict]:
    """Beste-Antwort-Folgen zu den Parametern der Seitenleiste: von weit links und weit rechts, bei drei Wurzeln zusätzlich knapp unter und über der mittleren Wurzel (H11)."""
    e = R.roots_regime(alpha, beta, y, c)
    starts = [("kaum einer (x₀ = −50)", I.START_LOW), ("fast alle (x₀ = +50)", I.START_HIGH)]
    if e.count == 3:
        starts += [("knapp unter der mittleren Wurzel", e.x[1] - 1e-3), ("knapp über der mittleren Wurzel", e.x[1] + 1e-3)]
    return [{"label": label, "path": I.trajectory_regime(alpha, beta, y, c, x0, rounds)} for label, x0 in starts]


# --------------------------------------------------------------------------------- E6
def e6_summary(eps: float, loss: float, p_g: float = C.EMAIL_P_G, j: int = C.EMAIL_J) -> dict:
    """Das größte Gleichgewicht des E-Mail-Spiels, seine Überzeugungen und die gemeinsame q-Überzeugung von G (Gewinn M = 1, Verlust L)."""
    game = EM.build(eps, p_g, j)
    q = EM.q_of(loss)
    eq = EM.greatest_equilibrium(game, q)
    common, common_rounds = EM.common_p_belief(game, q)
    return {"game": game, "q": q, "q_star": EM.q_star(eps), "threshold": EM.attack_threshold(eps, p_g), "eq": eq,
            "g_states": int(game.is_g[:-1].sum()), "common_states": int(common.sum()), "common_rounds": common_rounds}


def e6_knowledge_rows(eps: float, levels=(1, 2, 5, 10, 20), games: int = C.EMAIL_SIM_GAMES, seed: int = C.EMAIL_SIM_SEED, p_g: float = C.EMAIL_P_G) -> list[dict]:
    """Gegenseitiges Wissen der Stufe n: Wahrscheinlichkeit (1−ε)^n, dazu der Anteil im simulierten Protokoll (Nachricht für Nachricht)."""
    sim = EM.simulate_protocol(eps, p_g, games, seed)
    in_g = int(sim["is_g"].sum())
    rows = []
    for n in levels:
        p = EM.knowledge_level_prob(eps, n)
        hat = float((sim["k"] >= n + 1).sum() / in_g)
        rows.append({"n": n, "formula": p, "simulated": hat, "se": math.sqrt(p * (1 - p) / in_g), "games_in_g": in_g})
    return rows
