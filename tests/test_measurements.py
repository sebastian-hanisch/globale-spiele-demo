"""Das vorgerechnete Raster (data/grid.json) gehört zum aktuellen Rechencode und stimmt mit dem geschlossenen Band überein.

Eigene Datei, weil jede Änderung an glo_grid.py, glo_regime.py, glo_linear.py oder glo_core.py den Erzeuger-Hash ändert: so bleibt die Meldung „Messdatei neu erzeugen“ von den Tests der Rechnung getrennt."""

import json

import pytest

import glo_experiments as X
import glo_grid as GRID
import glo_linear as L
import glo_regime as R
from glo_core import SQ2PI


def stored():
    return json.loads(X.GRID_FILE.read_text(encoding="utf-8"))


def test_file_matches_the_generator_hash():
    assert stored()["generator_hash"] == X.generator_hash(), "glo_grid/regime/linear/core wurde geändert: python tools/make_grid.py ausführen"
    assert X.load_grid() is not None


def test_load_returns_none_for_a_stale_hash(monkeypatch):
    monkeypatch.setattr(X, "generator_hash", lambda: "0000000000000000")
    assert X.load_grid() is None


def test_load_returns_none_when_the_file_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(X, "GRID_FILE", tmp_path / "fehlt.json")
    assert X.load_grid() is None


def test_axes_and_shape_match_the_code():
    g = stored()
    assert g["alphas"] == list(GRID.ALPHAS) and g["betas"] == list(GRID.BETAS)
    for model, cells in (("regime", 25), ("linear", 9)):
        assert g[model]["cells"] == cells and len(g[model]["multi"]) == len(GRID.ALPHAS) and all(len(row) == len(GRID.BETAS) for row in g[model]["multi"])
        assert all(0 <= v <= cells for row in g[model]["multi"] for v in row)


def test_axes_cover_the_slider_ranges():
    import glo_constants as C
    assert GRID.ALPHAS[0] == C.ALPHA_MIN and GRID.ALPHAS[-1] == C.ALPHA_MAX and GRID.BETAS[0] == C.BETA_MIN and GRID.BETAS[-1] == C.BETA_MAX


def test_stored_counts_equal_the_closed_form_counts_in_every_cell_with_margin():
    """Wurzelsuche (gespeichert) gegen geschlossenes Band/Faltenkurve in allen Zellen. Abstand zu den Berührpunkten: jede Zelle liegt mindestens 10^-5 von der Kante entfernt,
    sonst könnte die Zuordnung zwischen Windows und Linux kippen."""
    g = stored()
    min_margin_r = min_margin_l = 1.0
    for i, a in enumerate(GRID.ALPHAS):
        for j, b in enumerate(GRID.BETAS):
            band = R.d_band(a, b)
            n_r = 0
            for y in GRID.R_YS:
                for c in GRID.R_CS:
                    d = R.d_value(a, b, y, c)
                    if band is not None:
                        min_margin_r = min(min_margin_r, abs(d - band[0]), abs(d - band[1]))
                    n_r += R.count_closed(a, b, y, c) > 1
            assert g["regime"]["multi"][i][j] == n_r, (a, b)
            kap = L.kappa(a, b)
            top = L.fold_edge(kap)
            n_l = 0
            for y in GRID.L_YS:
                if top is not None:
                    min_margin_l = min(min_margin_l, abs(y - top), abs(y - (1 - top)))
                min_margin_l = min(min_margin_l, abs(kap - SQ2PI))
                n_l += L.count_closed(a, b, y) > 1
            assert g["linear"]["multi"][i][j] == n_l, (a, b)
    assert min_margin_r > 1e-5 and min_margin_l > 1e-5, (min_margin_r, min_margin_l)


def test_above_the_formula_no_cell_has_several_equilibria_and_below_every_regime_cell_has_some():
    g = stored()
    for model in ("regime", "linear"):
        for i, a in enumerate(GRID.ALPHAS):
            for j, b in enumerate(GRID.BETAS):
                if b >= GRID.formula_bound(a):
                    assert g[model]["multi"][i][j] == 0, (model, a, b)
    assert X.grid_below_formula_without_multi(g, "regime") == 0          # bei y = c = ½ gibt es unterhalb der Formel immer drei
    assert X.grid_below_formula_without_multi(g, "linear") > 0           # im Modell L ist die Formel nicht notwendig


def test_a_few_cells_are_reproduced_by_the_root_search():
    g = stored()
    for i, j in ((0, 0), (14, 10), (28, 30), (20, 5)):
        a, b = GRID.ALPHAS[i], GRID.BETAS[j]
        assert g["regime"]["multi"][i][j] == GRID.multi_cells_regime(a, b)
        assert g["linear"]["multi"][i][j] == GRID.multi_cells_linear(a, b)


def test_summary_and_helper_on_a_tiny_hand_grid():
    tiny = {"alphas": [2.0, 4.0], "betas": [0.1, 10.0], "regime": {"cells": 25, "multi": [[10, 0], [25, 0]]}, "linear": {"cells": 9, "multi": [[0, 0], [3, 0]]}}
    s = X.grid_summary(tiny)
    # α = 2: Grenze 0,637 → β = 0,1 darunter, β = 10 darüber; α = 4: Grenze 2,546 → ebenso
    assert s["regime"]["below"] == {"cells": 50, "multi": 35} and s["regime"]["above"] == {"cells": 50, "multi": 0} and s["regime"]["below_share"] == pytest.approx(0.7)
    assert s["linear"]["below"] == {"cells": 18, "multi": 3} and s["linear"]["total"] == 36
    assert X.grid_below_formula_without_multi(tiny, "linear") == 1 and X.grid_below_formula_without_multi(tiny, "regime") == 0
