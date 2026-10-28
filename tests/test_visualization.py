"""Abbildungen und PDF: jede Figur baut ohne Fehler, hat feste Achsen und die erwarteten Spuren; das PDF enthält die Verdikte."""

import plotly.graph_objects as go

import glo_constants as C
import glo_evaluation as V
import glo_experiments as X
import glo_linear as L
import glo_pdf_export as PDF
import glo_regime as R
import glo_scenario as S
import glo_visualization as Z


def fixed(fig):
    return all(ax.fixedrange for ax in (fig.layout.xaxis, fig.layout.yaxis))


def test_curves_mark_every_equilibrium_with_its_stability():
    s = S.settings_of_preset("Ungenaue Signale (drei)")
    fig = Z.build_regime_curve(s, R.roots_regime(s.alpha, s.beta, s.y, s.c))
    marks = [t for t in fig.data if t.mode == "markers"]
    assert fixed(fig) and sorted(len(t.x) for t in marks) == [1, 2] and {t.name for t in marks} == {"Gleichgewicht (stabil)", "Gleichgewicht (instabil)"}
    sl = S.settings_of_preset("V3-Zelle (Modell L)")
    figl = Z.build_linear_curve(sl, L.roots_linear(sl.alpha, sl.beta, sl.y))
    assert fixed(figl) and sum(len(t.x) for t in figl.data if t.mode == "markers") == 3


def test_round_figure_uses_points_for_small_n_and_a_histogram_for_large_n():
    s = S.Settings(players=5)
    rnd = S.draw_round(s, 0.5, None)
    assert isinstance(Z.build_round(rnd).data[0], go.Scatter)
    big = S.draw_round(S.Settings(players=1000), 0.5, None)
    assert isinstance(Z.build_round(big).data[0], go.Histogram)


def test_map_has_the_formula_line_and_for_the_linear_model_the_true_boundary():
    grid = X.load_grid()
    s = S.Settings()
    assert [t.name for t in Z.build_map(grid, C.MODEL_R, s).data if getattr(t, "mode", None) == "lines"] == ["Faustformel β = α²/(2π)"]
    assert "wahre Grenze bei y = ½" in [t.name for t in Z.build_map(grid, C.MODEL_L, s).data]
    assert fixed(Z.build_map(grid, C.MODEL_R, s))


def test_band_fold_theta_players_rounds_trajectory_and_email_figures_build():
    s = S.Settings(alpha=8.0, beta=4.0)
    assert fixed(Z.build_c_band(X.e2_regime_band_rows(8.0, 4.0, [0.3, 0.5]), s))
    assert fixed(Z.build_c_band(X.e2_regime_band_rows(4.0, 4.0, [0.3, 0.5]), s))                 # ohne Band nur der Marker
    assert fixed(Z.build_fold(X.e2_linear_fold_rows(8.0, [0.1, 1.0, 100.0]), s))
    assert fixed(Z.build_theta_over_beta(X.e3_rows(8.0, 0.5, 0.5, (1, 16)), 0.5))
    assert fixed(Z.build_players(X.e4_rows(8.0, 4.0, 0.5, 0.5, players=(3,))))
    assert fixed(Z.build_rounds(X.e5_rows(4.0, factors=(0.5, 2.0)), X.e5_linear_rows(4.0, factors=(0.5, 2.0))))
    assert fixed(Z.build_rounds(X.e5_rows(4.0, factors=(0.5, 2.0)), None))
    assert fixed(Z.build_trajectories(X.e5_trajectories(8.0, 4.0, 0.5, 0.5, 20), R.roots_regime(8.0, 4.0, 0.5, 0.5).x))
    fig = Z.build_email_beliefs(X.e6_summary(0.1, 1.0))
    assert fixed(fig) and len(fig.data[0].x) == C.EMAIL_PLOT_TYPES + 1


def test_pdf_is_a_pdf_with_non_latin_characters_replaced():
    view = V.analyse(S.settings_of_preset("Ungenaue Signale (drei)"))
    data = PDF.build_pdf(view, [V.verdict_count(view), V.verdict_formula(view)])
    assert data[:4] == b"%PDF" and len(data) > 1500
    assert PDF._latin("θ̂ ≥ α²/(2π) – „x“") == 'theta^ >= alpha2/(2pi) - "x"'
    linear = V.analyse(S.settings_of_preset("V3-Zelle (Modell L)"))
    assert PDF.build_pdf(linear, [V.verdict_count(linear)])[:4] == b"%PDF"
    finite = V.analyse(S.settings_of_preset("Wenige Betriebe (N = 5)"))
    assert PDF.build_pdf(finite, [V.verdict_count(finite)])[:4] == b"%PDF"
