"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Regler an seine
Grenzen und prüft, dass kein Python-Fehler auftritt - insbesondere `StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key=)
und `StreamlitAPIException` bei Reglern mit berechneten Grenzen. Die ▶️-Knöpfe laufen einzeln (die Schleife lässt sie aus)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import glo_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer
AUTOPLAY_MARKER = "▶"  # ▶: Knöpfe mit Rechenlast, einzeln getestet


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([w.value for w in at.warning] + [s.value for s in at.success] + [i.value for i in at.info] + [e.value for e in at.error])


def test_app_loads_without_exception():
    at = _fresh_app()
    assert "Ein einziges Gleichgewicht" in _texts(at)             # Standard: genaue Signale


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run()
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            slider = at.slider[index]
            if slider.disabled:
                continue
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run()
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


@pytest.mark.parametrize("model", C.MODELS)
@pytest.mark.parametrize("edge", ["min", "max"])
def test_all_sidebar_sliders_at_their_minima_and_all_at_their_maxima_do_not_raise(model, edge):
    """Eckfall-Matrix (DEMO-PLAYBOOK §1d): alle Regler zugleich am Minimum bzw. Maximum, je Modell; im Modell R mit der kleinsten (N = 2) bzw. größten Spielerzahl (N = 1 000).
    Ausgegraute Regler (c und N im Modell L) lassen sich nicht setzen und bleiben auf ihrem Wert."""
    at = _fresh_app()
    at.selectbox(key="model_select").set_value(model).run()
    for slider in at.sidebar.slider:
        if slider.disabled:
            continue
        slider.set_value(slider.min if edge == "min" else slider.max)
    if model == C.MODEL_R:
        at.selectbox(key="players_select").set_value(2 if edge == "min" else 1000)
    at.run()
    assert not at.exception, (model, edge, [str(e) for e in at.exception])


def test_underflow_corner_of_the_linear_model_reports_the_flag():
    """α = 30, β = 0,1, y = ½ im Modell L: κ·½ > 37, die Randwurzeln liegen unter 10^-300 (als 0 und 1 gemeldet), der Hinweis steht im Verdikt."""
    at = _fresh_app()
    at.selectbox(key="model_select").set_value(C.MODEL_L).run()
    at.slider(key="alpha_slider").set_value(30.0)
    at.slider(key="beta_slider").set_value(0.1)
    at.run()
    assert not at.exception and "Wurzeln unter 10^-300 sind als 0 bzw. 1 gemeldet" in _texts(at)


def test_n_equals_two_and_n_equals_one_thousand_run_through():
    for n in (2, 1000):
        at = _fresh_app()
        at.selectbox(key="players_select").set_value(n).run()
        assert not at.exception, (n, [str(e) for e in at.exception])
        assert f"Bei N = {n} (Kontinuum: 1 Gleichgewicht)" in _texts(at)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_loads_and_shows_its_count(name):
    at = _fresh_app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    want = "Ein einziges Gleichgewicht" if name.startswith(("Genaue", "Knapp")) else "Drei Gleichgewichte"
    assert want in _texts(at), name


def test_the_play_buttons_work_individually():
    at = _fresh_app()
    assert "e4_signature" not in at.session_state
    at.button(key="e4_button").click().run()
    assert not at.exception and at.session_state["e4_signature"] is not None
    assert any("Betriebe N" in df.value.columns.tolist() for df in at.dataframe) and "Zahl der Gleichgewichte nach Spielerzahl" in _texts(at)

    at = _fresh_app()
    at.button(key="e5_button").click().run()
    assert not at.exception and at.session_state["e5_signature"] == 8.0
    assert any("Runden (Modell R)" in df.value.columns.tolist() for df in at.dataframe) and "Runden bis auf 10^-6" in _texts(at)

    at = _fresh_app()
    at.button(key="e6_sim_button").click().run()
    assert not at.exception and "Gegenseitiges Wissen der Stufe n" in _texts(at)


def test_e4_result_is_hidden_when_the_settings_change():
    at = _fresh_app()
    at.button(key="e4_button").click().run()
    assert "Zahl der Gleichgewichte nach Spielerzahl" in _texts(at)
    at.slider(key="beta_slider").set_value(8.0).run()
    assert not at.exception and "Zahl der Gleichgewichte nach Spielerzahl" not in _texts(at)


def test_c_and_players_are_disabled_in_the_linear_model_and_enabled_in_the_regime_model():
    at = _fresh_app()
    assert at.sidebar.slider(key="c_slider").disabled is False and at.sidebar.selectbox(key="players_select").disabled is False
    at.selectbox(key="model_select").set_value(C.MODEL_L).run()
    assert at.sidebar.slider(key="c_slider").disabled is True and at.sidebar.selectbox(key="players_select").disabled is True


def test_email_tab_controls_change_the_verdict():
    at = _fresh_app()
    assert "nie jemand an" in _texts(at)                          # ε = 0,1, L = 1: q = ½ liegt über 0,4737
    at.slider(key="e6_loss").set_value(0.5).run()
    assert not at.exception and "Das größte Gleichgewicht enthält einen Angriff" in _texts(at)


def test_round_selector_appears_only_with_several_equilibria():
    at = _fresh_app()
    assert not [s for s in at.selectbox if s.label == "Gespieltes Gleichgewicht"]
    at.sidebar.button(key="preset_Ungenaue Signale (drei)").click().run()
    picks = [s for s in at.selectbox if s.label == "Gespieltes Gleichgewicht"]
    assert len(picks) == 1 and len(picks[0].options) == 3
    picks[0].set_value(2).run()
    assert not at.exception


def test_pdf_button_creates_the_download():
    at = _fresh_app()
    at.button(key="pdf_button").click().run()
    assert not at.exception and at.session_state["pdf_bytes"][:4] == b"%PDF"


def test_permalink_sets_the_regulators_and_clamps_to_the_bounds():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["a"] = "8"
    at.query_params["b"] = "4"
    at.query_params["c"] = "0.6"
    at.query_params["n"] = "5"
    at.query_params["y"] = "7"                       # über der Grenze: wird auf 0,9 begrenzt
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["alpha_slider"] == 8.0 and at.session_state["beta_slider"] == 4.0 and at.session_state["c_slider"] == 0.6
    assert at.session_state["players_select"] == 5 and at.session_state["y_slider"] == 0.9


def test_permalink_ignores_values_outside_the_allowed_choices_and_garbage():
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.query_params["n"] = "7"                        # keine erlaubte Spielerzahl
    at.query_params["model"] = "Quatsch"
    at.query_params["b"] = "nan"
    at.query_params["a"] = "abc"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["players_select"] == 0 and at.session_state["model_select"] == C.MODEL_R and at.session_state["beta_slider"] == C.DEFAULT_BETA and at.session_state["alpha_slider"] == C.DEFAULT_ALPHA


def test_permalink_round_trip_after_a_preset_click():
    at = _fresh_app()
    at.sidebar.button(key="preset_Wenige Betriebe (N = 5)").click().run()
    assert at.query_params["n"] == ["5"] or at.query_params["n"] == "5"
