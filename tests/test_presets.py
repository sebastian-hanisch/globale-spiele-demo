"""Presets: Vollständigkeit, gültige Werte, erstes Preset = Standardeinstellungen, und jede gerechnete Aussage der Hilfetexte (Zahl der Gleichgewichte, Schwellen, Ränder)."""

import pytest

import glo_constants as C
import glo_evaluation as V
import glo_linear as L
import glo_presets as P
import glo_regime as R
import glo_scenario as S


def _view(name):
    return V.analyse(S.settings_of_preset(name))


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.choices is not None:
                assert preset[key] in spec.choices, (name, key)
            else:
                assert spec.lo <= preset[key] <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS))
    assert S.Settings(**C.PRESETS[first]) == S.Settings()
    assert first.startswith("Genaue Signale")


def test_setting_specs_cover_every_preset_key_and_permalink_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_no_two_presets_are_identical():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


def test_exact_signals_preset_has_one_equilibrium_above_the_formula():
    v = _view("Genaue Signale (eindeutig)")
    assert v.count == 1 and v.formula_ok and v.formula_bound == pytest.approx(2.546, abs=5e-4)
    assert v.rows[0].x == pytest.approx(0.5, abs=1e-12) and v.rows[0].level == pytest.approx(0.5, abs=1e-12)
    assert "0,5" in C.PRESET_HELP["Genaue Signale (eindeutig)"] and "2,55" in C.PRESET_HELP["Genaue Signale (eindeutig)"]


def test_inexact_signals_preset_has_three_equilibria_and_the_stated_numbers():
    name = "Ungenaue Signale (drei)"
    v = _view(name)
    assert v.count == 3 and not v.formula_ok and v.formula_bound == pytest.approx(10.186, abs=5e-4)
    assert [round(r.x, 3) for r in v.rows] == [-0.910, 0.500, 1.910] and [r.stable for r in v.rows] == [True, False, True]
    lo, hi = v.c_band
    assert (round(lo, 3), round(hi, 3)) == (0.416, 0.584)
    text = C.PRESET_HELP[name]
    for part in ("10,19", "−0,910", "0,500", "1,910", "0,416", "0,584"):
        assert part in text, part


def test_close_miss_preset_has_one_equilibrium_just_above_the_band_edge():
    name = "Knapp daneben (c = 0,6)"
    s = S.settings_of_preset(name)
    v = _view(name)
    lo, hi = v.c_band
    assert v.count == 1 and hi < s.c < hi + 0.02 and round(hi, 3) == 0.584
    assert round(v.rows[0].x, 3) == -1.195 and round(v.rows[0].level, 3) == 0.008
    text = C.PRESET_HELP[name]
    for part in ("0,584", "−1,195", "0,008", "c = 0,6"):
        assert part in text, part


def test_few_players_preset_adds_equilibria_the_continuum_does_not_have():
    name = "Wenige Betriebe (N = 5)"
    v = _view(name)
    assert S.settings_of_preset(name).finite and v.count == 3 and v.continuum_count == 1
    assert [round(r.x, 3) for r in v.rows] == [-0.510, 0.781, 1.591] and [r.stable for r in v.rows] == [True, False, True]
    text = C.PRESET_HELP[name]
    for part in ("N = 5", "−0,510", "0,781", "1,591"):
        assert part in text, part


def test_linear_preset_matches_the_v3_cell_and_the_stated_boundary():
    name = "V3-Zelle (Modell L)"
    v = _view(name)
    assert not S.settings_of_preset(name).is_regime and v.count == 3
    assert [round(r.level, 3) for r in v.rows] == [0.050, 0.5, 0.950]
    assert v.formula_bound == pytest.approx(2.546, abs=5e-4) and v.critical_beta == pytest.approx(1.9224, abs=5e-4) and not v.formula_ok
    text = C.PRESET_HELP[name]
    for part in ("0,050", "0,950", "2,55", "1,92"):
        assert part in text, part


def test_the_help_texts_do_not_claim_numbers_that_the_calculation_does_not_show():
    """Gegenprobe zu den Zeichenketten oben: die Schwellen der Hilfetexte sind die gerundeten Schwellen der Rechnung, nicht ein Rest des Plans."""
    e = R.roots_regime(8.0, 4.0, 0.5, 0.5)
    assert [f"{x:.3f}".replace(".", ",").replace("-", "−") for x in e.x] == ["−0,910", "0,500", "1,910"]
    assert f"{L.roots_linear(4.0, 1.0, 0.5).m[0]:.3f}" == "0.050"
