"""Experimente E2 bis E6: Reihen und Tabellen, Zweige (jede Reihe weicht auf mindestens einer Zelle von der Nachbarreihe ab), Zeitschätzungen."""

import pytest

import glo_constants as C
import glo_experiments as X
import glo_iteration as I
import glo_finite as F


def test_e2_regime_band_rows_move_down_with_y_and_vanish_without_the_fold():
    rows = X.e2_regime_band_rows(8.0, 4.0, [0.3, 0.5, 0.7])
    assert [round(r["lo"], 4) for r in rows] == [0.5989, 0.4163, 0.2504] and [round(r["hi"], 4) for r in rows] == [0.7496, 0.5837, 0.4011]
    assert all(r["lo"] is None and r["hi"] is None for r in X.e2_regime_band_rows(4.0, 4.0, [0.3, 0.5]))


def test_e2_linear_fold_rows_open_below_the_critical_beta():
    rows = X.e2_linear_fold_rows(4.0, [0.5, 1.0, 1.9, 2.0, 16.0])
    assert [r["hi"] is not None for r in rows] == [True, True, True, False, False]
    assert rows[0]["lo"] == pytest.approx(1 - rows[0]["hi"]) and rows[0]["hi"] > rows[1]["hi"] > rows[2]["hi"] > 0.5          # der Bereich schrumpft zum Berührpunkt


def test_e3_rows_unique_and_multiple_branches():
    rows = X.e3_rows(4.0, 0.5, 0.5, (1, 16))
    assert rows[0]["count"] == 3 and rows[0]["p_lost"] is None and rows[1]["count"] == 1 and rows[1]["p_lost"] == pytest.approx(0.3413, abs=5e-5)
    assert all(r["limit"] == 0.5 for r in rows)


def test_e4_rows_start_with_the_continuum_and_follow_the_requested_players():
    rows = X.e4_rows(8.0, 4.0, 0.5, 0.5, players=(3, 10))
    assert [r["n"] for r in rows] == [0, 3, 10] and [r["count"] for r in rows] == [3, 1, 3] and all(r["seconds"] >= 0 for r in rows)


def test_time_estimates_grow_with_n_and_with_the_alpha_series():
    est = [X.finite_estimated_seconds(n) for n in (2, 10, 100, 250, 1000)]
    assert est == sorted(est) and est[-1] > 1.0 and est[0] < 0.2
    assert X.e4_estimated_seconds() == pytest.approx(sum(X.finite_estimated_seconds(n) for n in C.E4_PLAYERS))
    assert X.e5_estimated_seconds(8.0) > 0.5


def test_e5_rows_cover_both_sides_and_the_linear_model_has_its_own_boundary():
    rows = X.e5_rows(4.0, factors=(0.5, 0.95, 1.05, 2.0))
    assert [r["count"] for r in rows] == [3, 3, 1, 1] and all(r["low"].reached and r["high"].reached for r in rows)
    lin = X.e5_linear_rows(4.0, factors=(0.5, 0.95, 1.05, 2.0))
    assert [r["count"] for r in lin] == [3, 3, 1, 1] and lin[1]["low"].rounds > lin[0]["low"].rounds
    assert all(a["beta"] != b["beta"] for a, b in zip(rows, lin))


def test_e5_cap_is_reported_as_not_reached():
    rows = X.e5_rows(8.0, factors=(1.0013,), cap=1000)
    assert not rows[0]["low"].reached and rows[0]["low"].rounds == 1000


def test_e5_trajectories_have_four_starts_only_with_three_roots():
    assert len(X.e5_trajectories(8.0, 4.0, 0.5, 0.5)) == 4 and len(X.e5_trajectories(4.0, 16.0, 0.5, 0.5)) == 2
    t = X.e5_trajectories(8.0, 4.0, 0.5, 0.5, rounds=10)
    assert all(len(x["path"]) == 11 for x in t) and t[0]["path"][0] == I.START_LOW


def test_e6_summary_and_knowledge_rows():
    sm = X.e6_summary(0.1, 1.0)
    assert sm["q"] == 0.5 and sm["threshold"] == pytest.approx(0.9 / 1.9) and sm["common_states"] == 0
    rows = X.e6_knowledge_rows(0.1, levels=(1, 5), games=20_000)
    assert [r["n"] for r in rows] == [1, 5] and rows[0]["formula"] == pytest.approx(0.9) and rows[1]["formula"] == pytest.approx(0.9 ** 5)
    assert all(abs(r["simulated"] - r["formula"]) < 4 * r["se"] for r in rows) and rows[0]["games_in_g"] > 9000


def test_finite_branches_differ_between_scan_sizes_only_in_the_raster_point_count():
    a, b = F.roots_finite(10, 8.0, 4.0, 0.5, 0.5, scan=41), F.roots_finite(10, 8.0, 4.0, 0.5, 0.5, scan=241)
    assert a.count == b.count == 3 and a.scan_points == 41 and b.scan_points == 241
    assert a.x == pytest.approx(b.x, abs=1e-7)
