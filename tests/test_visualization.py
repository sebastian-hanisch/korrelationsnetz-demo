"""Abbildungen: jede wird gebaut, hat die erwarteten Spuren und Achsen."""

import numpy as np
import pytest

import knz_algorithm as K
import knz_constants as C
import knz_evaluation as ev
import knz_visualization as V


@pytest.fixture(scope="module")
def grid():
    return ev.load_grid()


def _traces(fig, name):
    return [t for t in fig.data if t.name == name]


def test_every_chart_builds_and_locks_its_axes(grid):
    a = ev.analyse(ev.Settings(truth="gitter", p=15, n=200, seed=3))
    cells = {(c["truth"], c["n"]): c for c in grid["e1"] + grid["e2"]}
    figs = [V.build_networks(a), V.build_pairs(a), V.build_path(a), V.build_e1(grid["e1"], 0.5), V.build_f1_vs_n(grid["e2"], "kette", 0.5), V.build_edges_vs_n(grid["e2"], "gitter"),
            V.build_f1_vs_n(grid["e3"], "zufall", 0.0), V.build_edges_vs_n(grid["e3"], "kette"), V.build_e4(cells[("kette", 500)]), V.build_e5(grid["e5"])]
    for fig in figs:
        assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and len(fig.data) >= 1


def test_node_positions_are_a_grid_for_the_grid_world_and_a_unit_circle_otherwise():
    g = V.node_positions("gitter", 40)
    assert g.shape == (40, 2) and set(g[:, 1]) == {0.0, -1.0, -2.0, -3.0, -4.0} and set(g[:, 0]) == set(range(8)) and tuple(g[9]) == (1.0, -1.0)
    k = V.node_positions("kette", 40)
    assert [tuple(k[i]) for i in (0, 7, 8, 9, 15, 16)] == [(0.0, 0.0), (7.0, 0.0), (7.0, -1.0), (6.0, -1.0), (0.0, -1.0), (0.0, -2.0)]            # Schlange: aufeinanderfolgende Kennzahlen sind Nachbarn
    assert all(abs(k[i + 1] - k[i]).sum() == 1.0 for i in range(39))
    c = V.node_positions("zufall", 12)
    assert np.allclose(np.hypot(c[:, 0], c[:, 1]), 1.0) and tuple(np.round(c[0], 9)) == (0.0, 1.0) and c[3, 0] > 0.99
    assert V.node_positions("faktor", 10).shape == (10, 2)


def test_networks_draw_right_wrong_and_missing_edges_with_different_line_styles():
    a = ev.analyse(ev.Settings(truth="gitter", p=15, n=500, seed=3))
    fig = V.build_networks(a)
    names = [t.name for t in fig.data if t.showlegend]
    assert names == ["richtig", "falsch", "fehlend"]
    styles = {t.name: (t.line.color, t.line.dash) for t in fig.data if t.showlegend}
    assert len({v for v in styles.values()}) == 3 and len({v[1] for v in styles.values()}) == 3                       # drei Farben UND drei Linienarten
    assert len(fig.layout.annotations) == 4 and fig.layout.annotations[0].text == f"Wahrheit: {K.edge_count(a.T)} echte Kanten"
    # Zahl der Segmente je Spur = Zahl der Kanten (drei Punkte je Kante: Anfang, Ende, Lücke)
    right = [t for t in fig.data if t.name == "richtig"][0]
    assert sum(v is None for v in right.x) == a.row("schwelle").right


def test_networks_without_partial_net_show_a_note():
    a = ev.analyse(ev.Settings(truth="kette", p=10, n=11, seed=3))
    fig = V.build_networks(a)
    assert "gibt es hier nicht" in fig.layout.annotations[2].text and any("keine Inverse" in (x.text or "") for x in fig.layout.annotations)


def test_pairs_chart_marks_true_edges_and_both_thresholds():
    a = ev.analyse(ev.Settings(truth="zufall", p=20, n=300, seed=3))
    fig = V.build_pairs(a)
    assert [t.name for t in fig.data] == ["Paar ohne direkte Kante", "echte Kante"]
    assert len(fig.data[0].x) + len(fig.data[1].x) == 190 and len(fig.data[1].x) == K.edge_count(a.T)
    assert len(fig.layout.shapes) == 2 and fig.layout.shapes[0].x0 == a.settings.tau
    b = ev.analyse(ev.Settings(truth="faktor", p=20, n=300, seed=3))
    assert [t.name for t in V.build_pairs(b).data] == ["Paar ohne direkte Kante"]


def test_path_chart_has_the_two_curves_and_the_marks():
    a = ev.analyse(ev.Settings(truth="kette", p=12, n=300, seed=3))
    fig = V.build_path(a)
    assert list(fig.data[0].y) == a.path_edges and list(fig.data[1].y) == a.path_f1
    assert {"EBIC", "bestes λ"} <= {t.name for t in fig.data} and fig.layout.xaxis2.type == "log"


def test_e1_has_five_methods_and_none_for_the_missing_partial_net(grid):
    fig = V.build_e1(grid["e1"], 0.5)
    assert [t.name for t in fig.data] == [V.METHOD_LABELS[k] for k in ("schwelle", "teil", "ebic", "best", "rank")]
    assert list(fig.data[0].x) == ["Kette", "Gitter", "Zufallsnetz", "Faktor"]
    assert fig.data[0].y[0] == [c for c in grid["e1"] if c["truth"] == "kette"][0]["methods"]["schwelle"]["f1"]


def test_f1_vs_n_skips_methods_without_values_and_uses_the_chosen_gamma(grid):
    fig = V.build_f1_vs_n(grid["e3"], "kette", 0.5)
    teil = _traces(fig, V.METHOD_LABELS["teil"])[0]
    assert list(teil.x) == [45, 60]                                                                                    # erst ab n > p + 1
    ebic0, ebic1 = V.build_f1_vs_n(grid["e3"], "kette", 0.0), V.build_f1_vs_n(grid["e3"], "kette", 1.0)
    assert list(_traces(ebic0, V.METHOD_LABELS["ebic"])[0].y) != list(_traces(ebic1, V.METHOD_LABELS["ebic"])[0].y)


def test_edges_vs_n_shows_measured_and_expected_and_the_true_edge_line(grid):
    fig = V.build_edges_vs_n(grid["e2"], "kette")
    assert [t.name for t in fig.data] == ["Schwellennetz, gemessen", "Schwellennetz, erwartet", "Teilkorrelationsnetz, gemessen", "Teilkorrelationsnetz, erwartet"]
    assert list(fig.data[0].x) == list(C.E2_NS) and len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == 39
    no_teil = V.build_edges_vs_n([c for c in grid["e3"] if c["n"] == 20], "kette")
    assert len(no_teil.data) == 2


def test_e4_marks_the_ebic_values_for_all_gammas_and_the_best_lambda(grid):
    cell = [c for c in grid["e1"] if c["truth"] == "kette"][0]
    fig = V.build_e4(cell)
    names = [t.name for t in fig.data if t.name]
    assert names[-4:] == ["EBIC γ = 0,0", "EBIC γ = 0,5", "EBIC γ = 1,0", "bestes λ"] and fig.layout.xaxis2.autorange == "reversed"


def test_e5_has_the_pair_count_line(grid):
    fig = V.build_e5(grid["e5"])
    assert [t.name for t in fig.data] == ["Schwellennetz", "Teilkorrelationsnetz"] and fig.layout.shapes[0].y0 == 780
