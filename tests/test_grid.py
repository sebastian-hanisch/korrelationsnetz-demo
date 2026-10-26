"""Das vorgerechnete Raster (data/network_grid.json): Aufbau, Plausibilität und Übereinstimmung mit dem Erzeuger (Probe)."""

import pytest

import knz_algorithm as K
import knz_constants as C
import knz_evaluation as ev


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None, "data/network_grid.json fehlt: python tools/make_grid.py"
    return g


def _assert_close(a, b, path="probe", loose=False):
    """Verschachtelte Wörter und Listen: Zahlen mit relativer Toleranz (Summationsreihenfolge je Plattform, DEMO-PLAYBOOK §4), alles andere exakt.
    Zellen, die vom Graphical Lasso abhängen (EBIC, bestes λ, Pfad), dürfen am Rand einer Entscheidung abweichen (Gleitkomma je Plattform, Lehre aus Stück 10): dort nur grobe Toleranz."""
    if isinstance(a, dict):
        assert set(a) == set(b), path
        for k in a:
            _assert_close(a[k], b[k], f"{path}.{k}", loose or k.startswith("ebic") or k in ("best", "path"))
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{i}]", loose)
    elif isinstance(a, float):
        assert a == pytest.approx(b, rel=1e-8, abs=1e-10) if not loose else a == pytest.approx(b, abs=3.0 if abs(b) > 1.0 else 0.2), path
    else:
        assert a == b, path


def test_probe_still_matches_the_generator(grid):
    """Ändert sich Erzeuger, Statistik oder Seed-Ableitung, passt die gespeicherte Probe nicht mehr: dann tools/make_grid.py neu laufen lassen."""
    _assert_close(ev.grid_probe(), grid["probe"])


def test_the_exact_probe_cells_do_not_depend_on_the_platform(grid):
    probe = grid["probe"]
    assert probe["methods"]["schwelle"]["edges"] > 0 and probe["methods"]["rank"]["edges"] == 39.0 and probe["datasets"] == 2


def test_grid_has_every_cell_exactly_once(grid):
    def keys(cells):
        return sorted((c["truth"], c["n"]) for c in cells)

    assert keys(grid["e1"]) == sorted((t, C.E1_N) for t in C.TRUTHS)
    assert keys(grid["e2"]) == sorted((t, n) for t in C.E2_TRUTHS for n in C.E2_NS)
    assert keys(grid["e3"]) == sorted((t, n) for t in C.E2_TRUTHS for n in C.E3_NS)
    assert keys(grid["e5"]) == sorted(("faktor", n) for n in C.E5_NS)


def test_grid_meta_and_dataset_counts(grid):
    assert grid["meta"]["seeds"] == list(C.GRID_SEEDS)
    assert all(c["datasets"] == len(C.GRID_SEEDS) * C.E1_RUNS and c["p"] == C.GRID_P for c in grid["e1"])
    assert all(c["datasets"] == len(C.GRID_SEEDS) * C.E2_RUNS for c in grid["e2"]) and all(c["datasets"] == len(C.GRID_SEEDS) * C.E3_RUNS for c in grid["e3"])
    assert all(c["datasets"] == len(C.GRID_SEEDS) * C.E5_RUNS and "path" not in c for c in grid["e5"])
    assert all("path" in c for c in grid["e1"] + grid["e2"] + grid["e3"])


def test_every_cell_is_internally_consistent(grid):
    for c in grid["e1"] + grid["e2"] + grid["e3"] + grid["e5"]:
        T = K.true_edges(c["truth"], c["p"])
        assert c["true_edges"] == K.edge_count(T) and c["trans_true"] == K.transitivity(T)
        for key, m in c["methods"].items():
            assert m["edges"] == pytest.approx(m["right"] + m["wrong"], abs=1e-9) and 0.0 <= m["f1"] <= 1.0 and 0.0 <= m["trans"] <= 1.0, (c["truth"], c["n"], key)
        assert c["methods"]["rank"]["edges"] == c["true_edges"]
        assert ("teil" in c["methods"]) == (c["n"] - c["p"] - 1 > 0), (c["truth"], c["n"])


def test_stored_expectations_equal_a_fresh_computation(grid):
    for c in grid["e1"] + grid["e2"] + grid["e3"] + grid["e5"]:
        Sg, T = K.true_cov(c["truth"], c["p"]), K.true_edges(c["truth"], c["p"])
        assert c["expected"]["schwelle"] == pytest.approx(K.expected_threshold_edges(Sg, c["n"], C.DEFAULT_TAU, T), abs=1e-9)
        exp = K.expected_partial_edges(Sg, c["n"], C.DEFAULT_ALPHA, T)
        assert (c["expected"]["teil"] is None) == (exp is None) and (exp is None or c["expected"]["teil"] == pytest.approx(exp, abs=1e-9))


def test_measured_edges_agree_with_the_exact_expectation(grid):
    for c in grid["e1"] + grid["e2"] + grid["e3"] + grid["e5"]:
        for key in ("schwelle", "teil"):
            if key in c["methods"]:
                exp = c["expected"][key]["edges"]
                assert c["methods"][key]["edges"] == pytest.approx(exp, abs=max(3.0, 0.06 * exp)), (c["truth"], c["n"], key)


def test_ebic_picks_fewer_edges_with_a_larger_gamma(grid):
    for c in grid["e1"] + grid["e2"] + grid["e3"]:
        m = c["methods"]
        assert m["ebic_0.0"]["edges"] >= m["ebic_0.5"]["edges"] - 0.5 >= m["ebic_1.0"]["edges"] - 1.0, (c["truth"], c["n"])


def test_path_curves_start_empty_and_the_best_lambda_beats_every_path_mean(grid):
    for c in grid["e1"] + grid["e2"] + grid["e3"]:
        path = c["path"]
        assert path["edges"][0] == 0.0 and len(path["f1"]) == len(path["edges"]) == len(path["wrong"]) == len(path["ratio"]) == C.PATH_K
        assert c["methods"]["best"]["f1"] >= max(path["f1"]) - 1e-9, (c["truth"], c["n"])                  # je Datensatz das beste, im Mittel nie schlechter als der beste Mittelwert


def test_the_headline_numbers_of_the_four_worlds(grid):
    by = {c["truth"]: c["methods"] for c in grid["e1"]}
    assert by["faktor"]["schwelle"]["edges"] > 700 and by["faktor"]["teil"]["edges"] < 1.0 and by["faktor"]["schwelle"]["trans"] > 0.9
    assert by["gitter"]["schwelle"]["edges"] < 12 and by["kette"]["teil"]["f1"] > 0.99 and by["kette"]["schwelle"]["wrong"] > 2.0
