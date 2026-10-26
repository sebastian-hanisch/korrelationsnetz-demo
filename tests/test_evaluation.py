"""Auswertung: Zeilen der Hauptansicht, Regeln für λ, Pfad, Zellen des Rasters."""

import numpy as np
import pytest

import knz_algorithm as K
import knz_constants as C
import knz_evaluation as ev


def _small(**kw):
    base = dict(truth="kette", p=10, n=300, seed=3)
    base.update(kw)
    return ev.Settings(**base)


def test_defaults_are_the_documented_standard_case():
    s = ev.Settings()
    assert (s.truth, s.p, s.n, s.tau, s.alpha, s.rule, s.lam, s.gamma, s.seed) == ("kette", 40, 500, 0.3, 0.05, "ebic", 0.15, 0.5, 35)


def test_settings_are_keyword_only_and_hashable():
    with pytest.raises(TypeError):
        ev.Settings("kette", 40)
    assert hash(ev.Settings()) == hash(ev.Settings()) and ev.Settings(n=100) != ev.Settings()


def test_analysis_has_the_four_rows_in_order_and_they_match_direct_calls():
    s = _small()
    a = ev.analyse(s)
    assert [r.key for r in a.rows] == ["schwelle", "teil", "glasso", "rank"]
    X, R = ev.make_data(s)
    assert np.array_equal(R, K.sample_corr(X)) and np.array_equal(a.R, R)
    T = K.true_edges("kette", 10)
    assert np.array_equal(a.nets["schwelle"], K.threshold_net(R, 0.3)) and np.array_equal(a.nets["teil"], K.partial_net(R, 300, 0.05)[0])
    assert np.array_equal(a.nets["rank"], K.rank_net(R, K.edge_count(T)))
    for r in a.rows:
        sc = K.score(a.nets[r.key], T)
        assert (r.edges, r.right, r.wrong, r.missing, r.f1) == (sc["edges"], sc["right"], sc["wrong"], sc["missing"], sc["f1"])
        assert r.trans == K.transitivity(a.nets[r.key]) and r.name == ev.METHOD_NAMES[r.key]


def test_expected_values_only_for_the_two_simple_nets_and_they_match_the_algorithm():
    a = ev.analyse(_small())
    Sg, T = K.true_cov("kette", 10), K.true_edges("kette", 10)
    assert a.row("schwelle").expected == K.expected_threshold_edges(Sg, 300, 0.3, T) and a.row("teil").expected == K.expected_partial_edges(Sg, 300, 0.05, T)
    assert a.row("glasso").expected is None and a.row("rank").expected is None


def test_no_partial_row_when_n_is_at_most_p_plus_one():
    a = ev.analyse(_small(n=11))
    assert a.row("teil") is None and a.partial is None and "teil" not in a.nets and [r.key for r in a.rows] == ["schwelle", "glasso", "rank"]
    assert ev.analyse(_small(n=12)).row("teil") is not None                               # df = 1


def test_rule_ebic_best_and_fest_pick_the_documented_path_values():
    s = _small(n=200)
    path = ev.compute_path(s.truth, s.p, s.n, s.seed)
    e = ev.analyse(ev.Settings(**{**s.__dict__, "rule": "ebic"}), path)
    b = ev.analyse(ev.Settings(**{**s.__dict__, "rule": "best"}), path)
    f = ev.analyse(ev.Settings(**{**s.__dict__, "rule": "fest", "lam": 0.2}), path)
    assert e.lam_used == pytest.approx(path["lams"][e.i_ebic]) and b.lam_used == pytest.approx(path["lams"][b.i_best]) and f.lam_used == 0.2
    assert np.array_equal(e.nets["glasso"], K.glasso_net(path["thetas"][e.i_ebic])) and np.array_equal(b.nets["glasso"], K.glasso_net(path["thetas"][b.i_best]))
    _x, R = ev.make_data(s)
    assert np.array_equal(f.nets["glasso"], K.glasso_net(K.glasso(R, 0.2)[0]))
    assert max(b.path_f1) == b.path_f1[b.i_best] and b.row("glasso").f1 == pytest.approx(max(b.path_f1))


def test_gamma_only_changes_the_ebic_choice_and_a_larger_gamma_never_picks_more_edges():
    s = _small(n=300, p=14)
    path = ev.compute_path(s.truth, s.p, s.n, s.seed)
    counts = []
    for g in C.GAMMAS:
        a = ev.analyse(ev.Settings(**{**s.__dict__, "gamma": g}), path)
        counts.append(a.path_edges[a.i_ebic])
        assert a.row("schwelle").edges == ev.analyse(s, path).row("schwelle").edges
    assert counts == sorted(counts, reverse=True)


def test_path_reuse_gives_the_same_analysis_as_a_fresh_path():
    s = _small()
    a1 = ev.analyse(s, ev.compute_path(s.truth, s.p, s.n, s.seed))
    a2 = ev.analyse(s)
    assert [(r.edges, r.right, r.f1) for r in a1.rows] == [(r.edges, r.right, r.f1) for r in a2.rows] and a1.i_ebic == a2.i_ebic and a1.lam_used == a2.lam_used


def test_the_path_has_the_documented_length_and_starts_with_the_empty_net():
    a = ev.analyse(_small())
    assert len(a.lams) == C.PATH_K == len(a.path_edges) == len(a.path_f1) == len(a.path_ebic) and a.path_edges[0] == 0
    assert a.trans_true == 0.0


def test_factor_world_rows_count_every_edge_as_wrong():
    a = ev.analyse(ev.Settings(truth="faktor", p=12, n=400, seed=2))
    assert a.T.sum() == 0 and all(r.right == 0 and r.missing == 0 for r in a.rows) and a.row("schwelle").wrong == a.row("schwelle").edges and a.row("rank").f1 == 1.0


def test_net_cell_structure_and_consistency():
    c = ev.net_cell("kette", 200, 12, 1, seeds=(5, 6))
    assert (c["truth"], c["n"], c["p"], c["datasets"], c["true_edges"]) == ("kette", 200, 12, 2, 11)
    assert set(c["methods"]) == {"schwelle", "teil", "rank", "ebic_0.0", "ebic_0.5", "ebic_1.0", "best"}
    for m in c["methods"].values():
        assert m["edges"] == pytest.approx(m["right"] + m["wrong"], abs=1e-9) and 0.0 <= m["f1"] <= 1.0 and m["f1_sd"] >= 0.0
    assert c["methods"]["rank"]["edges"] == 11 and c["expected"]["schwelle"]["edges"] > 0 and c["expected"]["teil"] is not None
    path = c["path"]
    assert len(path["ratio"]) == C.PATH_K and path["ratio"][0] == 1.0 and path["ratio"][-1] == pytest.approx(C.PATH_RATIO) and path["edges"][0] == 0.0
    assert set(path["ebic_ratio"]) == {"0.0", "0.5", "1.0"} and C.PATH_RATIO <= path["best_ratio"] <= 1.0
    assert ev.net_cell("kette", 200, 12, 1, seeds=(5, 6)) == c                                # deterministisch


def test_net_cell_without_path_and_without_partial_net():
    c = ev.net_cell("faktor", 10, 12, 2, seeds=(5,), with_path=False)
    assert set(c["methods"]) == {"schwelle", "rank"} and "path" not in c and c["expected"]["teil"] is None and c["datasets"] == 2


def test_find_cell():
    cells = [{"truth": "kette", "n": 1}, {"truth": "gitter", "n": 1}, {"truth": "kette", "n": 2}]
    assert ev.find_cell(cells, "kette", 2) is cells[2] and ev.find_cell(cells, "faktor", 1) is None


def test_load_grid_returns_none_for_a_missing_or_broken_file(tmp_path):
    assert ev.load_grid(tmp_path / "gibt-es-nicht.json") is None
    bad = tmp_path / "kaputt.json"
    bad.write_text("{", encoding="utf-8")
    assert ev.load_grid(bad) is None
    good = tmp_path / "gut.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert ev.load_grid(good) == {"a": 1}


def test_de_formats_with_a_decimal_comma():
    assert ev.de(0.5) == "0,50" and ev.de(12.345, 1) == "12,3" and ev.de(3, 0) == "3"
