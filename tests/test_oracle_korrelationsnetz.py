"""Orakel (ORACLE-PLAYBOOK): unabhängige Wege zu denselben Zahlen.

Teilkorrelation gegen die Korrelation von Regressionsresiduen; Fisher-z-Entscheidungen gegen scipy.stats.norm in einer Schleife über alle Paare; der eigene Graphical Lasso gegen
die Optimalitätsbedingungen (ohne Bibliothek) und gegen sklearn.covariance.graphical_lasso (Zielfunktion); Transitivität gegen networkx; EBIC gegen eine Schleifenfassung;
die exakten Erwartungen beider einfacher Netze gegen scipy und gegen Simulationen."""

import math
import warnings

import networkx as nx
import numpy as np
import pytest
from scipy import stats
from sklearn.covariance import graphical_lasso

import knz_algorithm as K
import knz_constants as C
from prs_scenario import VRng


def _data(truth="kette", n=300, p=12, seed=3):
    X = K.draw(truth, n, p, VRng(seed))
    return X, K.sample_corr(X)


def test_partial_correlation_equals_the_correlation_of_regression_residuals():
    X, R = _data("zufall", 200, 10)
    Xc = X - X.mean(axis=0)
    Pc = K.partial_corr(R)
    for i, j in [(0, 1), (2, 7), (4, 9), (3, 5)]:
        rest = [k for k in range(10) if k not in (i, j)]
        Z = Xc[:, rest]
        ei = Xc[:, i] - Z @ np.linalg.lstsq(Z, Xc[:, i], rcond=None)[0]
        ej = Xc[:, j] - Z @ np.linalg.lstsq(Z, Xc[:, j], rcond=None)[0]
        assert Pc[i, j] == pytest.approx(np.corrcoef(ei, ej)[0, 1], abs=1e-10), (i, j)


def test_bonferroni_critical_value_against_scipy_for_many_cases():
    for p in (3, 10, 40):
        for alpha in (0.01, 0.05):
            m = p * (p - 1) // 2
            assert K.bonferroni_z(p, alpha) == pytest.approx(stats.norm.ppf(1 - alpha / (2 * m)), abs=1e-9)


def test_partial_net_matches_a_pair_by_pair_scipy_test():
    X, R = _data("gitter", 400, 15)
    A, Pc = K.partial_net(R, 400, 0.05)
    m = 15 * 14 // 2
    for i in range(15):
        for j in range(15):
            if i == j:
                assert not A[i, j]
                continue
            z = math.atanh(Pc[i, j]) * math.sqrt(400 - 15 - 1)
            assert A[i, j] == (2 * stats.norm.sf(abs(z)) < 0.05 / m), (i, j)


def test_threshold_net_matches_a_double_loop():
    _x, R = _data("zufall", 100, 14)
    for tau in (0.1, 0.3, 0.5):
        A = K.threshold_net(R, tau)
        for i in range(14):
            for j in range(14):
                assert A[i, j] == (i != j and abs(R[i, j]) > tau)


def test_rank_net_matches_a_sorted_list_of_pairs():
    _x, R = _data("zufall", 100, 14)
    pairs = sorted(((abs(R[i, j]), i, j) for i in range(14) for j in range(i + 1, 14)), key=lambda t: -t[0])
    for k in (0, 1, 5, 20):
        A = K.rank_net(R, k)
        assert {(i, j) for _, i, j in pairs[:k]} == {(i, j) for i in range(14) for j in range(i + 1, 14) if A[i, j]}


def test_transitivity_matches_networkx_on_random_graphs_and_the_worlds():
    rng = np.random.default_rng(4)
    for _ in range(20):
        U = np.triu(rng.random((12, 12)) < 0.3, 1)
        A = U | U.T
        assert K.transitivity(A) == pytest.approx(nx.transitivity(nx.from_numpy_array(A.astype(int))), abs=1e-12)
    for truth in C.TRUTHS:
        T = K.true_edges(truth, 40)
        assert K.transitivity(T) == pytest.approx(nx.transitivity(nx.from_numpy_array(T.astype(int))), abs=1e-12)


def _objective(S, Theta, lam):
    off = ~np.eye(S.shape[0], dtype=bool)
    return -np.linalg.slogdet(Theta)[1] + np.trace(S @ Theta) + lam * np.abs(Theta[off]).sum()


@pytest.mark.parametrize("truth, lam", [("kette", 0.1), ("kette", 0.3), ("gitter", 0.15), ("zufall", 0.12), ("faktor", 0.2)])
def test_glasso_matches_the_optimality_conditions_and_sklearn(truth, lam):
    _x, R = _data(truth, 300, 15)
    Theta, _W, _B = K.glasso(R, lam, tol=1e-8)
    assert K.kkt_violation(R, Theta, lam) < 1e-4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _cov, prec = graphical_lasso(R, alpha=lam, tol=1e-6, max_iter=2000)
    assert _objective(R, Theta, lam) <= _objective(R, prec, lam) + 1e-6                    # unser Minimum ist höchstens so hoch wie das von sklearn
    assert np.abs(Theta - prec).max() < 5e-3
    assert K.edge_count(K.glasso_net(Theta)) == pytest.approx(int((np.abs(np.triu(prec, 1)) > C.EDGE_EPS).sum()), abs=2)       # sklearn rechnet bis zu seiner Toleranz: Kanten am Rand dürfen abweichen


def test_glasso_warm_start_gives_the_same_solution_as_a_cold_start():
    _x, R = _data("kette", 300, 12)
    lams, thetas = K.glasso_path(R, k=6)
    cold, _W, _B = K.glasso(R, float(lams[-1]), tol=1e-9)
    assert np.abs(thetas[-1] - cold).max() < 1e-4


def test_ebic_matches_a_loop_version():
    _x, R = _data("gitter", 250, 10)
    Theta, _W, _B = K.glasso(R, 0.12)
    p, n = 10, 250
    tr = sum(R[i, j] * Theta[j, i] for i in range(p) for j in range(p))
    edges = sum(1 for i in range(p) for j in range(i + 1, p) if abs(Theta[i, j]) > C.EDGE_EPS)
    for gamma in (0.0, 0.5, 1.0):
        expected = -2 * 0.5 * n * (math.log(np.linalg.det(Theta)) - tr) + edges * math.log(n) + 4 * gamma * edges * math.log(p)
        assert K.ebic(R, Theta, n, gamma) == pytest.approx(expected, rel=1e-9)


def test_expected_threshold_edges_match_scipy_pair_by_pair():
    Sg, T = K.true_cov("kette", 40), K.true_edges("kette", 40)
    for n in (50, 200, 2000):
        e = K.expected_threshold_edges(Sg, n, 0.3, T)
        total = right = 0.0
        for i in range(40):
            for j in range(i + 1, 40):
                mu, s, c = math.atanh(Sg[i, j]), 1 / math.sqrt(n - 3), math.atanh(0.3)
                pr = stats.norm.sf(c, loc=mu, scale=s) + stats.norm.cdf(-c, loc=mu, scale=s)
                total += pr
                right += pr if T[i, j] else 0.0
        assert e["edges"] == pytest.approx(total, abs=1e-9) and e["right"] == pytest.approx(right, abs=1e-9) and e["wrong"] == pytest.approx(total - right, abs=1e-9)


def _mean_edges(truth, n, p, runs, seed, make):
    rng = VRng(seed)
    out = []
    for _ in range(runs):
        R = K.sample_corr(K.draw(truth, n, p, rng))
        out.append(make(R))
    return np.mean(out, axis=0)


@pytest.mark.parametrize("truth, n", [("kette", 200), ("gitter", 200), ("zufall", 100)])
def test_expected_threshold_edges_match_a_simulation(truth, n):
    T = K.true_edges(truth, 40)
    exp = K.expected_threshold_edges(K.true_cov(truth, 40), n, 0.3, T)
    sim = _mean_edges(truth, n, 40, 300, 11, lambda R: [K.score(K.threshold_net(R, 0.3), T)[k] for k in ("edges", "right", "wrong")])
    assert sim[0] == pytest.approx(exp["edges"], abs=1.0) and sim[1] == pytest.approx(exp["right"], abs=0.8) and sim[2] == pytest.approx(exp["wrong"], abs=0.8)


@pytest.mark.parametrize("truth, n", [("kette", 100), ("kette", 200), ("gitter", 500), ("zufall", 200)])
def test_expected_partial_edges_match_a_simulation(truth, n):
    T = K.true_edges(truth, 40)
    exp = K.expected_partial_edges(K.true_cov(truth, 40), n, 0.05, T)
    sim = _mean_edges(truth, n, 40, 300, 12, lambda R: [K.score(K.partial_net(R, n, 0.05)[0], T)[k] for k in ("edges", "right", "wrong")])
    assert sim[0] == pytest.approx(exp["edges"], abs=0.9) and sim[1] == pytest.approx(exp["right"], abs=0.9)
    assert sim[2] <= 0.2 and exp["wrong"] == pytest.approx(0.05, abs=0.005)               # Bonferroni: höchstens α falsche Kanten je Datensatz


def test_the_factor_world_has_nearly_all_pairs_in_the_threshold_net_and_none_in_the_partial_net():
    T = K.true_edges("faktor", 40)
    exp = K.expected_threshold_edges(K.true_cov("faktor", 40), 500, 0.3, T)
    assert exp["edges"] == pytest.approx(728.08, abs=0.01) and exp["right"] == 0.0
    sim = _mean_edges("faktor", 500, 40, 100, 13, lambda R: [K.edge_count(K.threshold_net(R, 0.3)), K.edge_count(K.partial_net(R, 500, 0.05)[0])])
    assert sim[0] == pytest.approx(exp["edges"], abs=25) and sim[1] <= 0.5
    big = K.expected_partial_edges(K.true_cov("faktor", 40), 50000, 0.05, T)
    assert big["edges"] > 700 and K.expected_partial_edges(K.true_cov("faktor", 40), 500, 0.05, T)["edges"] < 0.5


def test_expected_partial_edges_do_not_exist_for_n_at_most_p_plus_one():
    T = K.true_edges("kette", 40)
    assert K.expected_partial_edges(K.true_cov("kette", 40), 41, 0.05, T) is None
    assert K.expected_partial_edges(K.true_cov("kette", 40), 42, 0.05, T) is not None
