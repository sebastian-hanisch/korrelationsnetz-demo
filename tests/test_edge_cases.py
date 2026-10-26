"""Randfälle (vor dem ersten Mutationstest): kleinste Welt, n ≤ p + 1, inklusive Grenzen von Schwelle und Freiheitsgraden, λ am oberen Ende, leere und volle Netze."""

import numpy as np
import pytest

import knz_algorithm as K
import knz_constants as C
from prs_scenario import VRng


@pytest.mark.parametrize("truth", list(C.TRUTHS))
def test_every_world_runs_at_the_smallest_size(truth):
    p = min(C.P_OPTIONS)
    X = K.draw(truth, 30, p, VRng(2))
    R = K.sample_corr(X)
    assert X.shape == (30, p) and np.allclose(np.diag(R), 1.0)
    assert K.true_edges(truth, p).shape == (p, p) and np.all(np.linalg.eigvalsh(K.true_cov(truth, p)) > 0)
    lams, thetas = K.glasso_path(R, k=4)
    assert len(thetas) == 4 and K.edge_count(K.glasso_net(thetas[0])) == 0


def test_the_partial_net_needs_at_least_two_more_weeks_than_variables():
    R = K.sample_corr(K.draw("kette", 42, 40, VRng(1)))
    assert K.partial_net(R, 41, 0.05) is None and K.partial_net(R, 42, 0.05) is not None      # df = n − p − 1: 0 gibt es nicht, 1 schon


def test_fewer_weeks_than_variables_still_gives_a_threshold_and_a_glasso_net():
    X = K.draw("kette", 20, 40, VRng(1))
    R = K.sample_corr(X)
    assert np.linalg.matrix_rank(R) <= 19
    assert K.edge_count(K.threshold_net(R, 0.3)) > 39                                      # Rauschen: weit mehr Kanten als echte
    Theta, _W, _B = K.glasso(R, 0.3)
    assert np.all(np.linalg.eigvalsh(Theta) > 0) and K.kkt_violation(R, Theta, 0.3) < 1e-4


def test_threshold_extremes_give_the_empty_and_the_complete_net():
    R = K.sample_corr(K.draw("kette", 100, 10, VRng(2)))
    assert K.edge_count(K.threshold_net(R, 1.0)) == 0
    assert K.edge_count(K.threshold_net(R, 0.0)) == 45


def test_lambda_at_or_above_the_largest_correlation_gives_the_empty_net_and_below_it_an_edge():
    R = K.sample_corr(K.draw("kette", 100, 10, VRng(2)))
    lm = K.lambda_max(R)
    assert K.edge_count(K.glasso_net(K.glasso(R, lm)[0])) == 0 and K.edge_count(K.glasso_net(K.glasso(R, lm * 1.5)[0])) == 0
    assert K.edge_count(K.glasso_net(K.glasso(R, lm * 0.98, tol=1e-10)[0])) >= 1


def test_the_path_grid_runs_from_lambda_max_down_to_the_ratio():
    R = K.sample_corr(K.draw("gitter", 100, 10, VRng(2)))
    g = K.lambda_grid(R)
    assert len(g) == C.PATH_K and g[0] == pytest.approx(K.lambda_max(R)) and g[-1] == pytest.approx(C.PATH_RATIO * K.lambda_max(R)) and np.all(np.diff(g) < 0)


def test_transitivity_of_a_net_without_connected_triples_is_zero():
    A = np.zeros((5, 5), dtype=bool)
    A[0, 1] = A[1, 0] = A[2, 3] = A[3, 2] = True
    assert K.transitivity(A) == 0.0 and K.transitivity(np.zeros((5, 5), dtype=bool)) == 0.0


def test_the_expected_edge_counts_stay_inside_the_possible_range():
    for truth in C.TRUTHS:
        T = K.true_edges(truth, 40)
        for n in (20, 30, 45, 60, 100, 2000, 50000):
            e = K.expected_threshold_edges(K.true_cov(truth, 40), n, 0.3, T)
            assert 0.0 <= e["right"] <= K.edge_count(T) + 1e-9 and 0.0 <= e["edges"] <= 780.0 + 1e-9 and e["edges"] == pytest.approx(e["right"] + e["wrong"], abs=1e-9)
        assert K.expected_partial_edges(K.true_cov(truth, 40), 41, 0.05, T) is None


def test_a_threshold_of_exactly_the_correlation_is_not_an_edge_but_just_below_is():
    R = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert K.edge_count(K.threshold_net(R, 0.3)) == 0 and K.edge_count(K.threshold_net(R, 0.2999)) == 1
    assert K.edge_count(K.threshold_net(-R + 2 * np.eye(2), 0.2999)) == 1                  # negative Korrelation zählt mit ihrem Betrag


def test_glasso_does_at_least_one_sweep_and_one_inner_round_when_asked_for_exactly_one():
    R = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.1], [0.2, 0.1, 1.0]])
    assert K.edge_count(K.glasso_net(K.glasso(R, 0.15, max_outer=1, max_inner=1)[0])) >= 1
    assert K.edge_count(K.glasso_net(K.glasso(R, 0.15, max_outer=1)[0])) >= 1


def test_the_edge_threshold_of_the_precision_matrix_is_strict():
    eps = C.EDGE_EPS
    assert not K.glasso_net(np.array([[1.0, eps], [eps, 1.0]]))[0, 1] and K.glasso_net(np.array([[1.0, 2 * eps], [2 * eps, 1.0]]))[0, 1]


def test_kkt_violation_reports_each_kind_of_violation_separately():
    S = np.array([[1.0, 0.6], [0.6, 1.0]])
    def theta(w12, w11=1.0):
        return np.linalg.inv(np.array([[w11, w12], [w12, 1.0]]))
    assert K.kkt_violation(S, theta(0.4), 0.2) < 1e-9                                           # optimal: W₁₂ = r − λ
    assert K.kkt_violation(S, np.eye(2), 0.2) == pytest.approx(0.4, abs=1e-12)                  # Betrag der Abweichung über λ: |0 − 0,6| − 0,2
    assert K.kkt_violation(S, theta(0.55), 0.2) == pytest.approx(0.15, abs=1e-12)               # Vorzeichenbedingung: |−0,05 + 0,2|, der Betrag ist ≤ λ
    assert K.kkt_violation(np.eye(2), theta(0.0, w11=1.2), 0.5) == pytest.approx(0.2, abs=1e-12)  # Diagonale W₁₁ ≠ S₁₁
    eps = C.EDGE_EPS
    assert K.kkt_violation(np.eye(2), np.array([[1.0, eps], [eps, 1.0]]), 0.5) < 1e-6          # Eintrag genau an der Kantenschwelle zählt nicht als Kante
