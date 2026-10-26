"""Einheiten des Kerns an Mini-Instanzen von Hand (DEMO-PLAYBOOK §1b): Teilkorrelation, Netze, Transitivität, Güte, Graphical Lasso, Wahrheiten."""

import math

import numpy as np
import pytest

import knz_algorithm as K
import knz_constants as C
from prs_scenario import VRng


# --- Teilkorrelation ---------------------------------------------------------------------------------------------------------------------------

def test_partial_correlation_of_the_three_chain(chain3):
    Pc = K.partial_corr(chain3)
    assert Pc[0, 2] == pytest.approx(0.0, abs=1e-12)
    assert Pc[0, 1] == pytest.approx(0.5 / math.sqrt(1.25), abs=1e-12) and Pc[1, 2] == pytest.approx(Pc[0, 1], abs=1e-12)
    P = np.linalg.inv(chain3)
    assert P[0, 2] == pytest.approx(0.0, abs=1e-12) and P[0, 0] == pytest.approx(1 / 0.75, abs=1e-12) and P[1, 1] == pytest.approx(1.25 / 0.75, abs=1e-12)


def test_partial_correlation_of_the_three_factor_variables(factor3):
    Pc = K.partial_corr(factor3)
    assert Pc[0, 1] == pytest.approx((0.36 - 0.36**2) / (1 - 0.36**2), abs=1e-12) and Pc[0, 1] == pytest.approx(0.2647058823529412, abs=1e-12)
    assert np.allclose(np.diag(Pc), 1.0) and np.allclose(Pc, Pc.T)


def test_partial_degrees_of_freedom_are_n_minus_p_minus_one():
    assert K.partial_df(100, 40) == 59 and K.partial_df(41, 40) == 0 and K.partial_df(30, 40) == -11


def test_bonferroni_critical_value_for_three_pairs():
    assert K.bonferroni_z(3, 0.05) == pytest.approx(2.3940, abs=5e-4)       # Φ⁻¹(1 − 0,05 / 6)
    assert K.bonferroni_z(40, 0.01) > K.bonferroni_z(40, 0.05) > K.bonferroni_z(10, 0.05)


# --- Netze -------------------------------------------------------------------------------------------------------------------------------------

def test_threshold_net_is_strict_and_has_no_diagonal(chain3):
    A = K.threshold_net(chain3, 0.5)
    assert not A.any()                                                      # |r| = 0,5 ist keine Kante bei τ = 0,5
    A = K.threshold_net(chain3, 0.4999)
    assert A[0, 1] and A[1, 2] and not A[0, 2] and not A.diagonal().any() and K.edge_count(A) == 2


def test_partial_net_of_the_chain_has_no_edge_between_the_ends():
    R = np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])
    A, Pc = K.partial_net(R, 400, 0.05)
    assert A[0, 1] and A[1, 2] and not A[0, 2] and Pc[0, 2] == pytest.approx(0.0, abs=1e-12)


def test_partial_net_is_not_defined_for_n_at_most_p_plus_one():
    R = np.eye(5)
    assert K.partial_net(R, 6, 0.05) is None and K.partial_net(R, 3, 0.05) is None
    assert K.partial_net(R, 7, 0.05) is not None                            # df = 1


def test_rank_net_takes_the_k_largest_absolute_correlations(chain3):
    A = K.rank_net(chain3, 2)
    assert K.edge_count(A) == 2 and A[0, 1] and A[1, 2] and not A[0, 2]
    assert K.edge_count(K.rank_net(chain3, 0)) == 0 and K.edge_count(K.rank_net(chain3, 3)) == 3
    neg = np.array([[1.0, -0.9, 0.1], [-0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])
    assert K.rank_net(neg, 1)[0, 1]                                         # Betrag, nicht Vorzeichen


@pytest.mark.parametrize("edges, expected", [
    ([(0, 1), (1, 2), (0, 2)], 1.0),                                        # Dreieck
    ([(0, 1), (1, 2)], 0.0),                                                # Pfad
    ([(0, 1), (1, 2), (0, 2), (2, 3)], 0.6),                                # Dreieck mit Anhang: 3 · 1 / (1 + 1 + 3 + 0)
    ([(0, 1), (0, 2), (0, 3)], 0.0),                                        # Stern
    ([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], 1.0),                # K4
    ([], 0.0),
])
def test_transitivity_by_hand(edges, expected):
    A = np.zeros((4, 4), dtype=bool)
    for i, j in edges:
        A[i, j] = A[j, i] = True
    assert K.transitivity(A) == pytest.approx(expected, abs=1e-12)


def test_score_counts_right_wrong_and_missing():
    T = np.zeros((4, 4), dtype=bool)
    A = np.zeros((4, 4), dtype=bool)
    for i, j in [(0, 1), (1, 2), (2, 3)]:
        T[i, j] = T[j, i] = True
    for i, j in [(0, 1), (1, 2), (0, 3), (1, 3)]:
        A[i, j] = A[j, i] = True
    s = K.score(A, T)
    assert (s["edges"], s["right"], s["wrong"], s["missing"]) == (4, 2, 2, 1)
    assert s["f1"] == pytest.approx(2 * 2 / (2 * 2 + 2 + 1), abs=1e-12)


def test_score_of_nothing_against_nothing_is_one_and_of_false_edges_alone_zero():
    T = np.zeros((3, 3), dtype=bool)
    A = np.zeros((3, 3), dtype=bool)
    assert K.score(A, T)["f1"] == 1.0
    A[0, 1] = A[1, 0] = True
    assert K.score(A, T)["f1"] == 0.0


# --- Graphical Lasso ---------------------------------------------------------------------------------------------------------------------------

def test_glasso_with_two_variables_by_hand(two_vars):
    Theta, W, _B = K.glasso(two_vars, 0.2, tol=1e-12)
    assert W[0, 1] == pytest.approx(0.4, abs=1e-9) and W[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert Theta[0, 0] == pytest.approx(1 / 0.84, abs=1e-8) and Theta[0, 1] == pytest.approx(-0.4 / 0.84, abs=1e-8)
    assert K.partial_from_precision(Theta)[0, 1] == pytest.approx(0.4, abs=1e-8)


def test_glasso_at_lambda_max_is_the_identity_and_just_below_it_has_the_edge(two_vars):
    assert K.lambda_max(two_vars) == pytest.approx(0.6, abs=1e-12)
    Theta, _W, _B = K.glasso(two_vars, 0.6)
    assert np.allclose(Theta, np.eye(2), atol=1e-12) and not K.glasso_net(Theta).any()
    Theta, _W, _B = K.glasso(two_vars, 0.59, tol=1e-12)
    assert K.glasso_net(Theta)[0, 1]
    assert np.allclose(K.glasso(np.eye(4), 0.01)[0], np.eye(4))


def test_glasso_satisfies_the_optimality_conditions_on_a_small_world():
    X = K.draw("kette", 200, 8, VRng(5))
    R = K.sample_corr(X)
    for lam in (0.05, 0.15, 0.3):
        Theta, _W, _B = K.glasso(R, lam, tol=1e-9)
        assert K.kkt_violation(R, Theta, lam) < 1e-5


def test_path_edge_count_never_grows_with_lambda_up_to_the_first_nonempty_net():
    R = K.sample_corr(K.draw("kette", 300, 12, VRng(5)))
    lams, thetas = K.glasso_path(R, k=8)
    counts = [K.edge_count(K.glasso_net(t)) for t in thetas]
    assert lams[0] == pytest.approx(K.lambda_max(R)) and counts[0] == 0 and lams[-1] == pytest.approx(0.05 * K.lambda_max(R))
    assert counts == sorted(counts) and counts[-1] > counts[0]


def test_ebic_penalizes_edges_with_gamma():
    R = np.array([[1.0, 0.6], [0.6, 1.0]])
    Theta, _W, _B = K.glasso(R, 0.2, tol=1e-12)
    base = K.ebic(R, Theta, 100, 0.0)
    assert K.ebic(R, Theta, 100, 0.5) == pytest.approx(base + 4 * 0.5 * 1 * math.log(2), abs=1e-9)        # eine Kante, p = 2
    ll = 0.5 * 100 * (math.log(np.linalg.det(Theta)) - np.trace(R @ Theta))
    assert base == pytest.approx(-2 * ll + 1 * math.log(100), abs=1e-9)


def test_select_ebic_and_select_best_pick_the_expected_path_values():
    R = K.sample_corr(K.draw("kette", 500, 12, VRng(9)))
    lams, thetas = K.glasso_path(R, k=10)
    T = K.true_edges("kette", 12)
    i_best = K.select_best(thetas, T)
    assert K.score(K.glasso_net(thetas[i_best]), T)["f1"] == max(K.score(K.glasso_net(t), T)["f1"] for t in thetas)
    i_e = K.select_ebic(R, thetas, 500, 0.5)
    assert K.ebic(R, thetas[i_e], 500, 0.5) == min(K.ebic(R, t, 500, 0.5) for t in thetas)


# --- Wahrheiten --------------------------------------------------------------------------------------------------------------------------------

def test_the_worlds_have_the_documented_edge_counts():
    assert [K.edge_count(K.true_edges(t, 40)) for t in C.TRUTHS] == [39, 67, 43, 0]
    assert K.edge_count(K.true_edges("gitter", 40)) == 5 * 7 + 4 * 8 and K.edge_count(K.true_edges("kette", 10)) == 9


def test_chain_precision_is_tridiagonal_with_partial_correlation_0_4_inside():
    P = K.true_precision("kette", 10)
    assert np.allclose(P, np.linalg.inv(K.true_cov("kette", 10)), atol=1e-12)
    assert np.all(P[np.abs(np.subtract.outer(range(10), range(10))) > 1] == 0.0)
    Pc = K.true_partial_corr("kette", 10)
    assert Pc[3, 4] == pytest.approx(0.4, abs=1e-12) and Pc[0, 1] == pytest.approx(0.5 / math.sqrt(1.25), abs=1e-12)     # am Rand 0,4472


def test_precision_is_zero_exactly_outside_the_edges_for_the_three_sparse_worlds():
    for truth in ("kette", "gitter", "zufall"):
        for p in (10, 20, 40):
            P, T = K.true_precision(truth, p), K.true_edges(truth, p)
            off = ~np.eye(p, dtype=bool)
            Q = np.linalg.inv(K.true_cov(truth, p))                          # Kovarianz mit Diagonale 1: andere Skalierung, gleiche Teilkorrelationen und Nullen
            assert np.allclose(K.partial_from_precision(P), K.partial_from_precision(Q), atol=1e-10) and np.abs(Q[off & ~T]).max() < 1e-10, (truth, p)
            assert np.abs(P[off & ~T]).max() < 1e-10 and np.all(np.abs(P[off & T]) > 0.1), (truth, p)
            assert np.all(np.linalg.eigvalsh(P) > 0)


def test_factor_world_has_no_direct_edge_but_partial_correlation_0_0245():
    Pc = K.true_partial_corr("faktor", 40)
    off = ~np.eye(40, dtype=bool)
    assert not K.true_edges("faktor", 40).any() and Pc[off].min() == pytest.approx(Pc[off].max(), abs=1e-12) and Pc[0, 1] == pytest.approx(0.0245, abs=5e-5)
    assert K.true_cov("faktor", 40)[0, 1] == C.FACTOR_RHO


def test_the_grid_world_partial_correlation_is_the_coupling_and_the_random_world_has_both_signs():
    Pc = K.true_partial_corr("gitter", 40)
    assert Pc[0, 1] == pytest.approx(C.GRID_COUPLING, abs=1e-12)
    pz = K.true_partial_corr("zufall", 40)[K.true_edges("zufall", 40)]
    assert (pz > 0).any() and (pz < 0).any() and np.abs(pz).min() > 0.17 and np.abs(pz).max() < 0.4


def test_draw_is_reproducible_and_standardized_on_average():
    X1, X2 = K.draw("zufall", 50, 40, VRng(3)), K.draw("zufall", 50, 40, VRng(3))
    assert np.array_equal(X1, X2) and not np.array_equal(X1, K.draw("zufall", 50, 40, VRng(4)))
    big = K.draw("kette", 40000, 6, VRng(1))
    assert np.abs(big.std(axis=0) - 1).max() < 0.02 and abs(np.corrcoef(big.T)[0, 1] - 0.5) < 0.02


def test_sample_corr_has_unit_diagonal_and_is_symmetric():
    R = K.sample_corr(K.draw("gitter", 80, 20, VRng(2)))
    assert np.allclose(np.diag(R), 1.0) and np.allclose(R, R.T)


def test_unknown_world_raises():
    for fn in (K.true_precision, K.true_cov, K.true_edges):
        with pytest.raises(ValueError):
            fn("stern", 10)
