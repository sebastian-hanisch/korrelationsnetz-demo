"""Eigenschaften: Symmetrie, Verschachtelung der Netze, Invarianzen, Reproduzierbarkeit, Monotonie der exakten Erwartungen."""

import numpy as np
import pytest

import knz_algorithm as K
import knz_constants as C
from prs_scenario import VRng


def _R(truth="gitter", n=200, p=20, seed=3):
    return K.sample_corr(K.draw(truth, n, p, VRng(seed)))


def test_all_nets_are_symmetric_without_loops():
    R = _R()
    A = [K.threshold_net(R, 0.2), K.partial_net(R, 200, 0.05)[0], K.rank_net(R, 10), K.glasso_net(K.glasso(R, 0.1)[0])]
    for a in A:
        assert np.array_equal(a, a.T) and not a.diagonal().any()


def test_threshold_nets_are_nested_in_tau_and_rank_nets_in_k():
    R = _R()
    taus = [0.1, 0.2, 0.3, 0.4]
    nets = [K.threshold_net(R, t) for t in taus]
    for big, small in zip(nets, nets[1:]):
        assert not (small & ~big).any()
    ranks = [K.rank_net(R, k) for k in (5, 10, 20)]
    for small, big in zip(ranks, ranks[1:]):
        assert not (small & ~big).any()


def test_a_smaller_alpha_gives_a_subset_of_the_partial_net():
    R = _R("kette", 150, 20)
    a05, a01 = K.partial_net(R, 150, 0.05)[0], K.partial_net(R, 150, 0.01)[0]
    assert not (a01 & ~a05).any() and K.edge_count(a01) <= K.edge_count(a05)


def test_nets_follow_a_permutation_of_the_variables():
    X = K.draw("zufall", 200, 20, VRng(5))
    perm = np.random.default_rng(1).permutation(20)
    R, Rp = K.sample_corr(X), K.sample_corr(X[:, perm])
    assert np.array_equal(K.threshold_net(Rp, 0.2), K.threshold_net(R, 0.2)[np.ix_(perm, perm)])
    assert np.array_equal(K.partial_net(Rp, 200, 0.05)[0], K.partial_net(R, 200, 0.05)[0][np.ix_(perm, perm)])
    assert np.array_equal(K.glasso_net(K.glasso(Rp, 0.1)[0]), K.glasso_net(K.glasso(R, 0.1)[0])[np.ix_(perm, perm)])


def test_the_correlation_matrix_and_the_nets_do_not_change_when_columns_are_shifted_and_scaled():
    X = K.draw("kette", 150, 15, VRng(8))
    Y = X * np.linspace(0.5, 4.0, 15) + np.linspace(-3, 3, 15)
    assert np.allclose(K.sample_corr(X), K.sample_corr(Y), atol=1e-12)
    assert np.array_equal(K.partial_net(K.sample_corr(X), 150, 0.05)[0], K.partial_net(K.sample_corr(Y), 150, 0.05)[0])


def test_the_same_seed_gives_the_same_data_and_the_same_nets():
    a, b = _R(seed=5), _R(seed=5)
    assert np.array_equal(a, b) and not np.array_equal(a, _R(seed=6))
    assert np.array_equal(K.glasso(a, 0.1)[0], K.glasso(b, 0.1)[0])


def test_f1_stays_in_the_unit_interval_and_rows_add_up():
    R, T = _R("zufall", 150, 20), K.true_edges("zufall", 20)
    for A in (K.threshold_net(R, 0.1), K.threshold_net(R, 0.5), K.rank_net(R, 7), K.partial_net(R, 150, 0.05)[0]):
        s = K.score(A, T)
        assert 0.0 <= s["f1"] <= 1.0 and s["edges"] == s["right"] + s["wrong"] and s["right"] + s["missing"] == K.edge_count(T)


def test_the_expected_threshold_edges_fall_with_tau_and_the_expected_partial_edges_rise_with_n():
    T = K.true_edges("kette", 40)
    Sg = K.true_cov("kette", 40)
    by_tau = [K.expected_threshold_edges(Sg, 500, tau, T)["edges"] for tau in (0.1, 0.2, 0.3, 0.4, 0.5)]
    assert by_tau == sorted(by_tau, reverse=True)
    by_n = [K.expected_partial_edges(Sg, n, 0.05, T)["right"] for n in (50, 100, 200, 500, 2000)]
    assert by_n == sorted(by_n) and by_n[-1] == pytest.approx(39.0, abs=0.01)


def test_the_expected_threshold_edges_in_the_grid_world_fall_when_n_grows():
    T, Sg = K.true_edges("gitter", 40), K.true_cov("gitter", 40)
    right = [K.expected_threshold_edges(Sg, n, 0.3, T)["right"] for n in (500, 1000, 2000, 5000)]
    assert right == sorted(right, reverse=True) and right[-1] < 0.1


def test_worlds_are_valid_for_every_slider_value_of_p():
    for p in C.P_OPTIONS:
        for truth in C.TRUTHS:
            assert np.all(np.linalg.eigvalsh(K.true_cov(truth, p)) > 0), (truth, p)
