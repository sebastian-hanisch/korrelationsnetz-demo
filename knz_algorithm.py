"""Kern: vier Welten mit bekannter Struktur der direkten Zusammenhänge (Kette, Gitter, Zufallsnetz, Faktor), Stichprobenmatrix, Teilkorrelation aus der Inversen, drei Netze
(Schwelle, Teilkorrelation mit Fisher-z und Bonferroni, Graphical Lasso), Rangfolge, Transitivität, Güte gegen die Wahrheit, EBIC und die exakten Erwartungen der beiden einfachen Netze.

Die direkten Zusammenhänge stehen in der Präzisionsmatrix P = Σ⁻¹: die Teilkorrelation zweier Kennzahlen gegeben alle übrigen ist −P_ij / √(P_ii P_jj), und genau dort, wo P null ist,
gibt es keine Kante. Ein Netz heißt hier immer ungerichtet; Ursachen und Richtungen gehören der Kausal-Linie. Zufall und Verteilungsfunktionen kommen aus den wörtlichen Kopien von
pearson-demo (prs_*.py). Ohne scipy zur Laufzeit."""

import math

import numpy as np

import knz_constants as C
from prs_algorithm import norm_ppf
from prs_scenario import VRng


# --- Welten ----------------------------------------------------------------------------------------------------------------------------------

def _adjacency_grid(p: int) -> np.ndarray:
    rows = C.GRID_ROWS
    cols = p // rows
    A = np.zeros((p, p), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            if c + 1 < cols:
                A[i, i + 1] = A[i + 1, i] = True
            if r + 1 < rows:
                A[i, i + cols] = A[i + cols, i] = True
    return A


def _random_weights(p: int) -> np.ndarray:
    """Fester Zufallsgraph: je Paar (i < j) drei Zufallszahlen aus dem Strom mit Seed C.RANDOM_SEED (Kante, Stärke, Vorzeichen); Gewichte ±(0,2 bis 0,4)."""
    iu = np.triu_indices(p, 1)
    u = VRng(C.RANDOM_SEED).uniform((len(iu[0]), 3))
    W = np.zeros((p, p))
    on = u[:, 0] < C.RANDOM_PROB
    w = (C.RANDOM_W_LO + C.RANDOM_W_SPAN * u[:, 1]) * np.where(u[:, 2] < 0.5, 1.0, -1.0)
    W[iu[0][on], iu[1][on]] = w[on]
    return W + W.T


def true_precision(truth: str, p: int) -> np.ndarray:
    """Wahre Präzisionsmatrix P = Σ⁻¹ (bei Eigenschaften der Kovarianz mit Diagonale 1 bis auf die Skalierung). Einträge außerhalb der Kanten sind exakt 0."""
    if truth == "kette":
        phi = C.CHAIN_PHI
        P = np.zeros((p, p))
        idx = np.arange(p)
        P[idx, idx] = 1.0 + phi * phi
        P[0, 0] = P[p - 1, p - 1] = 1.0
        P[idx[:-1], idx[:-1] + 1] = P[idx[:-1] + 1, idx[:-1]] = -phi
        return P / (1.0 - phi * phi)
    if truth == "gitter":
        return np.eye(p) - C.GRID_COUPLING * _adjacency_grid(p)
    if truth == "zufall":
        W = _random_weights(p)
        return np.diag(np.maximum(1.0, np.abs(W).sum(axis=1) + C.RANDOM_DIAG_EXTRA)) - W
    if truth == "faktor":
        return np.linalg.inv(true_cov(truth, p))
    raise ValueError(f"unbekannte Welt: {truth}")


def true_cov(truth: str, p: int) -> np.ndarray:
    """Wahre Kovarianz mit Diagonale 1 (die Daten sind standardisiert)."""
    if truth == "kette":
        idx = np.arange(p)
        return C.CHAIN_PHI ** np.abs(idx[:, None] - idx[None, :])
    if truth == "faktor":
        R = np.full((p, p), C.FACTOR_RHO)
        np.fill_diagonal(R, 1.0)
        return R
    if truth in ("gitter", "zufall"):
        S = np.linalg.inv(true_precision(truth, p))
        d = np.sqrt(np.diag(S))
        return S / d[:, None] / d[None, :]
    raise ValueError(f"unbekannte Welt: {truth}")


def true_edges(truth: str, p: int) -> np.ndarray:
    """Kanten der Wahrheit (bool, symmetrisch, ohne Diagonale). Faktor: keine direkte Kante (alles hängt am verborgenen Faktor)."""
    if truth == "kette":
        A = np.zeros((p, p), dtype=bool)
        i = np.arange(p - 1)
        A[i, i + 1] = A[i + 1, i] = True
        return A
    if truth == "gitter":
        return _adjacency_grid(p)
    if truth == "zufall":
        return _random_weights(p) != 0.0
    if truth == "faktor":
        return np.zeros((p, p), dtype=bool)
    raise ValueError(f"unbekannte Welt: {truth}")


def partial_from_precision(P: np.ndarray) -> np.ndarray:
    """Teilkorrelationen −P_ij / √(P_ii P_jj), Diagonale 1."""
    d = np.sqrt(np.diag(P))
    out = -P / d[:, None] / d[None, :]
    np.fill_diagonal(out, 1.0)
    return out


def true_partial_corr(truth: str, p: int) -> np.ndarray:
    return partial_from_precision(true_precision(truth, p))


def draw(truth: str, n: int, p: int, rng: VRng) -> np.ndarray:
    """n Wochen mit p Kennzahlen aus der Normalverteilung der Welt (Kovarianz mit Diagonale 1)."""
    L = np.linalg.cholesky(true_cov(truth, p))
    return rng.normal((n, p)) @ L.T


# --- Stichprobe und Netze ----------------------------------------------------------------------------------------------------------------------

def sample_corr(X: np.ndarray) -> np.ndarray:
    """Stichprobenkorrelationsmatrix (Diagonale 1)."""
    Xc = X - X.mean(axis=0)
    Xc = Xc / np.linalg.norm(Xc, axis=0)
    return Xc.T @ Xc


def partial_df(n: int, p: int) -> int:
    """Freiheitsgrade des Fisher-z der Teilkorrelation gegeben p − 2 weitere Kennzahlen: n − (p − 2) − 3 = n − p − 1."""
    return n - p - 1


def partial_corr(R: np.ndarray) -> np.ndarray:
    """Teilkorrelationen aus der Inversen der Stichprobenmatrix; verlangt eine invertierbare Matrix (n > p)."""
    return partial_from_precision(np.linalg.inv(R))


def zero_diagonal(A: np.ndarray) -> np.ndarray:
    A = A.copy()
    np.fill_diagonal(A, False)
    return A


def threshold_net(R: np.ndarray, tau: float) -> np.ndarray:
    """Schwellennetz: Kante, wenn |r| > τ (strikt)."""
    return zero_diagonal(np.abs(R) > tau)


def bonferroni_z(p: int, alpha: float) -> float:
    """Kritischer Betrag der Fisher-z-Statistik bei Bonferroni über alle p(p−1)/2 Paare (zweiseitig)."""
    m = p * (p - 1) // 2
    return norm_ppf(1.0 - alpha / (2.0 * m))


def partial_net(R: np.ndarray, n: int, alpha: float):
    """Teilkorrelationsnetz: Fisher-z mit n − p − 1 Freiheitsgraden, Bonferroni über alle Paare. Gibt (Netz, Teilkorrelationen) zurück; None, wenn n ≤ p + 1 (die Stichprobenmatrix hat dann
    keine Inverse, die Rechnung gibt es nicht)."""
    p = R.shape[0]
    df = partial_df(n, p)
    if df <= 0:
        return None
    Pc = partial_corr(R)
    stat = np.arctanh(np.clip(Pc, -0.999999, 0.999999)) * math.sqrt(df)
    return zero_diagonal(np.abs(stat) > bonferroni_z(p, alpha)), Pc


def rank_net(R: np.ndarray, k: int) -> np.ndarray:
    """Die k Paare mit dem größten |r| (bei Gleichstand die zuerst genannten): der Vergleich 'kennt die Zahl der echten Kanten'."""
    p = R.shape[0]
    iu = np.triu_indices(p, 1)
    order = np.argsort(-np.abs(R[iu]), kind="stable")[: max(int(k), 0)]
    A = np.zeros((p, p), dtype=bool)
    A[iu[0][order], iu[1][order]] = True
    return A | A.T


def edge_count(A: np.ndarray) -> int:
    return int(np.triu(A, 1).sum())


def transitivity(A: np.ndarray) -> float:
    """Globaler Clusterkoeffizient: 3 · Dreiecke / verbundene Dreier (0, wenn es keine verbundenen Dreier gibt)."""
    M = A.astype(float)
    deg = M.sum(axis=1)
    triples = float((deg * (deg - 1.0)).sum())
    return float(np.trace(M @ M @ M) / triples) if triples > 0 else 0.0


def score(A: np.ndarray, T: np.ndarray) -> dict:
    """Güte gegen die Wahrheit: richtig, falsch, fehlend und F1 = 2·richtig / (2·richtig + falsch + fehlend); ohne echte und gefundene Kanten gilt F1 = 1 (richtig leer)."""
    a, t = np.triu(A, 1), np.triu(T, 1)
    right = int((a & t).sum())
    wrong = int((a & ~t).sum())
    missing = int((~a & t).sum())
    denom = 2 * right + wrong + missing
    return {"edges": right + wrong, "right": right, "wrong": wrong, "missing": missing, "f1": 1.0 if denom == 0 else 2.0 * right / denom}


# --- Graphical Lasso -----------------------------------------------------------------------------------------------------------------------

def glasso(S: np.ndarray, lam: float, W0=None, B0=None, tol: float = 1e-6, max_outer: int = 200, max_inner: int = 200):
    """Graphical Lasso, Blockkoordinatenabstieg (Friedman, Hastie, Tibshirani 2008) auf der Korrelationsmatrix S: minimiert −log det Θ + tr(SΘ) + λ Σ_{i≠j} |Θ_ij|.
    Die Diagonale wird nicht bestraft (W_jj = S_jj). Anwärmen mit W0, B0 des vorigen λ. Gibt (Θ, W, B) zurück."""
    p = S.shape[0]
    W = S.copy() if W0 is None else W0.copy()
    B = np.zeros((p, p)) if B0 is None else B0.copy()
    scale = max(float(np.abs(S - np.diag(np.diag(S))).mean()), 1e-12)
    for _ in range(max_outer):
        W_old = W.copy()
        for j in range(p):
            idx = np.arange(p) != j
            W11 = W[np.ix_(idx, idx)]
            s12 = S[idx, j]
            beta = B[idx, j].copy()
            diag = np.diag(W11)
            for _ in range(max_inner):
                b_old = beta.copy()
                for k in range(p - 1):
                    r = s12[k] - W11[k] @ beta + diag[k] * beta[k]
                    beta[k] = math.copysign(max(abs(r) - lam, 0.0), r) / diag[k]
                if np.abs(beta - b_old).max() < 1e-9:
                    break
            B[idx, j] = beta
            w12 = W11 @ beta
            W[idx, j] = w12
            W[j, idx] = w12
        if np.abs(W - W_old).mean() < tol * scale:
            break
    Theta = np.zeros((p, p))
    for j in range(p):
        idx = np.arange(p) != j
        th = 1.0 / (W[j, j] - W[idx, j] @ B[idx, j])
        Theta[j, j] = th
        Theta[idx, j] = -B[idx, j] * th
    return (Theta + Theta.T) / 2.0, W, B


def glasso_net(Theta: np.ndarray) -> np.ndarray:
    return zero_diagonal(np.abs(Theta) > C.EDGE_EPS)


def lambda_max(S: np.ndarray) -> float:
    """Kleinstes λ, bei dem der Graphical Lasso das leere Netz liefert: das größte |r|."""
    return float(np.abs(S - np.diag(np.diag(S))).max())


def lambda_grid(S: np.ndarray, k: int = C.PATH_K, ratio: float = C.PATH_RATIO) -> np.ndarray:
    return lambda_max(S) * np.exp(np.linspace(0.0, math.log(ratio), k))


def glasso_path(S: np.ndarray, k: int = C.PATH_K, ratio: float = C.PATH_RATIO):
    """Pfad der Lösungen von groß nach klein, jeweils mit den Werten des vorigen λ angewärmt. Gibt (λ-Werte, Liste der Θ) zurück."""
    lams = lambda_grid(S, k, ratio)
    thetas, W, B = [], None, None
    for lam in lams:
        Theta, W, B = glasso(S, float(lam), W, B)
        thetas.append(Theta)
    return lams, thetas


def ebic(S: np.ndarray, Theta: np.ndarray, n: int, gamma: float) -> float:
    """Erweitertes Bayes-Kriterium für Graphen (Foygel, Drton 2010) mit dem Verbundwert der (bestraften) Schätzung: −2 ℓ + E ln n + 4 γ E ln p, ℓ = n/2 (log det Θ − tr(SΘ))."""
    p = S.shape[0]
    _, logdet = np.linalg.slogdet(Theta)
    loglik = 0.5 * n * (logdet - float(np.trace(S @ Theta)))
    edges = edge_count(glasso_net(Theta))
    return -2.0 * loglik + edges * math.log(n) + 4.0 * gamma * edges * math.log(p)


def select_ebic(S: np.ndarray, thetas, n: int, gamma: float) -> int:
    """Index des Pfadwerts mit dem kleinsten EBIC (bei Gleichstand der größere λ)."""
    return int(np.argmin([ebic(S, t, n, gamma) for t in thetas]))


def select_best(thetas, T: np.ndarray) -> int:
    """Index des Pfadwerts mit dem höchsten F1 gegen die Wahrheit (bei Gleichstand der größere λ): kennt die Wahrheit, nur eine Obergrenze."""
    f1 = [score(glasso_net(t), T)["f1"] for t in thetas]
    return int(np.argmax(f1))


def kkt_violation(S: np.ndarray, Theta: np.ndarray, lam: float) -> float:
    """Größte Verletzung der Optimalitätsbedingungen des Graphical Lasso, unabhängig vom Löser: mit W = Θ⁻¹ muss W_jj = S_jj gelten, |W_ij − S_ij| ≤ λ, und wo Θ_ij ≠ 0 ist
    W_ij − S_ij = λ · sign(Θ_ij)."""
    W = np.linalg.inv(Theta)
    p = S.shape[0]
    off = ~np.eye(p, dtype=bool)
    diff = W - S
    worst = float(np.abs(np.diag(diff)).max())
    worst = max(worst, float(np.maximum(np.abs(diff[off]) - lam, 0.0).max()))
    on = off & (np.abs(Theta) > C.EDGE_EPS)
    if on.any():
        worst = max(worst, float(np.abs(diff[on] - lam * np.sign(Theta[on])).max()))
    return worst


# --- Exakte Erwartungen (Fisher-z-Näherung) ----------------------------------------------------------------------------------------------------

def _normal_sf(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.vectorize(math.erfc)(np.asarray(x, dtype=float) / math.sqrt(2.0))


def expected_threshold_edges(Sigma: np.ndarray, n: int, tau: float, T: np.ndarray) -> dict:
    """Erwartete Zahl der Kanten des Schwellennetzes: atanh r̂ ist näherungsweise normal mit Mittel atanh ρ und Varianz 1/(n − 3); das Paar ist Kante mit P(|atanh r̂| > atanh τ).
    Gibt Kanten, richtige und falsche zurück."""
    iu = np.triu_indices(Sigma.shape[0], 1)
    mu = np.arctanh(np.clip(Sigma[iu], -0.999999, 0.999999))
    s = 1.0 / math.sqrt(n - 3)
    c = math.atanh(min(tau, 0.999999))
    prob = _normal_sf((c - mu) / s) + (1.0 - _normal_sf((-c - mu) / s))
    t = T[iu]
    return {"edges": float(prob.sum()), "right": float(prob[t].sum()), "wrong": float(prob[~t].sum())}


def expected_partial_edges(Sigma: np.ndarray, n: int, alpha: float, T: np.ndarray):
    """Erwartete Zahl der Kanten des Teilkorrelationsnetzes: atanh der Stichprobenteilkorrelation ist näherungsweise normal mit Mittel atanh(wahre Teilkorrelation) und Varianz
    1/(n − p − 1); Bonferroni-Schwelle wie in partial_net. None, wenn n ≤ p + 1."""
    p = Sigma.shape[0]
    df = partial_df(n, p)
    if df <= 0:
        return None
    iu = np.triu_indices(p, 1)
    Pc = partial_from_precision(np.linalg.inv(Sigma))[iu]
    mu = np.arctanh(np.clip(Pc, -0.999999, 0.999999)) * math.sqrt(df)
    z = bonferroni_z(p, alpha)
    prob = _normal_sf(z - mu) + (1.0 - _normal_sf(-z - mu))
    t = T[iu]
    return {"edges": float(prob.sum()), "right": float(prob[t].sum()), "wrong": float(prob[~t].sum())}
