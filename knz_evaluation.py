"""Auswertung: eine Welt mit drei Netzen (Settings -> Analysis), Zellen des vorgerechneten Rasters (Messreihen), Verdikte als reine Funktionen und die Probe für den Rastertest."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import knz_algorithm as K
import knz_constants as C
from prs_scenario import VRng

GRID_FILE = Path(__file__).resolve().parent / "data" / "network_grid.json"
METHOD_NAMES = {
    "schwelle": "Schwellennetz",
    "teil": "Teilkorrelationsnetz",
    "glasso": "Graphical Lasso",
    "rank": "Rangfolge nach |r| (kennt die Zahl der echten Kanten)",
}


def de(v: float, digits: int = 2) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{v:.{digits}f}".replace(".", ",")


@dataclass(frozen=True, kw_only=True)
class Settings:
    truth: str = C.DEFAULT_TRUTH
    p: int = C.DEFAULT_P
    n: int = C.DEFAULT_N
    tau: float = C.DEFAULT_TAU
    alpha: float = C.DEFAULT_ALPHA
    rule: str = "ebic"
    lam: float = C.DEFAULT_LAM
    gamma: float = C.DEFAULT_GAMMA
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Row:
    key: str
    name: str
    edges: int
    right: int
    wrong: int
    missing: int
    f1: float
    trans: float
    expected: dict | None = None        # erwartete Kanten, richtige und falsche (nur Schwellen- und Teilkorrelationsnetz)


@dataclass
class Analysis:
    settings: Settings
    R: np.ndarray
    T: np.ndarray                        # Kanten der Wahrheit
    nets: dict                           # key -> bool-Matrix (nur vorhandene Netze)
    rows: list
    partial: np.ndarray | None           # Stichprobenteilkorrelationen (None bei n ≤ p + 1)
    lams: np.ndarray                     # Pfad des Graphical Lasso
    path_edges: list
    path_f1: list
    path_ebic: list
    i_ebic: int
    i_best: int
    lam_used: float
    trans_true: float

    def row(self, key: str):
        return next((r for r in self.rows if r.key == key), None)


def make_data(s: Settings):
    X = K.draw(s.truth, s.n, s.p, VRng(s.seed))
    return X, K.sample_corr(X)


def compute_path(truth: str, p: int, n: int, seed: int) -> dict:
    """Der Pfad des Graphical Lasso zu einem Datensatz (teuer: eine bis wenige Sekunden); hängt nicht von τ, α, γ und der Regel ab."""
    R = K.sample_corr(K.draw(truth, n, p, VRng(seed)))
    lams, thetas = K.glasso_path(R)
    return {"lams": lams, "thetas": thetas}


def _row(key: str, A: np.ndarray, T: np.ndarray, expected=None) -> Row:
    sc = K.score(A, T)
    return Row(key, METHOD_NAMES[key], sc["edges"], sc["right"], sc["wrong"], sc["missing"], sc["f1"], K.transitivity(A), expected)


def analyse(s: Settings, path: dict | None = None) -> Analysis:
    """Eine Welt, drei Netze plus Vergleichszeile. `path` ist der (gerechnete) Pfad des Graphical Lasso zu genau diesen Daten; fehlt er, wird er hier gerechnet."""
    _X, R = make_data(s)
    T = K.true_edges(s.truth, s.p)
    Sigma = K.true_cov(s.truth, s.p)
    if path is None:
        lams, thetas = K.glasso_path(R)
    else:
        lams, thetas = path["lams"], path["thetas"]
    nets = {"schwelle": K.threshold_net(R, s.tau)}
    rows = [_row("schwelle", nets["schwelle"], T, K.expected_threshold_edges(Sigma, s.n, s.tau, T))]
    pn = K.partial_net(R, s.n, s.alpha)
    partial = None
    if pn is not None:
        nets["teil"], partial = pn
        rows.append(_row("teil", nets["teil"], T, K.expected_partial_edges(Sigma, s.n, s.alpha, T)))
    i_ebic = K.select_ebic(R, thetas, s.n, s.gamma)
    i_best = K.select_best(thetas, T)
    if s.rule == "fest":
        theta, lam_used = K.glasso(R, s.lam)[0], s.lam
    else:
        i = i_ebic if s.rule == "ebic" else i_best
        theta, lam_used = thetas[i], float(lams[i])
    nets["glasso"] = K.glasso_net(theta)
    rows.append(_row("glasso", nets["glasso"], T))
    nets["rank"] = K.rank_net(R, K.edge_count(T))
    rows.append(_row("rank", nets["rank"], T))
    path_nets = [K.glasso_net(t) for t in thetas]
    return Analysis(s, R, T, nets, rows, partial, np.asarray(lams), [K.edge_count(a) for a in path_nets], [K.score(a, T)["f1"] for a in path_nets],
                    [K.ebic(R, t, s.n, s.gamma) for t in thetas], i_ebic, i_best, lam_used, K.transitivity(T))


# --- Zellen des Rasters (Mittel über viele Datensätze) -----------------------------------------------------------------------------------------

def _stats(A: np.ndarray, T: np.ndarray) -> list:
    sc = K.score(A, T)
    return [sc["edges"], sc["right"], sc["wrong"], sc["f1"], K.transitivity(A)]


def _mean_block(rows: list) -> dict:
    a = np.array(rows, dtype=float)
    return {"edges": float(a[:, 0].mean()), "right": float(a[:, 1].mean()), "wrong": float(a[:, 2].mean()), "f1": float(a[:, 3].mean()), "f1_sd": float(a[:, 3].std()), "trans": float(a[:, 4].mean())}


def net_cell(truth: str, n: int, p: int, runs: int, seeds: tuple = C.GRID_SEEDS, with_path: bool = True) -> dict:
    """Mittel über len(seeds) · runs Datensätze: Schwellen-, Teilkorrelationsnetz, Rangfolge und (mit Pfad) Graphical Lasso mit EBIC (γ = 0, 0,5, 1) und bestem λ; dazu die exakten
    Erwartungen der beiden einfachen Netze und die Pfadkurven (Kanten, F1 je Pfadwert)."""
    T = K.true_edges(truth, p)
    Sigma = K.true_cov(truth, p)
    k_true = K.edge_count(T)
    acc: dict = {}
    path_edges, path_f1, path_wrong = [], [], []
    ebic_idx = {g: [] for g in C.GAMMAS}
    best_idx = []
    for seed in seeds:
        rng = VRng(seed)
        for _ in range(runs):
            R = K.sample_corr(K.draw(truth, n, p, rng))
            acc.setdefault("schwelle", []).append(_stats(K.threshold_net(R, C.DEFAULT_TAU), T))
            pn = K.partial_net(R, n, C.DEFAULT_ALPHA)
            if pn is not None:
                acc.setdefault("teil", []).append(_stats(pn[0], T))
            acc.setdefault("rank", []).append(_stats(K.rank_net(R, k_true), T))
            if not with_path:
                continue
            _lams, thetas = K.glasso_path(R)
            nets = [K.glasso_net(t) for t in thetas]
            scores = [K.score(a, T) for a in nets]
            path_edges.append([s["edges"] for s in scores])
            path_f1.append([s["f1"] for s in scores])
            path_wrong.append([s["wrong"] for s in scores])
            for g in C.GAMMAS:
                i = K.select_ebic(R, thetas, n, g)
                ebic_idx[g].append(i)
                acc.setdefault(f"ebic_{g}", []).append(_stats(nets[i], T))
            i = K.select_best(thetas, T)
            best_idx.append(i)
            acc.setdefault("best", []).append(_stats(nets[i], T))
    datasets = len(seeds) * runs
    out = {"truth": truth, "n": n, "p": p, "datasets": datasets, "true_edges": k_true, "trans_true": K.transitivity(T),
           "methods": {k: _mean_block(v) for k, v in acc.items()},
           "expected": {"schwelle": K.expected_threshold_edges(Sigma, n, C.DEFAULT_TAU, T), "teil": K.expected_partial_edges(Sigma, n, C.DEFAULT_ALPHA, T)}}
    if with_path:
        steps = C.PATH_K - 1
        out["path"] = {"ratio": [C.PATH_RATIO ** (i / steps) for i in range(C.PATH_K)], "edges": np.mean(path_edges, axis=0).tolist(), "f1": np.mean(path_f1, axis=0).tolist(),
                       "wrong": np.mean(path_wrong, axis=0).tolist(),
                       "ebic_ratio": {str(g): float(np.mean([C.PATH_RATIO ** (i / steps) for i in ebic_idx[g]])) for g in C.GAMMAS},
                       "best_ratio": float(np.mean([C.PATH_RATIO ** (i / steps) for i in best_idx]))}
    return out


def grid_probe() -> dict:
    """Kleine feste Probe, die tests/test_grid.py nachrechnet und mit der Datei vergleicht (eine Zelle der Messreihen, zwei Datensätze)."""
    return net_cell("kette", 200, 40, 1, seeds=(101, 102))


def load_grid(path: Path = GRID_FILE):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_cell(cells: list, truth: str, n: int):
    return next((c for c in cells if c["truth"] == truth and c["n"] == n), None)


# --- Verdikte (reine Funktionen: Art, Text) ----------------------------------------------------------------------------------------------------

def _count(k: int, singular: str, plural: str) -> str:
    return f"{k} {singular if k == 1 else plural}"


def verdict_main(a: Analysis):
    s = a.settings
    pairs = s.p * (s.p - 1) // 2
    t_edges = K.edge_count(a.T)
    sw, te, gl = a.row("schwelle"), a.row("teil"), a.row("glasso")
    if t_edges == 0:
        teil = f"das Teilkorrelationsnetz {_count(te.edges, 'Kante', 'Kanten')}" if te is not None else "das Teilkorrelationsnetz gibt es bei n ≤ p + 1 nicht"
        return "warning", (f"Es gibt keine direkte Kante, und doch hat das Schwellennetz **{sw.edges} von {pairs}** möglichen Kanten (Transitivität {de(sw.trans)}); {teil}; "
                           f"der Graphical Lasso zeichnet {_count(gl.edges, 'Kante', 'Kanten')}. Eine verborgene gemeinsame Ursache macht fast jedes Paar korreliert, ohne dass eines direkt am anderen hängt.")
    parts = [f"das Schwellennetz {sw.right} (und {sw.wrong} falsche)"]
    parts.append(f"das Teilkorrelationsnetz {te.right} (und {te.wrong} falsche)" if te is not None else "das Teilkorrelationsnetz gibt es bei n ≤ p + 1 nicht")
    parts.append(f"der Graphical Lasso {gl.right} (und {gl.wrong} falsche)")
    ranked = [r for r in a.rows if r.key != "rank"]
    best = max(ranked, key=lambda r: r.f1)
    kind = "success" if best.f1 >= 0.9 else "info" if best.f1 >= 0.6 else "warning"
    return kind, (f"Von **{t_edges}** echten Kanten findet {parts[0]}, {parts[1]}, {parts[2]}. F1: {de(sw.f1)} / " + (de(te.f1) if te is not None else "–") + f" / {de(gl.f1)}; "
                  f"am besten schneidet das {best.name} ab (F1 {de(best.f1)}).")


def verdict_threshold(a: Analysis):
    sw = a.row("schwelle")
    e = sw.expected
    tail = f"; bei diesen Daten sind {sw.wrong} davon falsch und {sw.missing} echte Kanten fehlen" if K.edge_count(a.T) else "; keine davon ist eine direkte Kante"
    return "info", (f"Schwellennetz (|r| > {de(a.settings.tau)}): **{sw.edges}** Kanten, erwartet {de(e['edges'], 1)}{tail}. Transitivität {de(sw.trans)}"
                    + (f" (Wahrheit {de(a.trans_true)})." if K.edge_count(a.T) else "."))


def verdict_partial(a: Analysis):
    s = a.settings
    te = a.row("teil")
    if te is None:
        return "warning", (f"Das Teilkorrelationsnetz gibt es bei n = {s.n} und p = {s.p} nicht: die Stichprobenmatrix hat dann keine Inverse, und für den Test fehlen Freiheitsgrade "
                           f"(n − p − 1 = {s.n - s.p - 1}). Der Graphical Lasso rechnet trotzdem.")
    e = te.expected
    if K.edge_count(a.T) == 0:
        return "info", f"Teilkorrelationsnetz (α = {de(s.alpha)}, Bonferroni): **{te.edges}** Kanten, erwartet {de(e['edges'], 2)}; die wahren Teilkorrelationen sind nicht null, aber bei n = {s.n} nicht nachweisbar."
    return "info", (f"Teilkorrelationsnetz (α = {de(s.alpha)}, Bonferroni): **{te.right} von {K.edge_count(a.T)}** echten Kanten, erwartet {de(e['right'], 1)}; falsche Kanten: {te.wrong} "
                    f"(erwartet {de(e['wrong'], 2)}). Wenig Daten machen den Test blind, nicht die Struktur.")


def verdict_glasso(a: Analysis):
    s = a.settings
    gl = a.row("glasso")
    lead = {"ebic": f"EBIC (γ = {de(s.gamma, 1)}) wählt λ = {de(a.lam_used, 3)}", "fest": f"Bei festem λ = {de(a.lam_used, 3)}",
            "best": f"Das beste λ des Pfads ({de(a.lam_used, 3)}; es kennt die Wahrheit)"}[s.rule]
    body = f"{lead}: **{gl.edges}** Kanten, davon {gl.wrong} falsch, F1 {de(gl.f1)}."
    if s.rule == "ebic" and gl.edges == 0:
        body = f"{lead}: das **leere Netz**, F1 {de(gl.f1)}."
    ref = a.row("rank")
    tail = f" Zum Vergleich: das beste λ des Pfads erreicht F1 {de(a.path_f1[a.i_best])}, die Rangfolge nach |r| bei bekannter Kantenzahl {de(ref.f1)} (beide kennen die Wahrheit)."
    return "info", body + tail


# --- Verdikte der Experimente (aus dem Raster) --------------------------------------------------------------------------------------------------

def _f1(cell: dict, key: str):
    m = cell["methods"].get(key)
    return None if m is None else m["f1"]


def verdict_e1(cells: list, truth: str):
    c = find_cell(cells, truth, C.E1_N)
    m, ex = c["methods"], c["expected"]
    names = {"kette": "Kette", "gitter": "Gitter", "zufall": "Zufallsnetz", "faktor": "Faktor"}
    if c["true_edges"] == 0:
        return "warning", (f"Welt {names[truth]} (keine direkte Kante, {c['datasets']} Datensätze, n = {c['n']}): das Schwellennetz hat im Mittel **{de(m['schwelle']['edges'], 1)} von 780** Kanten "
                           f"(erwartet {de(ex['schwelle']['edges'], 1)}), Transitivität {de(m['schwelle']['trans'])}; das Teilkorrelationsnetz {de(m['teil']['edges'], 1)}; der Graphical Lasso mit EBIC "
                           f"{de(m['ebic_0.5']['edges'], 1)} Kanten.")
    kind = "info"
    return kind, (f"Welt {names[truth]} ({c['true_edges']} echte Kanten, {c['datasets']} Datensätze, n = {c['n']}): F1 Schwelle {de(m['schwelle']['f1'], 3)}, Teilkorrelation {de(m['teil']['f1'], 3)}, "
                  f"Graphical Lasso mit EBIC {de(m['ebic_0.5']['f1'], 3)}, mit bestem λ {de(m['best']['f1'], 3)}, Rangfolge {de(m['rank']['f1'], 3)}. "
                  f"Das Schwellennetz hat {de(m['schwelle']['edges'], 1)} Kanten ({de(m['schwelle']['wrong'], 1)} falsch), erwartet {de(ex['schwelle']['edges'], 1)}.")


def verdict_e2(cells: list, truth: str):
    mine = sorted((c for c in cells if c["truth"] == truth), key=lambda c: c["n"])
    lo, hi = mine[0], mine[-1]
    sw_lo, sw_hi = lo["methods"]["schwelle"], hi["methods"]["schwelle"]
    t_lo, t_hi = _f1(lo, "teil"), _f1(hi, "teil")
    verb = "sinkt" if sw_hi["f1"] < sw_lo["f1"] else "steigt"
    return "info", (f"Schwellennetz: F1 {verb} von {de(sw_lo['f1'], 3)} (n = {lo['n']}, {de(sw_lo['edges'], 1)} Kanten) auf {de(sw_hi['f1'], 3)} (n = {hi['n']}, {de(sw_hi['edges'], 1)} Kanten); "
                    f"Teilkorrelationsnetz von {de(t_lo, 3)} auf {de(t_hi, 3)}; Graphical Lasso mit EBIC von {de(_f1(lo, 'ebic_0.5'), 3)} auf {de(_f1(hi, 'ebic_0.5'), 3)}, "
                    f"mit bestem λ von {de(_f1(lo, 'best'), 3)} auf {de(_f1(hi, 'best'), 3)}.")


def verdict_e3(cells: list, truth: str):
    mine = sorted((c for c in cells if c["truth"] == truth), key=lambda c: c["n"])
    lo, hi = mine[0], mine[-1]
    return "warning", (f"Mit n = {lo['n']} Wochen und {lo['p']} Kennzahlen hat das Schwellennetz **{de(lo['methods']['schwelle']['edges'], 1)}** Kanten bei {lo['true_edges']} echten (F1 {de(lo['methods']['schwelle']['f1'], 3)}); "
                       f"die Rangfolge nach |r| mit bekannter Kantenzahl erreicht {de(lo['methods']['rank']['f1'], 3)}, das beste λ des Graphical Lasso {de(lo['methods']['best']['f1'], 3)}, EBIC "
                       f"{de(lo['methods']['ebic_0.5']['f1'], 3)}. Bei n = {hi['n']}: {de(hi['methods']['schwelle']['f1'], 3)} / {de(hi['methods']['rank']['f1'], 3)} / {de(hi['methods']['best']['f1'], 3)} / "
                       f"{de(hi['methods']['ebic_0.5']['f1'], 3)}.")


def verdict_e4(cell: dict, gamma: float):
    m = cell["methods"]
    e = m[f"ebic_{gamma}"]
    return "info", (f"EBIC mit γ = {de(gamma, 1)} ({cell['truth']}, n = {cell['n']}): im Mittel **{de(e['edges'], 1)}** Kanten, davon {de(e['wrong'], 1)} falsch, F1 {de(e['f1'], 3)}; "
                    f"das beste λ des Pfads (kennt die Wahrheit) {de(m['best']['edges'], 1)} Kanten, {de(m['best']['wrong'], 1)} falsch, F1 {de(m['best']['f1'], 3)}; "
                    f"die Rangfolge nach |r| F1 {de(m['rank']['f1'], 3)}.")


def verdict_e5(cells: list):
    cells = sorted(cells, key=lambda c: c["n"])
    lo, hi = cells[0], cells[-1]
    return "warning", (f"Keine direkte Kante: das Teilkorrelationsnetz hat {de(lo['methods']['teil']['edges'], 1)} Kanten bei n = {lo['n']}, aber **{de(hi['methods']['teil']['edges'], 1)}** bei n = {hi['n']}; "
                       f"das Schwellennetz {de(lo['methods']['schwelle']['edges'], 1)} bzw. {de(hi['methods']['schwelle']['edges'], 1)} von 780. Die wahre Teilkorrelation jedes Paares ist nicht null.")
