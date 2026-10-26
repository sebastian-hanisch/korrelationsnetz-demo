"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen, nicht Seeds. Damit prüft der Test die Grenzen (>= 0,9, >= 0,6) und jede genannte Zahl direkt."""

import numpy as np

import knz_algorithm as K
import knz_evaluation as ev


def _row(key, edges, right, wrong, missing, f1, trans=0.0, expected=None):
    return ev.Row(key, ev.METHOD_NAMES[key], edges, right, wrong, missing, f1, trans, expected)


def _analysis(truth="kette", n=500, rule="ebic", rows=None, f1_best=0.9, i_best=1, lam=0.123, p=10):
    s = ev.Settings(truth=truth, p=p, n=n, tau=0.3, alpha=0.05, rule=rule, lam=0.15, gamma=0.5, seed=1)
    T = K.true_edges(truth, p)
    rows = rows if rows is not None else [
        _row("schwelle", 12, 9, 3, 0, 0.857, 0.2, {"edges": 11.5, "right": 9.0, "wrong": 2.5}),
        _row("teil", 9, 9, 0, 0, 1.0, 0.0, {"edges": 9.1, "right": 9.0, "wrong": 0.05}),
        _row("glasso", 15, 9, 6, 0, 0.75, 0.3),
        _row("rank", 9, 9, 0, 0, 1.0, 0.0),
    ]
    nets = {r.key: np.zeros((p, p), dtype=bool) for r in rows}
    return ev.Analysis(s, np.eye(p), T, nets, rows, None, np.array([0.5, 0.2]), [0, 5], [0.0, f1_best], [1.0, 2.0], 1, i_best, lam, 0.0)


def test_main_verdict_names_true_found_and_wrong_edges_and_f1():
    kind, text = ev.verdict_main(_analysis())
    assert kind == "success"
    assert "Von **9** echten Kanten findet das Schwellennetz 9 (und 3 falsche), das Teilkorrelationsnetz 9 (und 0 falsche), der Graphical Lasso 9 (und 6 falsche)" in text
    assert "F1: 0,86 / 1,00 / 0,75" in text and "am besten schneidet das Teilkorrelationsnetz ab (F1 1,00)" in text


def test_main_verdict_thresholds_are_inclusive_at_0_9_and_0_6():
    def kind(f1):
        rows = [_row("schwelle", 1, 1, 0, 8, f1), _row("teil", 1, 1, 0, 8, 0.1), _row("glasso", 1, 1, 0, 8, 0.1), _row("rank", 9, 9, 0, 0, 1.0)]
        return ev.verdict_main(_analysis(rows=rows))[0]
    assert (kind(0.9), kind(0.8999), kind(0.6), kind(0.5999)) == ("success", "info", "info", "warning")


def test_main_verdict_without_partial_net_says_so_and_prints_a_dash():
    rows = [_row("schwelle", 12, 9, 3, 0, 0.857), _row("glasso", 15, 9, 6, 0, 0.75), _row("rank", 9, 9, 0, 0, 1.0)]
    kind, text = ev.verdict_main(_analysis(rows=rows))
    assert "das Teilkorrelationsnetz gibt es bei n ≤ p + 1 nicht" in text and "F1: 0,86 / – / 0,75" in text and kind == "info"


def test_main_verdict_for_the_factor_world_counts_all_pairs():
    rows = [_row("schwelle", 40, 0, 40, 0, 0.0, 0.93, {"edges": 41.0, "right": 0.0, "wrong": 41.0}), _row("teil", 0, 0, 0, 0, 1.0, 0.0, {"edges": 0.2, "right": 0.0, "wrong": 0.2}),
            _row("glasso", 1, 0, 1, 0, 0.0, 0.0), _row("rank", 0, 0, 0, 0, 1.0)]
    kind, text = ev.verdict_main(_analysis("faktor", rows=rows))
    assert kind == "warning" and "**40 von 45** möglichen Kanten (Transitivität 0,93)" in text
    assert "das Teilkorrelationsnetz 0 Kanten" in text and "der Graphical Lasso zeichnet 1 Kante." in text
    rows[1:2] = []
    assert "das Teilkorrelationsnetz gibt es bei n ≤ p + 1 nicht" in ev.verdict_main(_analysis("faktor", rows=rows))[1]


def test_threshold_verdict_prints_edges_expected_and_wrong_or_the_factor_variant():
    kind, text = ev.verdict_threshold(_analysis())
    assert kind == "info" and "|r| > 0,30" in text and "**12** Kanten, erwartet 11,5" in text and "sind 3 davon falsch und 0 echte Kanten fehlen" in text and "Transitivität 0,20 (Wahrheit 0,00)" in text
    rows = [_row("schwelle", 40, 0, 40, 0, 0.0, 0.93, {"edges": 41.0, "right": 0.0, "wrong": 41.0})]
    text = ev.verdict_threshold(_analysis("faktor", rows=rows))[1]
    assert "keine davon ist eine direkte Kante" in text and "Transitivität 0,93." in text and "Wahrheit" not in text


def test_partial_verdict_for_the_three_cases():
    kind, text = ev.verdict_partial(_analysis())
    assert kind == "info" and "**9 von 9** echten Kanten, erwartet 9,0" in text and "falsche Kanten: 0 (erwartet 0,05)" in text and "α = 0,05" in text
    rows = [_row("schwelle", 12, 9, 3, 0, 0.857, 0.2, {"edges": 11.5, "right": 9.0, "wrong": 2.5})]
    kind, text = ev.verdict_partial(_analysis(n=10, rows=rows))
    assert kind == "warning" and "bei n = 10 und p = 10 nicht" in text and "(n − p − 1 = -1)" in text
    rows = [_row("schwelle", 40, 0, 40, 0, 0.0), _row("teil", 3, 0, 3, 0, 0.0, 0.0, {"edges": 2.5, "right": 0.0, "wrong": 2.5})]
    kind, text = ev.verdict_partial(_analysis("faktor", rows=rows))
    assert "**3** Kanten, erwartet 2,50" in text and "nicht null, aber bei n = 500 nicht nachweisbar" in text


def test_glasso_verdict_by_rule():
    a = _analysis(rule="ebic", f1_best=0.95)
    kind, text = ev.verdict_glasso(a)
    assert "EBIC (γ = 0,5) wählt λ = 0,123: **15** Kanten, davon 6 falsch, F1 0,75." in text
    assert "das beste λ des Pfads erreicht F1 0,95, die Rangfolge nach |r| bei bekannter Kantenzahl 1,00 (beide kennen die Wahrheit)" in text
    assert "Bei festem λ = 0,123: **15** Kanten" in ev.verdict_glasso(_analysis(rule="fest"))[1]
    assert "Das beste λ des Pfads (0,123; es kennt die Wahrheit): **15** Kanten" in ev.verdict_glasso(_analysis(rule="best"))[1]
    empty = [_row("schwelle", 12, 9, 3, 0, 0.857), _row("glasso", 0, 0, 0, 9, 0.0), _row("rank", 9, 9, 0, 0, 1.0)]
    assert "das **leere Netz**, F1 0,00." in ev.verdict_glasso(_analysis(rule="ebic", rows=empty))[1]
    assert "**0** Kanten" in ev.verdict_glasso(_analysis(rule="fest", rows=empty))[1]


def _cell(truth, n, edges=10.0, f1=0.5, true_edges=9, teil=True, trans=0.2):
    def m(e, f):
        return {"edges": e, "right": e / 2, "wrong": e / 2, "f1": f, "f1_sd": 0.0, "trans": trans}
    methods = {"schwelle": m(edges, f1), "rank": m(9.0, 0.9), "ebic_0.0": m(30.0, 0.4), "ebic_0.5": m(12.0, 0.7), "ebic_1.0": m(8.0, 0.6), "best": m(9.5, 0.95)}
    if teil:
        methods["teil"] = m(7.0, 0.8)
    return {"truth": truth, "n": n, "p": 40, "datasets": 20, "true_edges": true_edges, "trans_true": 0.0, "methods": methods,
            "expected": {"schwelle": {"edges": edges + 1.0, "right": 0.0, "wrong": 0.0}, "teil": {"edges": 7.5, "right": 0.0, "wrong": 0.0} if teil else None}}


def test_e1_verdict_for_a_world_with_edges_and_for_the_factor_world():
    kind, text = ev.verdict_e1([_cell("kette", 500)], "kette")
    assert kind == "info" and "Welt Kette (9 echte Kanten, 20 Datensätze, n = 500)" in text
    assert "F1 Schwelle 0,500, Teilkorrelation 0,800, Graphical Lasso mit EBIC 0,700, mit bestem λ 0,950, Rangfolge 0,900" in text and "hat 10,0 Kanten (5,0 falsch), erwartet 11,0" in text
    kind, text = ev.verdict_e1([_cell("faktor", 500, edges=700.0, true_edges=0, trans=0.95)], "faktor")
    assert kind == "warning" and "**700,0 von 780** Kanten (erwartet 701,0), Transitivität 0,95" in text and "das Teilkorrelationsnetz 7,0" in text and "mit EBIC 12,0 Kanten" in text


def test_e2_verdict_reports_rising_or_falling_f1():
    lo, hi = _cell("gitter", 100, edges=20.0, f1=0.4), _cell("gitter", 2000, edges=1.0, f1=0.1)
    kind, text = ev.verdict_e2([hi, lo], "gitter")
    assert "Schwellennetz: F1 sinkt von 0,400 (n = 100, 20,0 Kanten) auf 0,100 (n = 2000, 1,0 Kanten)" in text and "Teilkorrelationsnetz von 0,800 auf 0,800" in text
    lo["methods"]["schwelle"]["f1"], hi["methods"]["schwelle"]["f1"] = 0.1, 0.4
    assert "F1 steigt von 0,100" in ev.verdict_e2([lo, hi], "gitter")[1]


def test_e3_verdict_prints_the_small_and_large_cell():
    lo, hi = _cell("kette", 20, edges=199.8, f1=0.28, teil=False), _cell("kette", 60, edges=66.8, f1=0.72, teil=False)
    kind, text = ev.verdict_e3([hi, lo], "kette")
    assert kind == "warning" and "Mit n = 20 Wochen und 40 Kennzahlen hat das Schwellennetz **199,8** Kanten bei 9 echten (F1 0,280)" in text
    assert "erreicht 0,900, das beste λ des Graphical Lasso 0,950, EBIC 0,700. Bei n = 60: 0,720 / 0,900 / 0,950 / 0,700." in text


def test_e4_and_e5_verdicts():
    kind, text = ev.verdict_e4(_cell("zufall", 500), 1.0)
    assert "EBIC mit γ = 1,0 (zufall, n = 500): im Mittel **8,0** Kanten, davon 4,0 falsch, F1 0,600" in text and "das beste λ des Pfads (kennt die Wahrheit) 9,5 Kanten, 4,8 falsch, F1 0,950" in text
    lo, hi = _cell("faktor", 200, edges=640.0, true_edges=0), _cell("faktor", 50000, edges=780.0, true_edges=0)
    hi["methods"]["teil"]["edges"] = 727.0
    kind, text = ev.verdict_e5([hi, lo])
    assert kind == "warning" and "7,0 Kanten bei n = 200, aber **727,0** bei n = 50000" in text and "640,0 bzw. 780,0 von 780" in text
