"""Jede Zahl der README steht hier und wird aus dem vorgerechneten Raster (data/network_grid.json) oder über dieselben Funktionen wie in der App nachgerechnet (DEMO-PLAYBOOK §4).

Zufall nur aus SplitMix64 mit festen Seeds. Die Tabellen der README werden Zeile für Zeile in der Rundung der App (`_c`) gegen die README geprüft. Die Zeilen-Bauer (`e1_rows` usw.) und
`numbers()` dienen auch dazu, die README-Tabellen und -Zahlen zu erzeugen (siehe README: 'Befunde')."""

import json
from pathlib import Path

import pytest

import knz_algorithm as K
import knz_constants as C
import knz_evaluation as ev

ROOT = Path(__file__).resolve().parent.parent
README_PATH = ROOT / "README.md"
README = README_PATH.read_text(encoding="utf-8") if README_PATH.exists() else ""
NAMES = {"kette": "Kette", "gitter": "Gitter", "zufall": "Zufallsnetz", "faktor": "Faktor"}


def _c(v, d):
    return f"{round(v, d) + 0.0:.{d}f}".replace(".", ",")                  # kein „-0,000“


def _g(v):
    return f"{v:g}".replace(".", ",")


@pytest.fixture(scope="module")
def grid():
    g = ev.load_grid()
    assert g is not None
    return g


def cell(grid, group, truth, n):
    c = ev.find_cell(grid[group], truth, n)
    assert c is not None, (group, truth, n)
    return c


# --- Zeilen-Bauer (Rundung der App) ------------------------------------------------------------------------------------------------------------

def e1_rows(grid) -> list[str]:
    rows = []
    for t in ("kette", "gitter", "zufall", "faktor"):
        c = cell(grid, "e1", t, C.E1_N)
        m = c["methods"]
        sw, te, eb = m["schwelle"], m["teil"], m["ebic_0.5"]
        rows.append(f"| **{NAMES[t]}** ({c['true_edges']}; {_c(c['trans_true'], 2)}) | {_c(sw['edges'], 1)} / {_c(sw['wrong'], 1)} / {_c(sw['f1'], 3)} / {_c(sw['trans'], 2)} | "
                    f"{_c(te['edges'], 1)} / {_c(te['wrong'], 1)} / {_c(te['f1'], 3)} | {_c(eb['edges'], 1)} / {_c(eb['wrong'], 1)} / {_c(eb['f1'], 3)} | {_c(m['best']['f1'], 3)} | {_c(m['rank']['f1'], 3)} |")
    return rows


EXPECT_CELLS = (("e2", "kette", 100), ("e2", "kette", 200), ("e2", "kette", 500), ("e2", "kette", 2000), ("e2", "gitter", 500), ("e2", "gitter", 2000),
                ("e2", "zufall", 200), ("e2", "zufall", 500), ("e5", "faktor", 500), ("e5", "faktor", 5000), ("e5", "faktor", 50000))


def expect_rows(grid) -> list[str]:
    rows = []
    for group, t, n in EXPECT_CELLS:
        c = cell(grid, group, t, n)
        ex, m = c["expected"], c["methods"]
        part = f"{_c(ex['teil']['right'], 2)} / {_c(ex['teil']['edges'], 2)} | {_c(m['teil']['right'], 1)} / {_c(m['teil']['edges'], 1)}" if "teil" in m else "– | –"
        rows.append(f"| {NAMES[t]}, n = {n} | {_c(ex['schwelle']['edges'], 2)} | {_c(m['schwelle']['edges'], 1)} | {part} |")
    return rows


def datasize_rows(grid, truth) -> list[str]:
    cells = [cell(grid, "e2", truth, n) for n in C.E2_NS]
    out = []
    for key, name in (("schwelle", "Schwellennetz (Kanten)"), ("teil", "Teilkorrelationsnetz"), ("ebic_0.5", "Graphical Lasso, EBIC (γ = 0,5)"), ("best", "Graphical Lasso, bestes λ"), ("rank", "Rangfolge nach \\|r\\|")):
        vals = []
        for c in cells:
            m = c["methods"][key]
            vals.append(f"{_c(m['f1'], 3)} ({_c(m['edges'], 1)})" if key == "schwelle" else _c(m["f1"], 3))
        out.append(f"| {name} | " + " | ".join(vals) + " |")
    return out


def ebic_rows(grid) -> list[str]:
    rows = []
    for t, group, n in (("kette", "e1", 500), ("gitter", "e1", 500), ("zufall", "e1", 500), ("kette", "e2", 100), ("kette", "e2", 2000)):
        c = cell(grid, group, t, n)
        rows.append(f"| {NAMES[t]}, n = {n} | " + " | ".join(f"{_c(c['methods'][f'ebic_{g}']['edges'], 1)} / {_c(c['methods'][f'ebic_{g}']['wrong'], 1)} / {_c(c['methods'][f'ebic_{g}']['f1'], 3)}" for g in C.GAMMAS)
                    + f" | {_c(c['methods']['best']['edges'], 1)} / {_c(c['methods']['best']['wrong'], 1)} / {_c(c['methods']['best']['f1'], 3)} |")
    return rows


def fewweeks_rows(grid) -> list[str]:
    out = []
    for t in ("kette", "gitter", "zufall"):
        for n in C.E3_NS:
            c = cell(grid, "e3", t, n)
            m = c["methods"]
            teil = _c(m["teil"]["edges"], 1) if "teil" in m else "–"
            out.append(f"| {NAMES[t]}, n = {n} | {_c(m['schwelle']['edges'], 1)} ({_c(m['schwelle']['f1'], 3)}) | {_c(m['rank']['f1'], 3)} | {_c(m['best']['f1'], 3)} | {_c(m['ebic_0.5']['f1'], 3)} | {teil} |")
    return out


def factor_rows(grid) -> list[str]:
    out = []
    for n in C.E5_NS:
        c = cell(grid, "e5", "faktor", n)
        m = c["methods"]
        out.append(f"| {n} | {_c(m['schwelle']['edges'], 1)} ({_c(m['schwelle']['trans'], 3)}) | {_c(m['teil']['edges'], 1)} ({_c(m['teil']['trans'], 3)}) |")
    return out


def numbers(grid) -> dict:
    """Die Zahlen im Fließtext der README (Name -> Zeichenkette in der Rundung der App)."""
    e1 = {t: cell(grid, "e1", t, C.E1_N) for t in C.TRUTHS}
    e2 = {(t, n): cell(grid, "e2", t, n) for t in C.E2_TRUTHS for n in C.E2_NS}
    e3 = {(t, n): cell(grid, "e3", t, n) for t in C.E2_TRUTHS for n in C.E3_NS}
    gaps = [abs(c["methods"]["best"]["f1"] - c["methods"]["rank"]["f1"]) for c in grid["e1"] + grid["e2"] + grid["e3"]]
    f = e1["faktor"]
    return {
        "fak_edges": _c(f["methods"]["schwelle"]["edges"], 1), "fak_trans": _c(f["methods"]["schwelle"]["trans"], 2), "fak_teil": _c(f["methods"]["teil"]["edges"], 1),
        "fak_ebic": _c(f["methods"]["ebic_0.5"]["edges"], 1), "fak_exp": _c(f["expected"]["schwelle"]["edges"], 1),
        "git_sw": _c(e1["gitter"]["methods"]["schwelle"]["edges"], 1), "git_sw_f1": _c(e1["gitter"]["methods"]["schwelle"]["f1"], 3), "git_teil": _c(e1["gitter"]["methods"]["teil"]["edges"], 1),
        "ket_sw_wrong": _c(e1["kette"]["methods"]["schwelle"]["wrong"], 1), "ket_sw_trans": _c(e1["kette"]["methods"]["schwelle"]["trans"], 2),
        "git100_f1": _c(e2[("gitter", 100)]["methods"]["schwelle"]["f1"], 3), "git2000_f1": _c(e2[("gitter", 2000)]["methods"]["schwelle"]["f1"], 3), "git2000_edges": _c(e2[("gitter", 2000)]["methods"]["schwelle"]["edges"], 1),
        "ket100_teil": _c(e2[("kette", 100)]["methods"]["teil"]["edges"], 1), "ket100_teil_exp": _c(e2[("kette", 100)]["expected"]["teil"]["edges"], 2),
        "ket50_sw": _c(e2[("kette", 50)]["methods"]["schwelle"]["edges"], 1), "ket20_sw": _c(e3[("kette", 20)]["methods"]["schwelle"]["edges"], 1),
        "ket2000_ebic_wrong": _c(e2[("kette", 2000)]["methods"]["ebic_0.5"]["wrong"], 1), "ket500_ebic_wrong": _c(e1["kette"]["methods"]["ebic_0.5"]["wrong"], 1), "git500_ebic_wrong": _c(e1["gitter"]["methods"]["ebic_0.5"]["wrong"], 1),
        "ket50_ebic": _c(e2[("kette", 50)]["methods"]["ebic_0.5"]["edges"], 1), "git500_g1": _c(e1["gitter"]["methods"]["ebic_1.0"]["edges"], 1), "git500_g1_f1": _c(e1["gitter"]["methods"]["ebic_1.0"]["f1"], 3),
        "gap_max": _c(max(gaps), 3), "n_gap_cells": str(len(gaps)),
        "ket20_sw_f1": _c(e3[("kette", 20)]["methods"]["schwelle"]["f1"], 3), "ket20_rank": _c(e3[("kette", 20)]["methods"]["rank"]["f1"], 3), "ket20_best": _c(e3[("kette", 20)]["methods"]["best"]["f1"], 3),
        "fak_teil5000": _c(cell(grid, "e5", "faktor", 5000)["methods"]["teil"]["edges"], 1), "fak_teil50000": _c(cell(grid, "e5", "faktor", 50000)["methods"]["teil"]["edges"], 1),
        "fak_teil50000_trans": _c(cell(grid, "e5", "faktor", 50000)["methods"]["teil"]["trans"], 3), "fak_pc": _c(float(K.true_partial_corr("faktor", 40)[0, 1]), 4),
        "fak_n": str(f["datasets"]), "e1_runs": str(f["datasets"]), "e2_runs": str(e2[("kette", 50)]["datasets"]), "e5_runs": str(cell(grid, "e5", "faktor", 500)["datasets"]),
        "grid_seconds": str(grid["meta"]["seconds"]), "ket20_exp": _c(e3[("kette", 20)]["expected"]["schwelle"]["edges"], 1),
        "zuf_pc_min": _c(float(abs(K.true_partial_corr("zufall", 40)[K.true_edges("zufall", 40)]).min()), 2), "zuf_pc_max": _c(float(abs(K.true_partial_corr("zufall", 40)[K.true_edges("zufall", 40)]).max()), 2),
    }


# --- Tabellen und Zahlen der README --------------------------------------------------------------------------------------------------------------

def test_claim_readme_exists_and_has_the_required_sections():
    assert README, "README.md fehlt"
    for heading in ("## Warum dieses Problem", "## Modell", "## Methodik", "## Befunde (gemessen, keine Behauptungen)", "## Ehrliche Grenzen", "## Verifikation", "## Dateistruktur", "## Bewusst nicht umgesetzt"):
        assert heading in README, heading
    assert "TODO" not in README.replace("<!-- TODO nach dem Deploy", "")


@pytest.mark.parametrize("builder", [e1_rows, expect_rows, fewweeks_rows, factor_rows, ebic_rows])
def test_claim_readme_tables_match_the_grid_row_by_row(grid, builder):
    for row in builder(grid):
        assert row in README, row


@pytest.mark.parametrize("truth", ["kette", "gitter"])
def test_claim_readme_datasize_tables_match_the_grid(grid, truth):
    for row in datasize_rows(grid, truth):
        assert row in README, row


def test_claim_readme_numbers_in_the_text(grid):
    nums = numbers(grid)
    for name, text in nums.items():
        assert text in README, (name, text)


def test_claim_the_headline_findings_hold_in_the_grid(grid):
    e1 = {c["truth"]: c["methods"] for c in grid["e1"]}
    assert e1["faktor"]["schwelle"]["edges"] > 700 and e1["faktor"]["schwelle"]["trans"] > 0.9 and e1["faktor"]["teil"]["edges"] < 1.0
    assert e1["gitter"]["schwelle"]["edges"] < 12 and e1["kette"]["teil"]["f1"] > 0.99 and e1["kette"]["schwelle"]["wrong"] > 3
    g100, g2000 = cell(grid, "e2", "gitter", 100)["methods"]["schwelle"], cell(grid, "e2", "gitter", 2000)["methods"]["schwelle"]
    assert g2000["f1"] < g100["f1"] and g2000["edges"] < 1.0                                                 # mehr Daten, schlechteres Schwellennetz
    for c in grid["e1"] + grid["e2"] + grid["e3"]:
        assert abs(c["methods"]["best"]["f1"] - c["methods"]["rank"]["f1"]) < 0.03                         # bestes λ und Rangfolge liegen innerhalb 0,03
    ket = {n: cell(grid, "e2", "kette", n)["methods"] for n in C.E2_NS}
    assert ket[50]["ebic_0.5"]["edges"] == 0.0 and ket[2000]["ebic_0.5"]["wrong"] > ket[500]["ebic_0.5"]["wrong"] > ket[200]["ebic_0.5"]["wrong"] > ket[100]["ebic_0.5"]["wrong"]
    f5 = {n: cell(grid, "e5", "faktor", n)["methods"] for n in C.E5_NS}
    assert f5[500]["teil"]["edges"] < 1.0 and f5[50000]["teil"]["edges"] > 700


def test_claim_no_gamma_wins_everywhere(grid):
    """EBIC mit γ = 0 wählt in jeder Zelle mit echten Kanten mehr als doppelt so viele Kanten wie echte (oder wenigstens 15 falsche); γ = 1 ist beim Gitter zu streng, γ = 0,5 bei wenig Daten leer."""
    cells = grid["e1"] + grid["e2"] + grid["e3"]
    wins = {g: sum(1 for c in cells if c["true_edges"] and c["methods"][f"ebic_{g}"]["f1"] == max(c["methods"][f"ebic_{h}"]["f1"] for h in C.GAMMAS)) for g in C.GAMMAS}
    assert len([g for g in C.GAMMAS if wins[g] > 0]) >= 2, wins
    gitter = cell(grid, "e1", "gitter", 500)["methods"]
    assert gitter["ebic_1.0"]["f1"] < gitter["ebic_0.5"]["f1"] and cell(grid, "e2", "kette", 2000)["methods"]["ebic_1.0"]["f1"] > cell(grid, "e2", "kette", 2000)["methods"]["ebic_0.5"]["f1"]


# --- Texte der Presets ---------------------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def analyses():
    return {name: ev.analyse(ev.Settings(**p)) for name, p in C.PRESETS.items()}


def test_claim_the_preset_help_texts_quote_the_analysis(analyses):
    h = C.PRESET_HELP
    a = analyses["Standardfall (Voreinstellung)"]
    sw, te, gl = a.row("schwelle"), a.row("teil"), a.row("glasso")
    assert K.edge_count(a.T) == 39 and sw.right == 39 and te.right == 39 and te.wrong == 0 and gl.right == 39
    assert f"findet alle und {sw.wrong} falsche (erwartet {_c(sw.expected['edges'], 1)} Kanten)" in h["Standardfall (Voreinstellung)"]
    assert f"genau die {te.right}, der Graphical Lasso mit EBIC die {gl.right} und {gl.wrong} falsche" in h["Standardfall (Voreinstellung)"]
    a = analyses["Gemeinsame Ursache"]
    sw, te, gl = a.row("schwelle"), a.row("teil"), a.row("glasso")
    assert K.edge_count(a.T) == 0 and f"{sw.edges} von 780 möglichen Kanten (erwartet {_c(sw.expected['edges'], 1)}), das Teilkorrelationsnetz {te.edges}, der Graphical Lasso mit EBIC {gl.edges}" in h["Gemeinsame Ursache"]
    a = analyses["Die Schwelle verfehlt es"]
    sw, te, gl = a.row("schwelle"), a.row("teil"), a.row("glasso")
    assert K.edge_count(a.T) == 67 and gl.right == 67
    assert f"findet nur {sw.right} (erwartet {_c(sw.expected['right'], 1)}), das Teilkorrelationsnetz {te.right} (erwartet {_c(te.expected['right'], 1)}), der Graphical Lasso mit EBIC alle {gl.right} und {gl.wrong} falsche" in h["Die Schwelle verfehlt es"]
    a = analyses["Zu wenig Wochen für die Teilkorrelation"]
    sw, te = a.row("schwelle"), a.row("teil")
    assert f"{sw.edges} Kanten, davon {sw.wrong} falsche (erwartet {_c(sw.expected['edges'], 1)}), das Teilkorrelationsnetz nur {te.right} der {K.edge_count(a.T)} echten (erwartet {_c(te.expected['right'], 1)})" in h["Zu wenig Wochen für die Teilkorrelation"]
    a = analyses["Mehr Kennzahlen als Wochen"]
    sw = a.row("schwelle")
    assert a.row("teil") is None and a.settings.rule == "best"
    assert (f"{sw.edges} Kanten bei {K.edge_count(a.T)} echten (erwartet {_c(sw.expected['edges'], 1)}), das beste λ des Graphical Lasso erreicht F1 {_c(a.row('glasso').f1, 2)}, "
            f"die Rangfolge nach |r| {_c(a.row('rank').f1, 2)}") in h["Mehr Kennzahlen als Wochen"]
    a = analyses["Zufallsnetz"]
    sw, te, gl = a.row("schwelle"), a.row("teil"), a.row("glasso")
    assert f"findet {sw.right} (erwartet {_c(sw.expected['right'], 1)}), das Teilkorrelationsnetz {te.right}, der Graphical Lasso mit EBIC alle {gl.right} und {gl.wrong} falsche" in h["Zufallsnetz"]
    assert K.edge_count(a.T) == 43 and gl.right == 43


def test_claim_the_readme_names_the_standard_case_numbers(analyses):
    a = analyses["Standardfall (Voreinstellung)"]
    assert f"{a.row('schwelle').wrong} falsche" in C.PRESET_HELP["Standardfall (Voreinstellung)"]


def test_claim_hand_values_of_the_text():
    R3 = [[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]]
    import numpy as np
    assert abs(K.partial_corr(np.array(R3))[0, 2]) < 1e-12 and K.partial_corr(np.array([[1, .36, .36], [.36, 1, .36], [.36, .36, 1]]))[0, 1] == pytest.approx(0.2647, abs=5e-5)
    assert K.glasso(np.array([[1.0, 0.6], [0.6, 1.0]]), 0.2, tol=1e-12)[1][0, 1] == pytest.approx(0.4, abs=1e-9)
    assert K.bonferroni_z(40, 0.05) == pytest.approx(3.99, abs=0.01) and K.partial_df(100, 40) == 59


def test_claim_mutation_numbers_in_the_readme():
    state = json.loads((ROOT / "tests" / "mutation_state.json").read_text(encoding="utf-8"))
    killed = sum(1 for v in state.values() if v == "killed")
    equivalent = sum(1 for v in state.values() if v == "equivalent")
    assert (killed, equivalent) == (117, 5)
    assert "117 durch eigene Tests gefangen" in README and "5 als gleichwertig festgehalten" in README and "129 Stellen" in README
