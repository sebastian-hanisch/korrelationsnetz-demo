"""App: jedes Preset, der Permalink, jedes Experiment-Auswahlfeld einzeln und die Eckfall-Kombinationen (DEMO-PLAYBOOK §1d)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import knz_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texts(at):
    return " ".join([m.value for m in at.markdown] + [s.value for s in at.success] + [i.value for i in at.info] + [w.value for w in at.warning])


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_runs_and_shows_its_data(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception, [str(e) for e in at.exception]
    preset = C.PRESETS[name]
    assert at.session_state["truth_select"] == preset["truth"] and at.session_state["n_select"] == preset["n"] and at.session_state["rule_select"] == preset["rule"]
    assert at.session_state["p_select"] == preset["p"] and at.session_state["tau_slider"] == preset["tau"] and at.session_state["gamma_select"] == preset["gamma"]
    partial = preset["n"] - preset["p"] - 1 > 0
    assert len(at.get("plotly_chart")) == (3 if partial else 2) + 7        # Hauptansicht: Netze, Paare (nur mit Teilkorrelation), Pfad; E1: 1, E2: 2, E3: 2, E4: 1, E5: 1


def test_permalink_is_loaded_and_clamped():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["truth"] = "gitter"
    at.query_params["tau"] = "9"
    at.query_params["lam"] = "-1"
    at.query_params["n"] = "100"
    at.query_params["p"] = "12"                      # ungültig: wird ignoriert
    at.query_params["gamma"] = "1"
    at.run()
    assert not at.exception
    assert at.session_state["truth_select"] == "gitter" and at.session_state["tau_slider"] == C.TAU_MAX and at.session_state["lam_slider"] == C.LAM_MIN
    assert at.session_state["n_select"] == 100 and at.session_state["p_select"] == C.DEFAULT_P and at.session_state["gamma_select"] == 1.0


def test_permalink_ignores_unknown_choices():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["truth"] = "stern"
    at.query_params["rule"] = "zufall"
    at.query_params["alpha"] = "0.2"
    at.run()
    assert not at.exception and at.session_state["truth_select"] == C.DEFAULT_TRUTH and at.session_state["rule_select"] == "ebic" and at.session_state["alpha_select"] == C.DEFAULT_ALPHA


def test_four_metrics_show_the_nets_and_a_dash_when_the_partial_net_does_not_exist():
    at = _app()
    assert [m.label for m in at.metric] == ["Schwellennetz (F1)", "Teilkorrelationsnetz (F1)", "Graphical Lasso (F1)", "Rangfolge (kennt k) (F1)"]
    assert all(m.value != "–" for m in at.metric)
    at.sidebar.select_slider(key="n_select").set_value(30).run()
    assert not at.exception and at.metric[1].value == "–" and at.metric[0].value != "–"


def test_the_headline_verdict_depends_on_the_world():
    at = _app()
    assert "Von **39** echten Kanten" in _texts(at) and "Es gibt keine direkte Kante" not in _texts(at)
    at.sidebar.button(key="preset_Gemeinsame Ursache").click().run()
    assert not at.exception and "Es gibt keine direkte Kante" in _texts(at)


def test_the_table_lists_all_four_nets_and_the_expected_edges():
    at = _app()
    rows = at.dataframe[0].value
    assert list(rows["Netz"]) == ["Schwellennetz", "Teilkorrelationsnetz", "Graphical Lasso", "Rangfolge nach |r| (kennt die Zahl der echten Kanten)"]
    assert list(rows["erwartet (Kanten)"])[2:] == ["–", "–"] and list(rows["erwartet (Kanten)"])[0] != "–"


def test_every_sidebar_selector_runs():
    at = _app()
    for key, value in (("truth_select", "faktor"), ("truth_select", "gitter"), ("truth_select", "zufall"), ("rule_select", "fest"), ("rule_select", "best"), ("rule_select", "ebic")):
        at.sidebar.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])
    for key, value in (("p_select", 10), ("n_select", 20), ("n_select", 2000), ("alpha_select", 0.01), ("gamma_select", 0.0), ("gamma_select", 1.0), ("p_select", 40)):
        at.sidebar.select_slider(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_pdf_button_then_download():
    at = _app()
    at.button(key="pdf_make").click().run()
    assert not at.exception and len(at.get("download_button")) == 1


def test_experiment_selectors_run():
    at = _app()
    for key, value in (("e1_truth", "faktor"), ("e1_truth", "gitter"), ("e1_gamma", 0.0), ("e1_gamma", 1.0), ("e2_truth", "gitter"), ("e2_truth", "zufall"), ("e2_gamma", 1.0),
                       ("e3_truth", "gitter"), ("e3_gamma", 0.0), ("e4_cell", ("faktor", 500)), ("e4_cell", ("kette", 100)), ("e4_gamma", 0.0), ("e4_gamma", 1.0)):
        at.selectbox(key=key).set_value(value).run()
        assert not at.exception, (key, value, [str(e) for e in at.exception])


def test_the_experiment_sections_show_their_verdicts():
    text = _texts(_app())
    for needle in ("Welt Kette (39 echte Kanten, 20 Datensätze, n = 500)", "Schwellennetz: F1 steigt von", "Mit n = 20 Wochen und 40 Kennzahlen hat das Schwellennetz", "EBIC mit γ = 0,5 (", "Keine direkte Kante: das Teilkorrelationsnetz"):
        assert needle in text, needle
