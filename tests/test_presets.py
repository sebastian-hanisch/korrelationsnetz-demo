"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import pytest

import knz_constants as C
import knz_evaluation as ev
import knz_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, preset in C.PRESETS.items():
        assert preset["truth"] in C.TRUTHS and preset["p"] in C.P_OPTIONS and preset["n"] in C.N_OPTIONS, name
        assert C.TAU_MIN <= preset["tau"] <= C.TAU_MAX and preset["alpha"] in C.ALPHAS and preset["rule"] in C.LAMBDA_RULES, name
        assert C.LAM_MIN <= preset["lam"] <= C.LAM_MAX and preset["gamma"] in C.GAMMAS and 0 <= preset["seed"] <= C.SEED_MAX, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Standardfall (Voreinstellung)"


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_presets_differ_in_at_least_one_value():
    values = [tuple(sorted(p.items())) for p in C.PRESETS.values()]
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("key, raw, expected", [
    ("tau_slider", "9", C.TAU_MAX), ("tau_slider", "-1", C.TAU_MIN), ("tau_slider", "0.337", 0.34), ("tau_slider", "abc", None),
    ("lam_slider", "5", C.LAM_MAX), ("lam_slider", "0", C.LAM_MIN),
    ("n_select", "500", 500), ("n_select", "501", None), ("p_select", "40", 40), ("p_select", "12", None),
    ("alpha_select", "0.01", 0.01), ("alpha_select", "0.1", None), ("gamma_select", "1", 1.0), ("gamma_select", "2", None),
    ("truth_select", "gitter", "gitter"), ("truth_select", "stern", None), ("rule_select", "best", "best"), ("rule_select", "x", None),
    ("seed_input", "99999999", C.SEED_MAX), ("seed_input", "-5", 0),
])
def test_permalink_values_are_cleaned(key, raw, expected):
    assert P.clean_value(P.SETTING_SPECS[key], raw) == expected
