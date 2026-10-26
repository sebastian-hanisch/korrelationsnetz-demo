"""Konstanten: Regler-Grenzen, Welten, Raster der Messreihen, Farben und Presets.

Zufallserzeuger und Pearson kommen als wörtliche Kopien aus pearson-demo (prs_rng.py, prs_constants.py, prs_scenario.py, prs_algorithm.py).
Rechenbudget am Regler-Maximum (DEMO-PLAYBOOK §1c): p = 40 Kennzahlen und n = 2 000 Wochen sind 80 000 Werte; Stichprobenmatrix, Inverse, Schwellen- und Teilkorrelationsnetz
und die exakten Erwartungen brauchen Millisekunden, der Graphical Lasso bei festem λ 0,01 bis 1,2 s, sein Pfad aus 15 Werten wenige Sekunden (Faktor-Welt am längsten).
Die Wiederholungsexperimente stehen vorgerechnet im Raster (data/network_grid.json)."""

from prs_constants import SEED_MAX  # noqa: F401  (Wiederverwendung der Kopie)

P_OPTIONS = (10, 15, 20, 25, 30, 35, 40)
DEFAULT_P = 40
N_OPTIONS = (20, 30, 45, 60, 100, 200, 300, 500, 1000, 2000)
DEFAULT_N = 500
TAU_MIN, TAU_MAX, DEFAULT_TAU = 0.10, 0.60, 0.30
ALPHAS = (0.01, 0.05)
DEFAULT_ALPHA = 0.05
LAM_MIN, LAM_MAX, DEFAULT_LAM = 0.02, 0.60, 0.15
GAMMAS = (0.0, 0.5, 1.0)                         # Gewicht der EBIC-Strafe
DEFAULT_GAMMA = 0.5
DEFAULT_SEED = 35

# Welten: Schlüssel -> Anzeigename
TRUTHS = {
    "kette": "Kette (jede Kennzahl hängt nur an ihren Nachbarn)",
    "gitter": "Gitter (5 Zeilen, vier Nachbarn)",
    "zufall": "Zufallsnetz (43 Kanten bei 40 Kennzahlen, beide Vorzeichen)",
    "faktor": "Faktor (verborgene gemeinsame Ursache, keine direkte Kante)",
}
DEFAULT_TRUTH = "kette"
CHAIN_PHI = 0.5                                  # Kette: Korrelation 0,5^|i−j|
GRID_ROWS, GRID_COUPLING = 5, 0.2                # Gitter: Präzision I − 0,2 A
RANDOM_SEED, RANDOM_PROB = 7, 0.06               # Zufallsnetz: fester Graph, Kantenwahrscheinlichkeit je Paar
RANDOM_W_LO, RANDOM_W_SPAN, RANDOM_DIAG_EXTRA = 0.2, 0.2, 0.2
FACTOR_RHO = 0.36                                # Faktor: Ladung 0,6, alle Paare 0,6² = 0,36

# Regel für den Abschnitt des Graphical Lasso
LAMBDA_RULES = {"ebic": "EBIC (ohne Kenntnis der Wahrheit)", "fest": "fester Wert λ", "best": "bestes λ (kennt die Wahrheit)"}
PATH_K, PATH_RATIO = 15, 0.05                    # Pfad: 15 Werte von λ_max bis 0,05 λ_max, geometrisch
EDGE_EPS = 1e-8                                  # Einträge der Präzisionsmatrix darüber zählen als Kante

# Raster der Messreihen (tools/make_grid.py)
GRID_SEEDS = (101, 102, 103, 104)
GRID_P = 40
E1_N, E1_RUNS = 500, 5                           # Wochen und Datensätze je Seed (Vier Welten)
E2_TRUTHS = ("kette", "gitter", "zufall")
E2_NS, E2_RUNS = (50, 100, 200, 500, 2000), 4    # Datenmenge
E3_NS, E3_RUNS = (20, 30, 45, 60), 4             # Mehr Kennzahlen als Wochen
E5_NS, E5_RUNS = (200, 500, 5000, 50000), 25     # Verborgener Faktor (ohne Graphical Lasso)

COLOR_RIGHT = "#2F6B65"
COLOR_WRONG = "#d6492f"
COLOR_MISSING = "#8a8a8a"
COLOR_NODE = "#4c78a8"
COLOR_EXACT = "#222222"
METHOD_COLORS = {"schwelle": "#4c78a8", "teil": "#54a24b", "ebic": "#e45756", "best": "#f58518", "rank": "#9d9d9d"}

PRESETS = {
    "Standardfall (Voreinstellung)": {"truth": "kette", "p": 40, "n": 500, "tau": 0.30, "alpha": 0.05, "rule": "ebic", "lam": 0.15, "gamma": 0.5, "seed": DEFAULT_SEED},
    "Gemeinsame Ursache": {"truth": "faktor", "p": 40, "n": 500, "tau": 0.30, "alpha": 0.05, "rule": "ebic", "lam": 0.15, "gamma": 0.5, "seed": DEFAULT_SEED},
    "Die Schwelle verfehlt es": {"truth": "gitter", "p": 40, "n": 500, "tau": 0.30, "alpha": 0.05, "rule": "ebic", "lam": 0.15, "gamma": 0.5, "seed": DEFAULT_SEED},
    "Zu wenig Wochen für die Teilkorrelation": {"truth": "kette", "p": 40, "n": 100, "tau": 0.30, "alpha": 0.05, "rule": "ebic", "lam": 0.15, "gamma": 0.5, "seed": DEFAULT_SEED},
    "Mehr Kennzahlen als Wochen": {"truth": "kette", "p": 40, "n": 30, "tau": 0.30, "alpha": 0.05, "rule": "best", "lam": 0.15, "gamma": 0.5, "seed": DEFAULT_SEED},
    "Zufallsnetz": {"truth": "zufall", "p": 40, "n": 500, "tau": 0.30, "alpha": 0.05, "rule": "ebic", "lam": 0.15, "gamma": 0.5, "seed": DEFAULT_SEED},
}
# Die Hilfetexte nennen exakte Werte und Zahlen der Auswertung; tests/test_claims.py prüft jede davon.
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Eine Kette mit 40 Kennzahlen und 500 Wochen hat 39 echte Kanten. Das Schwellennetz findet alle und 4 falsche (erwartet 43,3 Kanten), das Teilkorrelationsnetz genau die 39, der Graphical Lasso mit EBIC die 39 und 16 falsche.",
    "Gemeinsame Ursache": "Keine direkte Kante, nur ein verborgener Faktor: das Schwellennetz hat 715 von 780 möglichen Kanten (erwartet 728,1), das Teilkorrelationsnetz 2, der Graphical Lasso mit EBIC 505.",
    "Die Schwelle verfehlt es": "Ein Gitter mit 67 echten Kanten: das Schwellennetz findet nur 7 (erwartet 8,2), das Teilkorrelationsnetz 43 (erwartet 42,6), der Graphical Lasso mit EBIC alle 67 und 28 falsche.",
    "Zu wenig Wochen für die Teilkorrelation": "Dieselbe Kette mit nur 100 Wochen: das Schwellennetz hat 54 Kanten, davon 15 falsche (erwartet 53,0), das Teilkorrelationsnetz nur 11 der 39 echten (erwartet 9,2).",
    "Mehr Kennzahlen als Wochen": "30 Wochen für 40 Kennzahlen: das Teilkorrelationsnetz gibt es nicht; das Schwellennetz hat 117 Kanten bei 39 echten (erwartet 129,1), das beste λ des Graphical Lasso erreicht F1 0,67, die Rangfolge nach |r| 0,62.",
    "Zufallsnetz": "43 direkte Kanten mit beiden Vorzeichen: das Schwellennetz findet 26 (erwartet 24,9), das Teilkorrelationsnetz 39, der Graphical Lasso mit EBIC alle 43 und 14 falsche.",
}
