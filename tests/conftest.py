"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def chain3():
    """Drei Kennzahlen A–B–C mit r_AB = r_BC = 0,5 und r_AC = 0,25: ρ_AC·B = (0,25 − 0,25) / (…) = 0, die Präzisionsmatrix ist tridiagonal
    (1/(1 − 0,25) · [[1, −0,5, 0], [−0,5, 1,25, −0,5], [0, −0,5, 1]]); Teilkorrelation der Nachbarn ρ_AB·C = 0,5 / √1,25 = 0,4472."""
    return np.array([[1.0, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 1.0]])


@pytest.fixture
def factor3():
    """Drei Kennzahlen mit einem Faktor, alle r = 0,36: ρ = (0,36 − 0,36²) / (1 − 0,36²) = 0,2304 / 0,8704 = 0,26471."""
    R = np.full((3, 3), 0.36)
    np.fill_diagonal(R, 1.0)
    return R


@pytest.fixture
def two_vars():
    """Zwei Kennzahlen mit r = 0,6: Graphical Lasso mit λ = 0,2 gibt W₁₂ = r − λ = 0,4 (Diagonale 1), Θ = W⁻¹ = 1/0,84 · [[1, −0,4], [−0,4, 1]], Teilkorrelation 0,4;
    bei λ ≥ 0,6 ist Θ die Einheitsmatrix."""
    return np.array([[1.0, 0.6], [0.6, 1.0]])
