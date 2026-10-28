"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b). Alle Erwartungswerte stehen als Rechnung dabei."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def lin2():
    """Modell L, α = 2, β = 1, y = ½: κ² = α²(α+β)/(β(α+2β)) = 4·3/(1·4) = 3, κ = 1,7321. m = ½ löst Φ(κ(m − y)) = m (Φ(0) = ½), also x* = (½·3 − 2·½)/1 = ½.
    Kritisches β bei y = ½ (κ² = 2π): 4πβ² + (2πα − α²)β − α³ = 4πβ² + (4π − 4)β − 8 = 0, β = (−(4π − 4) + √((4π − 4)² + 128π))/(8π) = 0,5268."""
    return {"alpha": 2.0, "beta": 1.0, "y": 0.5, "kappa": math.sqrt(3.0), "m_mid": 0.5, "x_mid": 0.5, "beta_crit": 0.5268}


@pytest.fixture
def reg44():
    """Modell R, α = β = 4, y = c = ½: ρ = α/√β = 2 < √(2π) = 2,5066 (eindeutig), z_c = Φ⁻¹(½) = 0, d = (αy + 0)/√β = 2/2 = 1.
    G(w) = 2Φ(w) − w = 1 hat w = 0 als Lösung (2·½ − 0 = 1), also θ̂ = Φ(0) = ½ und x* = θ̂ + w/√β = ½."""
    return {"alpha": 4.0, "beta": 4.0, "y": 0.5, "c": 0.5, "rho": 2.0, "d": 1.0, "w": 0.0, "theta": 0.5, "x": 0.5}


@pytest.fixture
def reg84():
    """Modell R, α = 8, β = 4, y = c = ½: ρ = 8/2 = 4 > 2,5066, w₀ = √(2 ln(4/2,5066)) = √(2·0,4672) = 0,9667, Φ(w₀) = 0,8331.
    G(−w₀) = 4·(1 − 0,8331) + 0,9667 = 1,6343, G(w₀) = 4·0,8331 − 0,9667 = 2,3657; d = (8·½ + 0)/2 = 2 liegt im Band (1,634; 2,366): drei Gleichgewichte.
    Rand in c: d = 2,3657 → z_c = (d√β − αy)/√(α+β) = (4,7314 − 4)/3,4641 = 0,2111, Φ(0,2111) = 0,5836 (genauer 0,5837); symmetrisch unten 0,4163. Mittlere Wurzel w = 0 (G(0) = 4·½ = 2 = d)."""
    return {"alpha": 8.0, "beta": 4.0, "y": 0.5, "c": 0.5, "rho": 4.0, "d": 2.0, "w0": 0.9667, "band": (1.6343, 2.3657), "c_upper": 0.5837, "c_lower": 0.4163}


@pytest.fixture
def finite_hand():
    """Endliches N, von Hand. N = 2, θ = 0,7, p = 0,3: Start nur bei K = 1 ((1 + K)/2 ≥ 0,7 ⟺ K ≥ 0,4), also P(K = 1) = 0,3.
    N = 3, θ = ½, p = ½: (1 + K)/3 ≥ ½ ⟺ K ≥ 0,5, also K ≥ 1 mit K ~ Bin(2, ½): 1 − (½)² = 0,75. θ ≤ 1/N: der eigene Beitrag genügt, Erfolg sicher (z. B. N = 3, θ = 0,3, p beliebig)."""
    return {"n2": (2, 0.7, 0.3, 0.3), "n3": (3, 0.5, 0.5, 0.75), "sure": (3, 0.3, 0.1, 1.0)}


@pytest.fixture
def mail():
    """E-Mail-Spiel, ε = 0,1. k = Zahl gesendeter Nachrichten, P(k) ∝ ε(1−ε)^(k−1). Typ t₁ = ⌈k/2⌉, t₂ = ⌊k/2⌋.
    P(t₂ = 0 | t₁ = 1) = P(k = 1)/(P(k = 1) + P(k = 2)) = ε/(ε + ε(1−ε)) = 1/(2 − ε) = 1/1,9 = 0,5263; die Gegenwahrscheinlichkeit P(k = 2 | t₁ = 1) = (1−ε)/(2−ε) = 0,9/1,9 = 0,4737 = q*.
    P(k ≥ 2 | G) = 1 − ε = 0,9. Auszahlung von Spieler 1 im Profil „angreifen ab Typ 1“ bei L = M = 1, p_G = ½: in G greift er immer an (t₁ ≥ 1), der andere (t₂ ≥ 1) nur bei k ≥ 2:
    ½·(0,9·1 − 0,1·1) = 0,4."""
    return {"eps": 0.1, "p_t2_zero": 1 / 1.9, "q_star": 0.9 / 1.9, "p_k2": 0.9, "payoff1": 0.4}
