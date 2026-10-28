"""E-Mail-Spiel: gemeinsames Wissen als Kette von Bestätigungen. Eigene Modellform, angelehnt an Rubinstein, nicht gegen die Originalarbeit geprüft.

Zustand G (Koordination lohnt) mit Wahrscheinlichkeit p_G, sonst B. In G schickt Spieler 1 eine Nachricht; jede Nachricht geht mit Wahrscheinlichkeit ε verloren,
jede angekommene löst automatisch eine Bestätigung aus, die ebenfalls verloren gehen kann. k sei die Zahl der gesendeten Nachrichten (die k-te geht verloren);
P(k) = p_G · ε (1−ε)^(k−1). Typ von Spieler 1: t₁ = ⌈k/2⌉, von Spieler 2: t₂ = ⌊k/2⌋ (in B: beide 0). Beide Spieler wählen „angreifen“ oder „nicht“.
Angriff bringt M, wenn beide angreifen und der Zustand G ist; sonst −L (allein oder im Zustand B); „nicht angreifen“ bringt 0. Ein Spieler greift also genau dann an, wenn die
bedingte Wahrscheinlichkeit „G und der andere greift an“ mindestens q = L/(L+M) ist.

Eigene Herleitung, durch Aufzählung aller reinen Profile (J = 3) und Zählung gestützt: das größte Gleichgewicht (Iteration ab „alle greifen an“) enthält einen Angriff genau dann, wenn
q ≤ max(λ₀, (1−ε)/(2−ε)) mit λ₀ = p_G ε/(1 − p_G + p_G ε) (Typ 0 von Spieler 2 sieht nur die erste Nachricht verloren); bei p_G = ½ und ε ≤ ½ ist das (1−ε)/(2−ε).
Die Typenkette wird bei J (Typ 400) abgeschnitten; der Rest („mehr als 2J+1 Nachrichten“) bekommt eigene Typen, die nie angreifen (vorsichtiger Rand).
Beliefs werden bedingt auf den Typ gerechnet, nie als unbedingte Erwartungsnutzen verglichen: Typen mit Wahrscheinlichkeit unter 10^-15 sähen sonst wie Angriffe aus."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from glo_rng import _GAMMA, _MASK, uniform_block

J_DEFAULT = 400
TOL = 1e-12


@dataclass(frozen=True)
class EmailGame:
    eps: float
    p_g: float
    j: int
    prob: np.ndarray         # Wahrscheinlichkeit je Zustand: 0 = B, 1..2J+1 = G mit k Nachrichten, letzter = Rest (mehr als 2J+1)
    t1: np.ndarray           # Typ von Spieler 1 je Zustand
    t2: np.ndarray
    is_g: np.ndarray         # Zustand G (der Rest zählt als G)
    n1: int                  # Zahl der Typen von Spieler 1 (der letzte ist der Rest-Typ)
    n2: int


def build(eps: float, p_g: float, j: int = J_DEFAULT) -> EmailGame:
    """Zustandsraum mit Typen bis J (Spieler 2) bzw. J+1 (Spieler 1) und einem Rest-Zustand mit eigenen Typen."""
    kmax = 2 * j + 1
    ks = np.arange(1, kmax + 1)
    pk = p_g * eps * (1.0 - eps) ** (ks - 1)
    tail = p_g * (1.0 - eps) ** kmax                    # exakt: mehr als kmax Nachrichten
    prob = np.concatenate([[1.0 - p_g], pk, [tail]])
    t1 = np.concatenate([[0], (ks + 1) // 2, [j + 2]]).astype(int)
    t2 = np.concatenate([[0], ks // 2, [j + 1]]).astype(int)
    is_g = np.concatenate([[False], np.ones(kmax, bool), [True]])
    return EmailGame(eps, p_g, j, prob, t1, t2, is_g, int(t1.max()) + 1, int(t2.max()) + 1)


def q_of(loss: float, gain: float = 1.0) -> float:
    """Schwelle q = L/(L+M): so sicher muss ein Spieler sein, dass G gilt und der andere angreift."""
    return loss / (loss + gain)


def belief(game: EmailGame, own: int, s_other: np.ndarray) -> np.ndarray:
    """Bedingte Wahrscheinlichkeit „Zustand G und der andere greift an“ je eigenem Typ. own = 1 oder 2; s_other: Strategie des anderen je Typ (bool)."""
    own_t, other_t, n_own = (game.t1, game.t2, game.n1) if own == 1 else (game.t2, game.t1, game.n2)
    hit = game.prob * (game.is_g & s_other[other_t])
    num = np.bincount(own_t, weights=hit, minlength=n_own)
    tot = np.bincount(own_t, weights=game.prob, minlength=n_own)
    return num / tot


def best_response(game: EmailGame, own: int, s_other: np.ndarray, q: float) -> np.ndarray:
    """Angriff genau dann, wenn die bedingte Wahrscheinlichkeit mindestens q ist; der Rest-Typ greift nie an."""
    s = belief(game, own, s_other) >= q - TOL
    s[-1] = False
    return s


def is_equilibrium(game: EmailGame, s1: np.ndarray, s2: np.ndarray, q: float) -> bool:
    """Ist (s1, s2) ein Gleichgewicht? Jeder Typ wählt seine beste Antwort; der Rest-Typ greift nie an (Modellrand)."""
    b1, b2 = belief(game, 1, s2), belief(game, 2, s1)
    ok1 = np.all(np.where(s1, b1 >= q - TOL, b1 <= q + TOL)[:-1])
    ok2 = np.all(np.where(s2, b2 >= q - TOL, b2 <= q + TOL)[:-1])
    return bool(ok1 and ok2 and not s1[-1] and not s2[-1])


@dataclass(frozen=True)
class EmailEquilibrium:
    s1: np.ndarray
    s2: np.ndarray
    rounds: int              # Runden der Beste-Antwort-Iteration bis zum Fixpunkt (gleichzeitig, ab „alle greifen an“)
    belief1: np.ndarray      # bedingte Wahrscheinlichkeit „G und der andere greift an“ im Gleichgewicht, je Typ
    belief2: np.ndarray

    @property
    def attacks(self) -> bool:
        return bool(self.s1.any() or self.s2.any())

    @property
    def first1(self) -> int | None:
        return int(np.argmax(self.s1)) if self.s1.any() else None

    @property
    def first2(self) -> int | None:
        return int(np.argmax(self.s2)) if self.s2.any() else None


def greatest_equilibrium(game: EmailGame, q: float, max_rounds: int = 100_000) -> EmailEquilibrium:
    """Das größte Gleichgewicht: Beste-Antwort-Iteration ab „alle greifen an“ (die Antwort ist monoton, die Folge fällt)."""
    s1 = np.ones(game.n1, bool)
    s2 = np.ones(game.n2, bool)
    s1[-1] = s2[-1] = False
    for rounds in range(max_rounds):
        n1 = best_response(game, 1, s2, q)
        n2 = best_response(game, 2, s1, q)
        if np.array_equal(n1, s1) and np.array_equal(n2, s2):
            return EmailEquilibrium(s1, s2, rounds, belief(game, 1, s2), belief(game, 2, s1))
        s1, s2 = n1, n2
    raise RuntimeError("keine Konvergenz")


def q_star(eps: float) -> float:
    """(1−ε)/(2−ε): bedingte Wahrscheinlichkeit, dass die nächste Nachricht der Kette noch ankommt, gegeben die Stufe davor; unabhängig von der Stufe."""
    return (1.0 - eps) / (2.0 - eps)


def first_message_lost(eps: float, p_g: float) -> float:
    """λ₀ = p_G ε/(1 − p_G + p_G ε): Wahrscheinlichkeit von „G, und es kam keine Nachricht an“ für Spieler 2 ohne Nachricht (Typ 0)."""
    return p_g * eps / (1.0 - p_g + p_g * eps)


def attack_threshold(eps: float, p_g: float) -> float:
    """Größtes q, bei dem das größte Gleichgewicht noch einen Angriff enthält: max(λ₀, (1−ε)/(2−ε))."""
    return max(first_message_lost(eps, p_g), q_star(eps))


def knowledge_level_prob(eps: float, n: int) -> float:
    """Wahrscheinlichkeit (bedingt auf G), dass mindestens n Nachrichten angekommen sind: (1−ε)^n (gegenseitiges Wissen der Stufe n)."""
    return (1.0 - eps) ** n


# ---------------------------------------------------------------- gemeinsame p-Überzeugung
def p_believes(game: EmailGame, own: int, member: np.ndarray, p: float) -> np.ndarray:
    """Typen von Spieler `own`, die das Ereignis `member` (bool je Zustand) mit Wahrscheinlichkeit ≥ p für wahr halten."""
    own_t, n_own = (game.t1, game.n1) if own == 1 else (game.t2, game.n2)
    num = np.bincount(own_t, weights=game.prob * member, minlength=n_own)
    tot = np.bincount(own_t, weights=game.prob, minlength=n_own)
    return num / tot >= p - TOL


def common_p_belief(game: EmailGame, p: float, event: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Gemeinsame p-Überzeugung des Ereignisses (Standard: G ohne den Rest-Zustand): der größte Fixpunkt von F ↦ F ∩ B₁ᵖ(F) ∩ B₂ᵖ(F).

    Rückgabe: (Zustände, in denen das Ereignis gemeinsame p-Überzeugung ist, Zahl der Operator-Anwendungen, die F noch verkleinert haben)."""
    f = (game.is_g.copy() if event is None else event.copy())
    f[-1] = False
    rounds = 0
    while f.any():
        b1, b2 = p_believes(game, 1, f, p), p_believes(game, 2, f, p)
        new = f & b1[game.t1] & b2[game.t2]
        if np.array_equal(new, f):
            break
        f, rounds = new, rounds + 1
    return f, rounds


# ---------------------------------------------------------------- Protokoll
def messages_to_types(k: int) -> tuple[int, int]:
    """(t₁, t₂) zu k gesendeten Nachrichten; k = 0 steht für den Zustand B."""
    return (k + 1) // 2, k // 2


def _block_seed(seed: int, offset: int) -> int:
    """Seed des Stroms ab Position `offset` (SplitMix64 ist ein Zähler: Seed + offset·γ)."""
    return (seed + offset * _GAMMA) & _MASK


def simulate_protocol(eps: float, p_g: float, games: int, seed: int, max_messages: int = 100_000) -> dict:
    """Das Protokoll Nachricht für Nachricht, `games` Spiele gleichzeitig mit SplitMix64-Strömen: Zustand (G mit Wahrscheinlichkeit p_G), dann je Nachricht ein Zug,
    ob sie verloren geht. Rückgabe: Zahl gesendeter Nachrichten k (0 in B) und Zustand je Spiel."""
    is_g = uniform_block(seed, games) < p_g
    alive = is_g.copy()
    k = np.zeros(games, dtype=np.int64)
    r = 1
    while alive.any():
        if r > max_messages:
            raise RuntimeError("Protokoll endet nicht")
        u = uniform_block(_block_seed(seed, r * games), games)
        k += alive
        alive &= u >= eps                  # die Nachricht ging mit Wahrscheinlichkeit ε verloren: der Austausch endet
        r += 1
    return {"is_g": is_g, "k": k}


def draw_game(eps: float, p_g: float, rng) -> tuple[bool, int, list[bool]]:
    """Ein einzelnes Spiel mit einem SplitMix64-Objekt: (Zustand G?, k, Liste „Nachricht i kam an“ für i = 1..k−1 plus die verlorene)."""
    is_g = rng.uniform() < p_g
    if not is_g:
        return False, 0, []
    arrived: list[bool] = []
    while True:
        lost = rng.uniform() < eps
        arrived.append(not lost)
        if lost:
            return True, len(arrived), arrived
