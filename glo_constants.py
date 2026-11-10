"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets.

Rechenaufwand am Slider-Maximum (DEMO-PLAYBOOK §1c): Wurzeln beider Modelle im Kontinuum O(1) (unter 1 ms: drei monotone Stücke, Bisektion), Bänder und Faltenkurve geschlossen;
endliches N: eine Auswertung der Startwahrscheinlichkeit kostet O(N·Knoten·N) (N = 100: 5 ms, N = 1 000: rund 50 ms), die Wurzelsuche braucht 81 bis 241 Auswertungen
(N = 1 000: rund 4 s, Spinner mit Zeitangabe); Beste-Antwort-Iteration: O(Runden) mit rund 5 µs je Runde (die engste gemessene Zelle brauchte 35 325 Runden, Obergrenze 200 000);
E-Mail-Spiel: Typen bis J = 400 (803 Zustände), eine Iteration O(J) je Runde, höchstens rund 800 Runden (unter 0,1 s)."""

# --- Modelle --------------------------------------------------------------------------------------------------------------------------------
MODELS = ("Regimewechsel (Modell R)", "Linear (Modell L)")
MODEL_R, MODEL_L = MODELS

# --- Regler ---------------------------------------------------------------------------------------------------------------------------------
Y_MIN, Y_MAX, DEFAULT_Y = 0.10, 0.90, 0.50           # Mittel des Priors für den Zustand θ (Mindestanteil, ab dem sich das Projekt lohnt)
ALPHA_MIN, ALPHA_MAX, DEFAULT_ALPHA = 0.5, 30.0, 4.0    # Genauigkeit des Priors (1/Varianz); Streuung = 1/√α. Bei α = 30, β = 0,1, y = ½ liegt die Randwurzel des Modells L unter 10^-300
BETA_MIN, BETA_MAX, DEFAULT_BETA = 0.1, 100.0, 16.0     # Genauigkeit des privaten Signals
C_MIN, C_MAX, DEFAULT_C = 0.05, 0.95, 0.50           # Aufwand der Umstellung, immer fällig (nur Modell R)
PLAYERS = (0, 2, 3, 5, 10, 25, 100, 250, 1000)       # 0 = Kontinuum; sonst Zahl der Betriebe N (nur Modell R: im Modell L ist der Erwartungsnutzen für jedes N gleich dem Kontinuumswert)
N_FINITE_MIN, N_FINITE_MAX = 2, 1000
SEED_MAX, DEFAULT_SEED = 999999, 35

# --- feste Annahmen -------------------------------------------------------------------------------------------------------------------------
DRAW_CONTINUUM = 100              # so viele Betriebe zeigt die gezeichnete Spielrunde im Kontinuum (die Wahrscheinlichkeit selbst ist exakt gerechnet)
DRAW_MAX = 400                    # höchstens so viele Punkte im Bild der Spielrunde (bei N = 1 000 wird alles gezählt, aber nur ein Teil gezeichnet)
RAT_BETAS = (1, 4, 16, 64, 256, 1024, 10000)       # E3: Signalgenauigkeiten der Reihe
E4_PLAYERS = (2, 3, 5, 10, 25, 100, 250, 1000)     # E4: Spielerzahlen der Tabelle
E5_FACTORS = (0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 1.02, 1.05, 1.1, 1.3, 2.0)    # E5: β als Vielfaches der Grenze α²/(2π), bei y = c = ½
E5_CAP = 200_000                  # Obergrenze der Runden der App (gemeldet als „abgebrochen“)
ROUNDS_PER_SECOND = 40_000        # grobe Schätzung für die Zeitangabe am Knopf (gemessen auf dem Entwicklungsrechner: 130 000 Runden je Sekunde)
EMAIL_J = 400                     # Typen bis 400 (Rest vorsichtig: greift nie an)
EMAIL_P_G = 0.5                   # Wahrscheinlichkeit des Zustands G (fest)
EMAIL_EPS_MIN, EMAIL_EPS_MAX, EMAIL_DEFAULT_EPS = 0.02, 0.50, 0.10
EMAIL_L_MIN, EMAIL_L_MAX, EMAIL_DEFAULT_L = 0.10, 3.00, 1.00       # Verlust bei einsamem Angriff; Gewinn M = 1
EMAIL_SIM_GAMES = 20_000
EMAIL_SIM_SEED = 11
EMAIL_PLOT_TYPES = 30             # so viele Typen zeigt das Bild der Überzeugungen
GRID_SEED_NOTE = "ohne Zufall (Wurzelsuche)"

# --- Farben ---------------------------------------------------------------------------------------------------------------------------------
COLOR_STABLE = "#2F6B65"          # stabiles Gleichgewicht
COLOR_UNSTABLE = "#C0762B"        # instabiles Gleichgewicht
COLOR_LINE = "#4c78a8"
COLOR_RED = "#d62728"
COLOR_GREY = "rgba(150,150,150,0.65)"

# --- Presets --------------------------------------------------------------------------------------------------------------------------------
BASE = {"model": MODEL_R, "y": DEFAULT_Y, "alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA, "c": DEFAULT_C, "players": 0, "seed": DEFAULT_SEED}
PRESETS = {
    "Genaue Signale (eindeutig)": dict(BASE),
    "Ungenaue Signale (drei)": {**BASE, "alpha": 8.0, "beta": 4.0},
    "Knapp daneben (c = 0,6)": {**BASE, "alpha": 8.0, "beta": 4.0, "c": 0.60},
    "Wenige Betriebe (N = 5)": {**BASE, "alpha": 8.0, "beta": 4.0, "c": 0.60, "players": 5},
    "V3-Zelle (Modell L)": {**BASE, "model": MODEL_L, "alpha": 4.0, "beta": 1.0},
}
PRESET_HELP = {
    "Genaue Signale (eindeutig)": "Modell R, Kontinuum, α = 4, β = 16, y = c = ½. β liegt über α²/(2π) = 2,55: ein einziges Gleichgewicht, das Projekt startet für θ ≤ 0,5 (x* = 0,5).",
    "Ungenaue Signale (drei)": "Modell R, Kontinuum, α = 8, β = 4, y = c = ½. β liegt unter α²/(2π) = 10,19: drei Gleichgewichte bei x* = −0,910, 0,500 und 1,910, das mittlere ist instabil. Das Band in c mit drei Gleichgewichten reicht von 0,416 bis 0,584.",
    "Knapp daneben (c = 0,6)": "Wie „Ungenaue Signale“, aber c = 0,6 liegt knapp über dem oberen Rand 0,584 des Bandes: nur noch ein Gleichgewicht, und zwar das mit kaum einem Mitmachenden (x* = −1,195, θ̂ = 0,008).",
    "Wenige Betriebe (N = 5)": "Dieselben Einstellungen mit N = 5 Betrieben statt Kontinuum: drei Gleichgewichte (x* = −0,510, 0,781 und 1,591), obwohl das Kontinuum nur eines hat (Experiment E4). Auf einem Raster gezählt, kein Beweis.",
    "V3-Zelle (Modell L)": "Modell L, α = 4, β = 1, y = ½: drei Gleichgewichte (Anteil m = 0,050, 0,5 und 0,950). Die Faustformel α²/(2π) = 2,55 liegt über β = 1; die wahre Grenze bei y = ½ liegt bei β = 1,92.",
}
