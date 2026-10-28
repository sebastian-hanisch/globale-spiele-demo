"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Gleichgewichte und die Verdikte. Erst nach Klick erzeugt (DEMO-PLAYBOOK §1c)."""

from __future__ import annotations

from fpdf import FPDF

import glo_evaluation as V

_REPLACE = {"–": "-", "—": "-", "≤": "<=", "≥": ">=", "≠": "!=", "√": "Wurzel", "²": "2", "Σ": "Summe", "°": " Grad", "·": "*", "−": "-", "„": '"', "“": '"', "∞": "unendlich",
            "→": "->", "ε": "eps", "β": "beta", "α": "alpha", "θ": "theta", "κ": "kappa", "ρ": "rho", "λ": "lambda", "σ": "sigma", "π": "pi", "Φ": "Phi", "Δ": "Delta", "×": "x", "↔": "<->",
            "≈": "~", "∝": "~", "‖": "||", "ℓ": "l", "½": "1/2", "⁻": "^-", "¹": "1", "⁵": "5", "⁶": "6", "⁰": "0", "∩": "und", "↦": "->", "ˆ": "^", "̂": "^", "⁴": "4", "³": "3"}


def _latin(text: str) -> str:
    """Kernschriften kennen nur Latin-1: Sonderzeichen ersetzen, Markdown-Sterne entfernen."""
    for a, b in _REPLACE.items():
        text = text.replace(a, b)
    return text.replace("**", "").replace("`", "").encode("latin-1", "replace").decode("latin-1")


def build_pdf(view: V.MainView, verdicts: list[tuple[str, str]]) -> bytes:
    s = view.settings
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _latin("Unsicherheit und Gleichgewicht: globale Spiele"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    c_text = f", Aufwand c = {V.de(s.c, 2)}" if s.is_regime else ""
    pdf.multi_cell(0, 5, _latin(f"{s.model}: Prior-Mittel y = {V.de(s.y, 2)}, Prior-Genauigkeit alpha = {V.de(s.alpha, 1)}, Signal-Genauigkeit beta = {V.de(s.beta, 1)}{c_text}, "
                                f"{V.players_text(s.players) if s.is_regime else 'Kontinuum'}. Zufalls-Seed {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Gleichgewichte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    level = "theta_hat" if s.is_regime else "m"
    for i, row in enumerate(view.rows, 1):
        lv = "" if row.level is None else f", {level} = {V.de(row.level, 4)}"
        pdf.multi_cell(0, 5, _latin(f"{i}. {row.label}: Schwelle x* = {V.fmt_x(row.x)}{lv} ({'stabil' if row.stable else 'instabil'})"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Verdikte", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for _, text in verdicts:
        pdf.multi_cell(0, 5, _latin(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _latin("Eigennutz, Schwellenstrategien und die Beste-Antwort-Iteration sind Modellannahmen, keine Aussagen über Menschen oder echte Terminals. Alle Zahlen sind eigene Rechnung, "
                                "durch Zählung, zweite Rechenwege und Simulation gestützt, nicht gegen Literatur geprüft."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
