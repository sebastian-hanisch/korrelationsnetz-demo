"""PDF-Export (fpdf2, Kernschriften): Einstellungen, Tabelle der Netze mit Kanten, richtig, falsch, fehlend, F1 und Transitivität, Verdikte der Hauptansicht.
Kein Gedankenstrich und kein Euro-Zeichen (Kernschriften)."""

from fpdf import FPDF

import knz_algorithm as K
import knz_constants as C
import knz_evaluation as ev


def _text(value: str) -> str:
    """Zeichen, die die Kernschriften nicht kennen, ersetzen (Gedankenstrich, Pfeile, Fettdruck-Markierung, griechische und mathematische Zeichen)."""
    table = {"–": "-", "—": "-", "→": "->", "**": "", "€": "EUR", "·": "*", "≤": "<=", "≥": ">=", "−": "-", "×": "x", "„": '"', "“": '"', "τ": "tau", "α": "alpha", "γ": "gamma", "λ": "lambda",
             "ρ": "rho", "σ": "sigma", "²": "^2", "√": "sqrt ", "…": "...", "π": "pi"}
    for old, new in table.items():
        value = value.replace(old, new)
    return value.encode("latin-1", "replace").decode("latin-1")


def _para(pdf: FPDF, height: float, text: str) -> None:
    """Absatz über die volle Breite; danach steht der Cursor wieder am linken Rand."""
    pdf.multi_cell(0, height, _text(text), new_x="LMARGIN", new_y="NEXT")


def build_pdf(a: ev.Analysis) -> bytes:
    s = a.settings
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _text("Korrelationsnetz und Teilkorrelation: welche Kanten sind direkt?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    _para(pdf, 5.5, f"Welt: {C.TRUTHS[s.truth]}   Kennzahlen p: {s.p}   Wochen n: {s.n}   Schwelle tau: {s.tau:g}   alpha: {s.alpha:g}   Graphical Lasso: {C.LAMBDA_RULES[s.rule]}"
                    f"{f' ({s.lam:g})' if s.rule == 'fest' else ''}   EBIC-Gewicht gamma: {s.gamma:g}   Zufalls-Seed: {s.seed}")
    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Netze (gegen die Wahrheit mit " + f"{K.edge_count(a.T)} echten Kanten)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in a.rows:
        exp = "" if r.expected is None else f", erwartet {r.expected['edges']:.1f} Kanten"
        _para(pdf, 5.5, f"{r.name}: {r.edges} Kanten, {r.right} richtig, {r.wrong} falsch, {r.missing} fehlend, F1 {r.f1:.3f}, Transitivitaet {r.trans:.3f}{exp}")
    if "teil" not in a.nets:
        _para(pdf, 5.5, f"Teilkorrelationsnetz: gibt es bei n = {s.n} und p = {s.p} nicht (n <= p + 1).")
    for title, verdict in (("Ergebnis", ev.verdict_main), ("Schwellennetz", ev.verdict_threshold), ("Teilkorrelationsnetz", ev.verdict_partial), ("Graphical Lasso", ev.verdict_glasso)):
        v = verdict(a)
        pdf.ln(1)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, _text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        _para(pdf, 5.5, v[1])
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 8)
    _para(pdf, 4.5, "Simulierte Daten mit bekannter Welt. Sebastian Hanisch - Operations Research und Machine Learning.")
    return bytes(pdf.output())
