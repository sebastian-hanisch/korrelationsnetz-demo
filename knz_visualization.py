"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben.
Kanten unterscheiden sich in Farbe UND Linienart (richtig: voll, falsch: gepunktet und dicker, fehlend: gestrichelt), nicht nur in der Farbe."""

import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import knz_algorithm as K
import knz_constants as C

METHOD_LABELS = {"schwelle": "Schwellennetz", "teil": "Teilkorrelationsnetz", "ebic": "Graphical Lasso, EBIC", "best": "Graphical Lasso, bestes λ (kennt die Wahrheit)",
                 "rank": "Rangfolge nach |r| (kennt die Kantenzahl)"}
TRUTH_SHORT = {"kette": "Kette", "gitter": "Gitter", "zufall": "Zufallsnetz", "faktor": "Faktor"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="top", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _fmt(v: float) -> str:
    return f"{v:.3f}".replace(".", ",")


def node_positions(truth: str, p: int) -> np.ndarray:
    """Feste Lage der Knoten je Welt: das Gitter als Gitter, die Kette als Schlange durch dasselbe Raster (Zeile für Zeile, die Richtung wechselt: ein Weg, kein Ring), alle anderen auf
    einem Kreis (Kennzahl 0 oben, im Uhrzeigersinn)."""
    if truth in ("gitter", "kette"):
        cols = p // C.GRID_ROWS
        idx = np.arange(p)
        row, col = idx // cols, idx % cols
        if truth == "kette":
            col = np.where(row % 2 == 0, col, cols - 1 - col)
        return np.column_stack([col, -row]).astype(float)
    ang = np.pi / 2 - 2 * np.pi * np.arange(p) / p
    return np.column_stack([np.cos(ang), np.sin(ang)])


def _segments(A: np.ndarray, pos: np.ndarray):
    xs, ys = [], []
    for i, j in zip(*np.nonzero(np.triu(A, 1))):
        xs += [float(pos[i, 0]), float(pos[j, 0]), None]
        ys += [float(pos[i, 1]), float(pos[j, 1]), None]
    return xs, ys


def _add_net(fig, row, col, A, T, pos, show_legend):
    """Eine Tafel: richtige, falsche und fehlende Kanten (Wahrheit gegen Netz) und die Knoten."""
    kinds = (("richtig", A & T, dict(color=C.COLOR_RIGHT, width=2)), ("falsch", A & ~T, dict(color=C.COLOR_WRONG, width=1.5, dash="dot")),
             ("fehlend", ~A & T, dict(color=C.COLOR_MISSING, width=1.5, dash="dash")))
    for name, mask, line in kinds:
        xs, ys = _segments(mask, pos)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=line, name=name, legendgroup=name, showlegend=show_legend, hoverinfo="skip"), row=row, col=col)
    fig.add_trace(go.Scatter(x=pos[:, 0].tolist(), y=pos[:, 1].tolist(), mode="markers", marker=dict(size=6, color=C.COLOR_NODE), showlegend=False,
                             text=[f"Kennzahl {i + 1}" for i in range(len(pos))], hovertemplate="%{text}<extra></extra>"), row=row, col=col)


def build_networks(a, height: int = 640):
    """Wahrheit und die drei Netze auf denselben Daten, je mit richtigen, falschen und fehlenden Kanten."""
    s = a.settings
    pos = node_positions(s.truth, s.p)
    panels = [("Wahrheit: " + f"{K.edge_count(a.T)} echte Kanten", None), (f"Schwellennetz: {a.row('schwelle').edges} Kanten", "schwelle"),
              (f"Teilkorrelationsnetz: {a.row('teil').edges} Kanten" if "teil" in a.nets else "Teilkorrelationsnetz: gibt es hier nicht", "teil"),
              (f"Graphical Lasso: {a.row('glasso').edges} Kanten", "glasso")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[t for t, _ in panels], horizontal_spacing=0.04, vertical_spacing=0.12)
    shown = False
    for idx, (_title, key) in enumerate(panels):
        row, col = idx // 2 + 1, idx % 2 + 1
        if key is None:
            xs, ys = _segments(a.T, pos)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_RIGHT, width=2), showlegend=False, hoverinfo="skip"), row=row, col=col)
            fig.add_trace(go.Scatter(x=pos[:, 0].tolist(), y=pos[:, 1].tolist(), mode="markers", marker=dict(size=6, color=C.COLOR_NODE), showlegend=False,
                                     text=[f"Kennzahl {i + 1}" for i in range(s.p)], hovertemplate="%{text}<extra></extra>"), row=row, col=col)
        elif key in a.nets:
            _add_net(fig, row, col, a.nets[key], a.T, pos, show_legend=not shown)
            shown = True
        else:
            fig.add_annotation(text="n ≤ p + 1:<br>keine Inverse", x=0.5, y=0.5, xref="x domain" if idx == 0 else f"x{idx + 1} domain", yref=f"y{idx + 1} domain", showarrow=False, font=dict(size=13))
    for idx in range(4):
        xname = "x" if idx == 0 else f"x{idx + 1}"
        fig.update_xaxes(visible=False, constrain="domain", row=idx // 2 + 1, col=idx % 2 + 1)
        fig.update_yaxes(visible=False, scaleanchor=xname, scaleratio=1, row=idx // 2 + 1, col=idx % 2 + 1)
    fig.update_annotations(font_size=12)
    fig.update_layout(height=height, margin=dict(l=5, r=5, t=30, b=5), legend=dict(orientation="h", yanchor="top", y=-0.02), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_pairs(a, height: int = 360):
    """Jedes Paar als Punkt: |r| gegen |Teilkorrelation|; echte Kanten hervorgehoben, Schwelle τ und Bonferroni-Schwelle der Teilkorrelation als Linien."""
    s = a.settings
    iu = np.triu_indices(s.p, 1)
    r, pc, t = np.abs(a.R[iu]), np.abs(a.partial[iu]), a.T[iu]
    crit = math.tanh(K.bonferroni_z(s.p, s.alpha) / math.sqrt(K.partial_df(s.n, s.p)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r[~t].tolist(), y=pc[~t].tolist(), mode="markers", name="Paar ohne direkte Kante", marker=dict(size=5, color=C.COLOR_MISSING, opacity=0.5), hovertemplate="|r| %{x:.2f}, |Teilkorrelation| %{y:.2f}<extra></extra>"))
    if t.any():
        fig.add_trace(go.Scatter(x=r[t].tolist(), y=pc[t].tolist(), mode="markers", name="echte Kante", marker=dict(size=8, color=C.COLOR_RIGHT, symbol="diamond", line=dict(width=1, color="#ffffff")),
                                 hovertemplate="|r| %{x:.2f}, |Teilkorrelation| %{y:.2f}<extra>echt</extra>"))
    fig.add_vline(x=s.tau, line=dict(color=C.METHOD_COLORS["schwelle"], dash="dash"), annotation_text="τ", annotation_position="top")
    fig.add_hline(y=crit, line=dict(color=C.METHOD_COLORS["teil"], dash="dash"), annotation_text="Teilkorrelation: Schwelle", annotation_position="top left")
    fig.update_xaxes(title_text="|Korrelation r|", range=[0, 1])
    fig.update_yaxes(title_text="|Teilkorrelation|", range=[0, 1])
    return _base(fig, height)


def build_path(a, height: int = 420):
    """Graphical Lasso: Kantenzahl und F1 gegen λ (logarithmisch); Marken für den gewählten, den EBIC- und den besten Wert."""
    lams = a.lams
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Kanten", "F1 gegen die Wahrheit"))
    fig.add_trace(go.Scatter(x=lams.tolist(), y=a.path_edges, mode="lines+markers", name="Kanten", line=dict(color=C.COLOR_NODE), showlegend=False,
                             hovertemplate="λ %{x:.3f}: %{y} Kanten<extra></extra>"), row=1, col=1)
    k = K.edge_count(a.T)
    if k:
        fig.add_hline(y=k, line=dict(color=C.COLOR_EXACT, dash="dot"), annotation_text=f"echte Kanten: {k}", annotation_position="top left", row=1, col=1)
    fig.add_trace(go.Scatter(x=lams.tolist(), y=a.path_f1, mode="lines+markers", name="F1", line=dict(color=C.COLOR_NODE), showlegend=False, hovertemplate="λ %{x:.3f}: F1 %{y:.2f}<extra></extra>"), row=2, col=1)
    marks = [("EBIC", float(lams[a.i_ebic]), C.METHOD_COLORS["ebic"], "solid"), ("bestes λ", float(lams[a.i_best]), C.METHOD_COLORS["best"], "dash")]
    for label, lam, color, dash in marks:
        for row in (1, 2):
            fig.add_vline(x=lam, line=dict(color=color, dash=dash, width=2), row=row, col=1)
        fig.add_trace(go.Scatter(x=[lam], y=[None], mode="lines", name=label, line=dict(color=color, dash=dash, width=2)), row=1, col=1)
    fig.update_xaxes(type="log", title_text="λ (Strafe)", row=2, col=1)
    fig.update_yaxes(title_text="Kanten", row=1, col=1)
    fig.update_yaxes(title_text="F1", range=[0, 1.05], row=2, col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="top", y=-0.2), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def _method_value(cell: dict, which: str, gamma: float):
    key = f"ebic_{gamma}" if which == "ebic" else which
    m = cell["methods"].get(key)
    return None if m is None else m


def build_e1(cells: list, gamma: float, height: int = 380):
    """E1: F1 je Welt und Verfahren (gruppierte Balken); Teilkorrelation fehlt, wo es sie nicht gibt."""
    order = ["kette", "gitter", "zufall", "faktor"]
    by_truth = {c["truth"]: c for c in cells}
    fig = go.Figure()
    for which in ("schwelle", "teil", "ebic", "best", "rank"):
        vals = []
        for t in order:
            m = _method_value(by_truth[t], which, gamma)
            vals.append(None if m is None else m["f1"])
        fig.add_trace(go.Bar(x=[TRUTH_SHORT[t] for t in order], y=vals, name=METHOD_LABELS[which], marker_color=C.METHOD_COLORS[which], text=[None if v is None else _fmt(v) for v in vals], textposition="outside",
                             hovertemplate="%{x}: F1 %{y:.3f}<extra>" + METHOD_LABELS[which] + "</extra>"))
    fig.update_yaxes(title_text="F1 (1 = alle Kanten richtig, keine falsche)", range=[0, 1.15])
    fig.update_layout(barmode="group")
    return _base(fig, height)


def build_f1_vs_n(cells: list, truth: str, gamma: float, height: int = 360):
    """E2 und E3: F1 gegen die Zahl der Wochen für die fünf Verfahren (logarithmische Achse)."""
    mine = sorted((c for c in cells if c["truth"] == truth), key=lambda c: c["n"])
    fig = go.Figure()
    for which in ("schwelle", "teil", "ebic", "best", "rank"):
        pts = [(c["n"], m["f1"]) for c in mine if (m := _method_value(c, which, gamma)) is not None]
        if pts:
            fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines+markers", name=METHOD_LABELS[which], line=dict(color=C.METHOD_COLORS[which], dash="dash" if which in ("best", "rank") else "solid"),
                                     hovertemplate="n = %{x}: F1 %{y:.3f}<extra>" + METHOD_LABELS[which] + "</extra>"))
    fig.update_xaxes(type="log", title_text="Wochen n", tickvals=[c["n"] for c in mine], ticktext=[str(c["n"]) for c in mine])
    fig.update_yaxes(title_text="F1", range=[0, 1.05])
    return _base(fig, height)


def build_edges_vs_n(cells: list, truth: str, height: int = 360):
    """Kanten der beiden einfachen Netze gegen n: gemessen (Linie) und nach der Fisher-z-Näherung erwartet (Rauten); Linie der echten Kanten."""
    mine = sorted((c for c in cells if c["truth"] == truth), key=lambda c: c["n"])
    fig = go.Figure()
    ns = [c["n"] for c in mine]
    fig.add_trace(go.Scatter(x=ns, y=[c["methods"]["schwelle"]["edges"] for c in mine], mode="lines+markers", name="Schwellennetz, gemessen", line=dict(color=C.METHOD_COLORS["schwelle"])))
    fig.add_trace(go.Scatter(x=ns, y=[c["expected"]["schwelle"]["edges"] for c in mine], mode="markers", name="Schwellennetz, erwartet", marker=dict(symbol="diamond-open", size=11, color=C.COLOR_EXACT, line=dict(width=2))))
    with_teil = [c for c in mine if "teil" in c["methods"]]
    if with_teil:
        fig.add_trace(go.Scatter(x=[c["n"] for c in with_teil], y=[c["methods"]["teil"]["edges"] for c in with_teil], mode="lines+markers", name="Teilkorrelationsnetz, gemessen", line=dict(color=C.METHOD_COLORS["teil"])))
        fig.add_trace(go.Scatter(x=[c["n"] for c in with_teil], y=[c["expected"]["teil"]["edges"] for c in with_teil], mode="markers", name="Teilkorrelationsnetz, erwartet",
                                 marker=dict(symbol="diamond-open", size=11, color=C.COLOR_EXACT, line=dict(width=2))))
    if mine[0]["true_edges"]:
        fig.add_hline(y=mine[0]["true_edges"], line=dict(color=C.COLOR_EXACT, dash="dot"), annotation_text=f"echte Kanten: {mine[0]['true_edges']}", annotation_position="top left")
    fig.update_xaxes(type="log", title_text="Wochen n", tickvals=ns, ticktext=[str(n) for n in ns])
    fig.update_yaxes(title_text="Kanten")
    return _base(fig, height)


def build_e4(cell: dict, height: int = 400):
    """E4: Kanten und F1 im Mittel entlang des Pfads (λ als Anteil von λ_max), mit den im Mittel gewählten Werten von EBIC (γ = 0, 0,5, 1) und dem besten λ."""
    path = cell["path"]
    ratio = path["ratio"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Kanten (Mittel)", "F1 (Mittel)"))
    fig.add_trace(go.Scatter(x=ratio, y=path["edges"], mode="lines+markers", name="Kanten", line=dict(color=C.COLOR_NODE), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio, y=path["wrong"], mode="lines", name="davon falsch", line=dict(color=C.COLOR_WRONG, dash="dot"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=ratio, y=path["f1"], mode="lines+markers", name="F1", line=dict(color=C.COLOR_NODE), showlegend=False), row=2, col=1)
    if cell["true_edges"]:
        fig.add_hline(y=cell["true_edges"], line=dict(color=C.COLOR_EXACT, dash="dot"), annotation_text=f"echte Kanten: {cell['true_edges']}", annotation_position="top left", row=1, col=1)
    marks = [(f"EBIC γ = {g}".replace(".", ","), r, C.METHOD_COLORS["ebic"], d) for (g, r), d in zip(path["ebic_ratio"].items(), ("solid", "dot", "dashdot"))]
    marks.append(("bestes λ", path["best_ratio"], C.METHOD_COLORS["best"], "dash"))
    for label, r, color, dash in marks:
        for row in (1, 2):
            fig.add_vline(x=r, line=dict(color=color, dash=dash, width=2), row=row, col=1)
        fig.add_trace(go.Scatter(x=[r], y=[None], mode="lines", name=label, line=dict(color=color, dash=dash, width=2)), row=1, col=1)
    fig.update_xaxes(type="log", title_text="λ als Anteil von λ_max (links: starke Strafe)", autorange="reversed", row=2, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text="F1", row=2, col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="top", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_e5(cells: list, height: int = 360):
    """E5: Kanten des Schwellen- und des Teilkorrelationsnetzes gegen n in der Faktor-Welt (keine direkte Kante), mit der Zahl aller Paare."""
    cells = sorted(cells, key=lambda c: c["n"])
    ns = [c["n"] for c in cells]
    pairs = cells[0]["p"] * (cells[0]["p"] - 1) // 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=[c["methods"]["schwelle"]["edges"] for c in cells], mode="lines+markers", name="Schwellennetz", line=dict(color=C.METHOD_COLORS["schwelle"])))
    fig.add_trace(go.Scatter(x=ns, y=[c["methods"]["teil"]["edges"] for c in cells], mode="lines+markers", name="Teilkorrelationsnetz", line=dict(color=C.METHOD_COLORS["teil"])))
    fig.add_hline(y=pairs, line=dict(color=C.COLOR_EXACT, dash="dot"), annotation_text=f"alle Paare: {pairs}", annotation_position="bottom right")
    fig.update_xaxes(type="log", title_text="Wochen n", tickvals=ns, ticktext=[f"{n:,}".replace(",", " ") for n in ns])
    fig.update_yaxes(title_text="Kanten (alle falsch: es gibt keine direkte Kante)")
    return _base(fig, height)
