"""Korrelationsnetz und Teilkorrelation: welche Kanten sind direkt? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Sechstes Stück der Korrelations-Reihe. p Kennzahlen über n Wochen aus einer Welt mit bekannten direkten Zusammenhängen (Kette, Gitter, Zufallsnetz, verborgener Faktor). Daraus werden drei Netze gebaut
(Schwelle auf |r|, Teilkorrelation mit Fisher-z und Bonferroni, Graphical Lasso) und mit der Wahrheit verglichen: richtige, falsche und fehlende Kanten, Transitivität, die exakte Erwartung.

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import knz_algorithm as K
import knz_constants as C
import knz_evaluation as ev
from knz_layout import dunkles_design, kopf_css
from knz_pdf_export import build_pdf
from knz_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from knz_visualization import (
    METHOD_LABELS,
    TRUTH_SHORT,
    build_e1,
    build_e4,
    build_e5,
    build_edges_vs_n,
    build_f1_vs_n,
    build_networks,
    build_pairs,
    build_path,
)

st.set_page_config(page_title="Korrelationsnetz und Teilkorrelation – Sebastian Hanisch", layout="wide")

de = ev.de


def n_text(v: float) -> str:
    return f"{v:g}".replace(".", ",")


@st.cache_data(show_spinner=False)
def _path(truth, p, n, seed):
    return ev.compute_path(truth, p, n, seed)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings, _path(settings.truth, settings.p, settings.n, settings.seed))


@st.cache_data(show_spinner=False)
def _grid():
    return ev.load_grid()


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in knz_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🕸️ Korrelationsnetz und Teilkorrelation: welche Kanten sind direkt?")
st.markdown(
    """
**Sechstes Stück der Korrelations-Reihe** und **die erste Demo des Portfolios, in der ein Netz aus Daten gebaut wird**. **p Kennzahlen** werden über **n Wochen** gemessen; dahinter steht eine Welt, in der
bekannt ist, welche Kennzahlen **direkt** zusammenhängen. Aus den Daten entstehen drei Netze: eines aus allen Paaren mit großem |r| (**Schwelle**), eines aus den **Teilkorrelationen** (gegeben alle übrigen)
und eines vom **Graphical Lasso**, der nur so viele Kanten zulässt, wie die Strafe λ erlaubt. Die Fragen: **Was zeigt ein Korrelationsnetz? Wann sind seine Kanten direkt? Und wo schneidet man ab?**

Weiter unten: fünf Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**. Ursachen und Pfeilrichtungen aus Daten (PC-Algorithmus) behandelt `strukturlernen-demo`
der Kausal-Linie; die **Matrix** selbst (Spektrum, Schrumpfung) `korrelationsmatrix-demo`; Netze, die schon da sind, die Netz-Linie. Hier geht es um das **Netz aus Daten**.
"""
)

with st.expander("So funktioniert es", expanded=True):
    st.markdown(
        r"""
| Begriff | Was er hier bedeutet |
|---|---|
| **Welt** | Vier Beispiele mit bekannter Struktur: **Kette** (jede Kennzahl hängt direkt nur an ihren Nachbarn), **Gitter**, **Zufallsnetz** und **Faktor** (alle hängen an einer verborgenen Ursache, keine direkte Kante). |
| **Direkte Kante** | Zwei Kennzahlen hängen auch dann noch zusammen, wenn man alle übrigen festhält (Teilkorrelation ungleich null). |
| **Schwellennetz** | Kante, wenn \|r\| größer als die Schwelle τ ist. Es zeigt auch **indirekte** Zusammenhänge und **gemeinsame Ursachen** als Kanten. |
| **Teilkorrelationsnetz** | Kante, wenn die Teilkorrelation (aus der Inversen der Stichprobenmatrix) nach Fisher-z und Bonferroni nicht null ist. Es braucht **n > p + 1** und viele Wochen. |
| **Graphical Lasso** | Schätzt die Inverse mit einer Strafe λ auf alle Außerdiagonalen; großes λ gibt wenige Kanten. **Welches λ?** EBIC wählt es aus den Daten, das **beste λ** kennt die Wahrheit. |
| **Rangfolge** | Die k Paare mit dem größten \|r\|, wobei k die Zahl der echten Kanten ist: **kennt die Wahrheit**, nur als Messlatte. |
| **Richtig, falsch, fehlend, F1** | Gegen die Wahrheit gezählt; F1 = 2 · richtig / (2 · richtig + falsch + fehlend), 1 ist perfekt. Ohne echte Kante ist ein leeres Netz F1 = 1. |
| **Transitivität** | Anteil geschlossener Dreiecke unter allen verbundenen Dreiern: ein Netz aus gemeinsamer Ursache ist dicht und voller Dreiecke. |
| **Erwartet** | Die Zahl der Kanten der beiden einfachen Netze lässt sich aus der Fisher-z-Näherung **vorher** ausrechnen. |
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Welt**")
    truth = st.selectbox("Welt", list(C.TRUTHS), format_func=C.TRUTHS.get, key="truth_select", help="Wie die Kennzahlen direkt zusammenhängen. Faktor: keine direkte Kante, nur eine verborgene gemeinsame Ursache.")
    p = st.select_slider("Kennzahlen p", options=C.P_OPTIONS, key="p_select", help="Zahl der Kennzahlen; das Gitter hat 5 Zeilen. Mehr als 40 würde den Graphical Lasso zu langsam machen.")
    n = st.select_slider("Wochen n", options=C.N_OPTIONS, key="n_select", help="Zahl der Beobachtungen. Bei n ≤ p + 1 gibt es kein Teilkorrelationsnetz.")
    st.markdown("**Netze**")
    tau = st.slider("Schwelle τ für |r|", *bounds("tau_slider"), step=0.05, key="tau_slider", format="%.2f", help="Schwellennetz: Kante, wenn |r| größer als τ ist.")
    alpha = st.select_slider("Irrtumswahrscheinlichkeit α", options=C.ALPHAS, key="alpha_select", help="Teilkorrelationsnetz: Fisher-z-Test mit Bonferroni über alle Paare.")
    rule = st.selectbox("λ des Graphical Lasso", list(C.LAMBDA_RULES), format_func=C.LAMBDA_RULES.get, key="rule_select",
                        help="EBIC wählt λ aus den Daten, 'bestes λ' kennt die Wahrheit (nur eine Obergrenze), 'fester Wert' nimmt den Regler darunter.")
    lam = st.slider("Fester Wert λ", *bounds("lam_slider"), step=0.01, key="lam_slider", format="%.2f", help="Wird nur bei der Regel 'fester Wert' verwendet.")
    gamma = st.select_slider("EBIC-Gewicht γ", options=C.GAMMAS, key="gamma_select", help="Strafe des EBIC für jede Kante: 0 ist das gewöhnliche BIC, größer wählt weniger Kanten.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Daten ziehen", width="stretch", on_click=randomize_seed)

sync_query_params({"truth_select": truth, "p_select": int(p), "n_select": int(n), "tau_slider": float(tau), "alpha_select": float(alpha), "rule_select": rule, "lam_slider": float(lam),
                   "gamma_select": float(gamma), "seed_input": int(seed)})

settings = ev.Settings(truth=truth, p=int(p), n=int(n), tau=float(tau), alpha=float(alpha), rule=rule, lam=float(lam), gamma=float(gamma), seed=int(seed))
with st.spinner("Rechne den Pfad des Graphical Lasso (eine bis wenige Sekunden, nur bei neuen Daten) …"):
    a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
k_true = K.edge_count(a.T)
pairs = settings.p * (settings.p - 1) // 2
st.caption(f"Wahrheit: **{k_true}** direkte Kanten von {pairs} möglichen (Transitivität {de(a.trans_true)})." if k_true else f"Wahrheit: **keine** direkte Kante unter {pairs} möglichen Paaren.")
cols = st.columns(2)
for col, key, label in ((cols[0], "schwelle", "Schwellennetz"), (cols[1], "teil", "Teilkorrelationsnetz")):
    r = a.row(key)
    col.metric(label + " (F1)", de(r.f1) if r else "–", help=("Nicht berechenbar: n ≤ p + 1." if r is None else f"{r.edges} Kanten, {r.right} richtig, {r.wrong} falsch, {r.missing} fehlend; erwartet {de(r.expected['edges'], 1)} Kanten."))
cols = st.columns(2)
for col, key, label in ((cols[0], "glasso", "Graphical Lasso"), (cols[1], "rank", "Rangfolge (kennt k)")):
    r = a.row(key)
    col.metric(label + " (F1)", de(r.f1), help=f"{r.edges} Kanten, {r.right} richtig, {r.wrong} falsch, {r.missing} fehlend.")
st.plotly_chart(build_networks(a), width="stretch", key="main_networks")
st.caption("Richtige Kanten: durchgezogen, falsche: gepunktet, echte, aber nicht gefundene: gestrichelt. Das Gitter ist als Gitter gezeichnet, die Kette als Schlange (ein Weg durch das Raster), Zufallsnetz und Faktor auf einem Kreis.")
rows = []
for r in a.rows:
    rows.append({"Netz": r.name, "Kanten": r.edges, "richtig": r.right, "falsch": r.wrong, "fehlend": r.missing, "F1": de(r.f1), "Transitivität": de(r.trans),
                 "erwartet (Kanten)": "–" if r.expected is None else de(r.expected["edges"], 1)})
st.dataframe(rows, hide_index=True, width="stretch")
for verdict in (ev.verdict_main, ev.verdict_threshold, ev.verdict_partial, ev.verdict_glasso):
    kind, text = verdict(a)
    getattr(st, kind)(text)
if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = (settings, build_pdf(a))
pdf = st.session_state.get("pdf_bytes")
if pdf and pdf[0] == settings:
    st.download_button("⬇️ PDF herunterladen", pdf[1], file_name="korrelationsnetz-demo.pdf", mime="application/pdf", key="pdf_download")

if a.partial is not None:
    st.markdown("### Warum sich die Netze unterscheiden: jedes Paar als Punkt")
    st.plotly_chart(build_pairs(a), width="stretch", key="main_pairs")
    st.caption("Rechts der Linie τ liegt das Schwellennetz, über der waagerechten Linie das Teilkorrelationsnetz. Paare ohne direkte Kante mit großem |r| (rechts unten) sind die indirekten Zusammenhänge.")
st.markdown("### Wo schneidet der Graphical Lasso ab? Der Pfad")
st.plotly_chart(build_path(a), width="stretch", key="main_path")
st.caption("Je kleiner λ, desto mehr Kanten. Die Marken zeigen den Wert, den EBIC wählt, und den besten Wert (er kennt die Wahrheit und ist nur eine Obergrenze).")

st.markdown("## ❓ Kernfrage: Welche Kanten sind direkt?")
st.markdown("**Nicht die, die die Schwelle findet:** ein Korrelationsnetz mischt direkte und indirekte Zusammenhänge und gemeinsame Ursachen. Das Teilkorrelationsnetz trennt sie, braucht aber viele Wochen, und der Graphical Lasso "
            "verlagert die Frage nur auf die Wahl von λ (Experimente unten).")

st.markdown("---")

grid = _grid()
if grid is None:
    st.warning("Das vorgerechnete Raster (data/network_grid.json) fehlt; es entsteht mit `python tools/make_grid.py`. Die Experimente zeigen es.")


def _table(cell: dict, gamma_value: float):
    out = []
    for key in ("schwelle", "teil", f"ebic_{gamma_value}", "best", "rank"):
        m = cell["methods"].get(key)
        if m is None:
            continue
        name = METHOD_LABELS["ebic"] + f" (γ = {n_text(gamma_value)})" if key.startswith("ebic") else METHOD_LABELS[key.split("_")[0]]
        exp = cell["expected"].get(key)
        out.append({"Verfahren": name, "Kanten": de(m["edges"], 1), "richtig": de(m["right"], 1), "falsch": de(m["wrong"], 1), "F1": de(m["f1"], 3), "Transitivität": de(m["trans"]),
                    "erwartet (Kanten)": "–" if not exp else de(exp["edges"], 1)})
    return out


# --- E1 Vier Welten ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Vier Welten, drei Netze")
if grid is not None:
    st.caption(f"p = {C.GRID_P} Kennzahlen, n = {C.E1_N} Wochen, {len(C.GRID_SEEDS) * C.E1_RUNS} Datensätze je Welt; Schwelle τ = {n_text(C.DEFAULT_TAU)}, α = {n_text(C.DEFAULT_ALPHA)}. Mittelwerte; F1 gegen die Wahrheit.")
    e1c1, e1c2 = st.columns(2)
    e1_truth = e1c1.selectbox("Welt für Tabelle und Verdikt", list(TRUTH_SHORT), format_func=TRUTH_SHORT.get, key="e1_truth")
    e1_gamma = e1c2.selectbox("EBIC-Gewicht γ", list(C.GAMMAS), index=1, key="e1_gamma", format_func=n_text)
    st.plotly_chart(build_e1(grid["e1"], e1_gamma), width="stretch", key="e1_chart")
    cell = ev.find_cell(grid["e1"], e1_truth, C.E1_N)
    st.dataframe(_table(cell, e1_gamma), hide_index=True, width="stretch")
    kind, text = ev.verdict_e1(grid["e1"], e1_truth)
    getattr(st, kind)(text)
    st.caption("Das Schwellennetz irrt in beide Richtungen: es erfindet Kanten (Faktor, zwei Schritte in der Kette) und verfehlt echte (Gitter). Das Teilkorrelationsnetz ist in der Kette fast fehlerfrei, im Gitter und im Zufallsnetz "
               "knapp an der Nachweisgrenze. Der Graphical Lasso mit EBIC findet die echten Kanten und fügt falsche hinzu; mit dem besten λ ist er so gut wie die Rangfolge nach |r|.")

st.markdown("---")

# --- E2 Datenmenge -------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Gegen die Datenmenge")
if grid is not None:
    st.caption(f"p = {C.GRID_P}, n = {', '.join(str(x) for x in C.E2_NS)} Wochen, {len(C.GRID_SEEDS) * C.E2_RUNS} Datensätze je Zelle. Rauten: die exakte Erwartung der beiden einfachen Netze (Fisher-z).")
    e2c1, e2c2 = st.columns(2)
    e2_truth = e2c1.selectbox("Welt", list(C.E2_TRUTHS), format_func=TRUTH_SHORT.get, key="e2_truth")
    e2_gamma = e2c2.selectbox("EBIC-Gewicht γ", list(C.GAMMAS), index=1, key="e2_gamma", format_func=n_text)
    st.plotly_chart(build_f1_vs_n(grid["e2"], e2_truth, e2_gamma), width="stretch", key="e2_f1")
    st.plotly_chart(build_edges_vs_n(grid["e2"], e2_truth), width="stretch", key="e2_edges")
    kind, text = ev.verdict_e2(grid["e2"], e2_truth)
    getattr(st, kind)(text)
    st.caption("Mehr Wochen machen das Teilkorrelationsnetz mächtig, aber nicht unbedingt das Schwellennetz besser: im Gitter liegen die echten Kanten bei |r| ≈ 0,2 unter der Schwelle, und je genauer r bekannt ist, "
               "desto sicherer bleiben sie darunter. EBIC wählt bei wenig Daten das leere Netz und bei vielen zu viele Kanten.")

st.markdown("---")

# --- E3 Mehr Kennzahlen als Wochen ----------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Mehr Kennzahlen als Wochen")
if grid is not None:
    st.caption(f"p = {C.GRID_P}, n = {', '.join(str(x) for x in C.E3_NS)} Wochen, {len(C.GRID_SEEDS) * C.E3_RUNS} Datensätze je Zelle. Bei n ≤ p + 1 = {C.GRID_P + 1} gibt es kein Teilkorrelationsnetz.")
    e3c1, e3c2 = st.columns(2)
    e3_truth = e3c1.selectbox("Welt", list(C.E2_TRUTHS), format_func=TRUTH_SHORT.get, key="e3_truth")
    e3_gamma = e3c2.selectbox("EBIC-Gewicht γ", list(C.GAMMAS), index=1, key="e3_gamma", format_func=n_text)
    st.plotly_chart(build_f1_vs_n(grid["e3"], e3_truth, e3_gamma), width="stretch", key="e3_f1")
    st.plotly_chart(build_edges_vs_n(grid["e3"], e3_truth), width="stretch", key="e3_edges")
    kind, text = ev.verdict_e3(grid["e3"], e3_truth)
    getattr(st, kind)(text)
    st.caption("Mit so wenigen Wochen verbindet das Schwellennetz fast jede Kennzahl mit jeder. Der Graphical Lasso rechnet auch bei n < p, aber EBIC wählt das leere Netz, und das beste λ liefert nicht mehr als die Rangfolge.")

st.markdown("---")

# --- E4 Pfad ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wo schneidet man ab?")
if grid is not None:
    cells = {(c["truth"], c["n"]): c for c in grid["e2"] + grid["e1"] if "path" in c}                      # Zellen der Vier Welten gehen vor
    labels = {key: f"{TRUTH_SHORT[key[0]]}, n = {key[1]}" for key in cells}
    st.caption("Mittel über die Datensätze einer Zelle: Kanten und F1 entlang des Pfads. Senkrechte Linien: im Mittel gewählter Wert von EBIC (γ = 0, 0,5, 1) und bestes λ (kennt die Wahrheit).")
    e4c1, e4c2 = st.columns(2)
    e4_key = e4c1.selectbox("Welt und Wochen", list(cells), format_func=labels.get, key="e4_cell")
    e4_gamma = e4c2.selectbox("EBIC-Gewicht γ für das Verdikt", list(C.GAMMAS), index=1, key="e4_gamma", format_func=n_text)
    st.plotly_chart(build_e4(cells[e4_key]), width="stretch", key="e4_chart")
    kind, text = ev.verdict_e4(cells[e4_key], e4_gamma)
    getattr(st, kind)(text)
    st.caption("Der Verlauf ist glatt, die Wahl nicht: EBIC bestraft die Kanten mit dem Verbundwert der bestraften Schätzung und ohne Neuanpassung. Das ist ein Befund über diese Regel, nicht über jede Regel (Kreuzvalidierung und Neuanpassung sind nicht gebaut).")

st.markdown("---")

# --- E5 Verborgener Faktor -------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Der verborgene Faktor")
if grid is not None:
    pc0 = float(K.true_partial_corr("faktor", C.GRID_P)[0, 1])
    st.caption(f"Faktor-Welt, p = {C.GRID_P}: alle Paare haben r = {n_text(C.FACTOR_RHO)}, die wahre Teilkorrelation jedes Paares ist {de(pc0, 4)} (der Faktor ist nicht mit gemessen). "
               f"{len(C.GRID_SEEDS) * C.E5_RUNS} Datensätze je Zelle.")
    st.plotly_chart(build_e5(grid["e5"]), width="stretch", key="e5_chart")
    kind, text = ev.verdict_e5(grid["e5"])
    getattr(st, kind)(text)
    st.caption("„Keine Kante“ war hier eine Frage der Macht, nicht der Wahrheit: bedingte Unabhängigkeit gilt nur, wenn alles Wichtige mit gemessen wurde.")

st.markdown("---")

# --- Grenzen ------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    r"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Gemeinsam normalverteilte, lineare Zusammenhänge** | Teilkorrelation und Graphical Lasso sehen nur lineare, gaußsche Abhängigkeit; eine nichtlineare Kante ohne lineare Teilkorrelation bleibt unsichtbar | nicht gebaut |
| **Alles Wichtige ist gemessen** | eine verborgene Ursache macht jedes Paar abhängig; kein Netz hier (Schwelle, Teilkorrelation, Graphical Lasso) trennt das (Experiment 5) | – |
| **Ursachen und Richtungen** | ein Netz hier ist ungerichtet und sagt nichts über Ursachen; der PC-Algorithmus und Pfeilrichtungen gehören der Kausal-Linie | `strukturlernen-demo` |
| **λ wird aus den Daten gewählt** | EBIC ohne Neuanpassung wählt bei wenig Daten das leere Netz und bei vielen zu viele Kanten; Kreuzvalidierung und Neuanpassung sind nicht gebaut | – |
| **Rangfolge und bestes λ kennen die Wahrheit** | beide sind nur Messlatten (Obergrenzen), kein Verfahren für echte Daten | – |
| **Vier Welten sind Beispiele** | Kette, Gitter, Zufallsnetz und Faktor sind Modellwahlen; andere Dichten und Stärken verschieben die Zahlen | – |
| **Ein Test je Paar mit Bonferroni** | streng und für viele Paare blind; weniger strenge Verfahren (Falscherkennungsrate) sind nicht gebaut | – |
"""
)
st.caption("Alle Daten sind simuliert; die Welten sind eine Modellwahl, kein Befund über echte Kennzahlen.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Die $p$ Kennzahlen sind gemeinsam normalverteilt mit Kovarianz $\Sigma$ (Diagonale 1). Die **Präzisionsmatrix** ist $P = \Sigma^{-1}$; ihre Nullen sind die Kanten der Wahrheit.

| Symbol | Bedeutung |
|---|---|
| $\hat R$ | Stichprobenkorrelationsmatrix aus $n$ Wochen |
| $\rho_{ij\cdot\text{rest}}$ | Teilkorrelation zweier Kennzahlen gegeben alle übrigen |
| $m = p(p-1)/2$ | Zahl der Paare |
| $\tau, \alpha, \lambda$ | Schwelle, Irrtumswahrscheinlichkeit, Strafe des Graphical Lasso |

**Teilkorrelation** (Stichprobe: $P$ durch $\hat R^{-1}$ ersetzen):

$$ \rho_{ij\cdot\text{rest}} = \frac{-P_{ij}}{\sqrt{P_{ii}\,P_{jj}}} $$

**Fisher-z-Test** der Teilkorrelation mit $n-p-1$ Freiheitsgraden, Bonferroni über alle $m$ Paare ($\Phi$: Normalverteilung):

$$ \sqrt{n-p-1}\;\operatorname{artanh}\hat\rho_{ij\cdot\text{rest}} \;>\; \Phi^{-1}\!\Big(1-\frac{\alpha}{2m}\Big) $$

**Erwartete Kanten des Schwellennetzes:** $\operatorname{artanh}\hat r_{ij}$ ist näherungsweise normal mit Mittel $\operatorname{artanh}\rho_{ij}$ und Varianz $1/(n-3)$; jedes Paar ist Kante mit der Wahrscheinlichkeit

$$ \Pr\big(|\operatorname{artanh}\hat r_{ij}| > \operatorname{artanh}\tau\big) = 1 - \Phi\!\big(\sqrt{n-3}\,(\operatorname{artanh}\tau - \operatorname{artanh}\rho_{ij})\big) + \Phi\!\big(-\sqrt{n-3}\,(\operatorname{artanh}\tau + \operatorname{artanh}\rho_{ij})\big) $$

**Erwartete Kanten des Teilkorrelationsnetzes:** ebenso mit der wahren Teilkorrelation, $\sqrt{n-p-1}$ statt $\sqrt{n-3}$ und der Bonferroni-Schwelle statt $\operatorname{artanh}\tau$.

**Graphical Lasso** (Diagonale nicht bestraft):

$$ \hat\Theta = \arg\min_{\Theta \succ 0}\; -\log\det\Theta + \operatorname{tr}(\hat R\,\Theta) + \lambda \sum_{i\neq j}|\Theta_{ij}| $$

**EBIC** mit $E$ Kanten und dem Gewicht $\gamma$:

$$ \mathrm{EBIC}_\gamma = -n\big(\log\det\hat\Theta - \operatorname{tr}(\hat R\,\hat\Theta)\big) + E\ln n + 4\gamma E\ln p $$

**Transitivität** ($A$ die Adjazenzmatrix, $d_i$ die Grade):

$$ C = \frac{3\cdot\text{Dreiecke}}{\text{verbundene Dreier}} = \frac{\operatorname{tr}(A^3)}{\sum_i d_i(d_i-1)} $$

**Güte** gegen die Wahrheit mit $r$ richtigen, $w$ falschen und $f$ fehlenden Kanten:

$$ F_1 = \frac{2r}{2r + w + f} $$

**Quellen.** In dieser Fassung sind keine Literaturangaben ausgewiesen; die Formeln sind durch unabhängige Rechenwege (Tests gegen scipy, sklearn, networkx, Optimalitätsbedingungen und Simulationen) und die Messungen der Experimente gestützt.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
