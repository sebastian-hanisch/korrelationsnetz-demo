# Korrelationsnetz und Teilkorrelation: welche Kanten sind direkt? – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-korrelationsnetz-demo.streamlit.app/)** -->

Sechstes Stück der Korrelations-Reihe in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, Nachfolger von [pearson-demo](https://github.com/sebastian-hanisch/pearson-demo) (Kante `fix`: ein Netz aus Korrelationen ist eine Aussage über direkte Zusammenhänge, und die muss man erst von indirekten und gemeinsamen Ursachen trennen), neben [rangkorrelation-demo](https://github.com/sebastian-hanisch/rangkorrelation-demo), [abhaengigkeitsmasse-demo](https://github.com/sebastian-hanisch/abhaengigkeitsmasse-demo), [scheinkorrelation-demo](https://github.com/sebastian-hanisch/scheinkorrelation-demo), [korrelationsmatrix-demo](https://github.com/sebastian-hanisch/korrelationsmatrix-demo), [randabhaengigkeit-demo](https://github.com/sebastian-hanisch/randabhaengigkeit-demo), [korrelierte-merkmale-demo](https://github.com/sebastian-hanisch/korrelierte-merkmale-demo) und [aggregationsebene-demo](https://github.com/sebastian-hanisch/aggregationsebene-demo). Es ist die **erste Demo des Portfolios, in der ein Netz aus Daten gebaut wird**: **p Kennzahlen** über **n Wochen** aus einer Welt, in der bekannt ist, welche Kennzahlen direkt zusammenhängen (Kette, Gitter, Zufallsnetz) oder gar keine (ein verborgener Faktor). Daraus entstehen drei Netze: eines aus allen Paaren mit großem \|r\| (**Schwelle**), eines aus den **Teilkorrelationen** (gegeben alle übrigen) und eines vom **Graphical Lasso**. Die Fragen: **Was zeigt ein Korrelationsnetz? Wann sind seine Kanten direkt? Wie viele Wochen braucht das Teilkorrelationsnetz? Und wo schneidet man ab?**

## Warum dieses Problem

Ein Netz aus Korrelationen sieht aus wie eine Landkarte der Zusammenhänge und ist keine: es mischt direkte und indirekte Zusammenhänge und zeigt gemeinsame Ursachen als Kanten. Befunde (p = 40 Kennzahlen): eine Welt **ohne jede direkte Kante** liefert bei n = 500 ein Schwellennetz mit rund **730 von 780** möglichen Kanten (Mittel der 20 Datensätze **737,2**, Streuung je Datensatz rund 40 Kanten; erwartet 728,1) und Transitivität rund **0,95**, das Teilkorrelationsnetz hat 0,2 Kanten; dieselbe Schwelle findet in einem Gitter mit 67 echten Kanten nur **8,9** und wird ab n = 100 mit mehr Daten **schlechter** (F1 rund 0,4 bei n = 100, rund 0,03 bei n = 2 000: 0,414 und 0,022 im Raster); das Teilkorrelationsnetz hat in der Kette bei n = 100 nur **9,4 von 39** echten Kanten (erwartet 9,2); und mit EBIC (ohne Neuanpassung) ist der Graphical Lasso kein Ausweg: die Wahl von λ ist hier die eigentliche Schwierigkeit: mit dem besten λ liegt er in allen 31 Zellen mit Graphical Lasso höchstens 0,028 F1 von der einfachen Rangfolge nach \|r\| entfernt (beide kennen die Wahrheit), mit EBIC wählt er bei wenig Daten das leere Netz und bei n = 2 000 in der Kette 27,7 falsche Kanten. Die Kantenzahl des Schwellen- und des Teilkorrelationsnetzes lässt sich vorher ausrechnen (Fisher-z), und die Demo zeigt die Erwartung neben jeder Messung.

## Befunde und Korrekturen gegenüber dem Plan

Der Einzelplan (`korrelation-planung/STUECK-6-korrelationsnetz.md`, nicht in diesem Repo) stützte sich auf Wegwerfskripte mit gleichem Erzeuger und eigenem Löser; die gebauten Zahlen stimmen mit der Vormessung überein, mit diesen Ausnahmen:

- **Die Nabe fliegt raus.** Die vierte Welt der ersten Messung (Sterne um Knotenpunkte) verhielt sich in der Vormessung wie die Kette und brachte keinen eigenen Befund; gebaut sind Kette, Gitter, Zufallsnetz und Faktor.
- **Faktor-Welt:** der PLAN (aus der ersten Vormessung) nannte 652 Kanten und Transitivität 0,86 aus **einem** Datensatz. Im Mittel über 20 Datensätze sind es **737,2** Kanten (erwartet 728,1; ein Lauf mit 300 Datensätzen gab 731,6) und Transitivität **0,95**: alle Paare hängen am selben Stichprobenwert des Faktors, die Kantenzahl schwankt von Datensatz zu Datensatz stark, ein einzelner Datensatz ist kein Befund.
- **Graphical Lasso ist kein Gewinner.** Der PLAN erwartete, dass die regularisierte Schätzung das Netz am besten trifft. Gemessen: mit dem besten λ des Pfads (er kennt die Wahrheit) liegt sie in allen 31 Zellen höchstens 0,028 F1 von der Rangfolge nach \|r\| mit bekannter Kantenzahl entfernt; mit EBIC (γ = 0,5, ohne Neuanpassung) ist sie bei wenig Daten leer und bei vielen zu dicht. Der Vorteil der Dünnbesetzung bleibt als Idee, der Vorteil im Ergebnis nicht.
- **Neu:** die **Erwartung (Fisher-z-Näherung)** der Kantenzahl für Schwellen- und Teilkorrelationsnetz aus der Fisher-z-Näherung (Tabelle unten), das Gewicht γ des EBIC als Regler und Experiment (**kein γ gewinnt überall**), die Transitivität als Fehlersignal.
- **Eigener Löser gegen `sklearn`:** `sklearn.covariance.graphical_lasso` bestraft die **Diagonale nicht**; die erste Fassung des eigenen Lösers tat es und wich um 0,2 bis 0,37 ab. Außerdem meldet `sklearn` bei sehr enger Toleranz „did not converge“: das Orakel vergleicht die **Zielfunktion** und die Einträge auf 5·10⁻³.

## Modell

Standardisierte, gemeinsam normalverteilte Kennzahlen; die Wahrheit steht in der **Präzisionsmatrix** P = Σ⁻¹ (die Teilkorrelation zweier Kennzahlen gegeben alle übrigen ist −P_ij / √(P_ii P_jj); wo P null ist, gibt es keine Kante). Vier Welten (`knz_algorithm.py`): **Kette** (Σ_ij = 0,5^\|i−j\|, P tridiagonal, Teilkorrelation der Nachbarn 0,4 im Inneren), **Gitter** (5 Zeilen, p/5 Spalten, P = I − 0,2·A mit der Nachbarschaft A, Teilkorrelation 0,2; bei p = 40 67 Kanten), **Zufallsnetz** (fester Zufallsgraph mit Seed 7 und Kantenwahrscheinlichkeit 0,06 je Paar; bei p = 40 43 Kanten mit beiden Vorzeichen und Teilkorrelationen im Betrag von 0,18 bis 0,38) und **Faktor** (x_i = 0,6 f + 0,8 e_i, alle Paare haben r = 0,36 und **keine** hat eine direkte Kante, die wahre Teilkorrelation jedes Paares ist 0,0245, weil der Faktor nicht mit gemessen wird). Zufall aus SplitMix64 in wörtlichen Kopien von `pearson-demo` (`prs_*.py`).

## Methodik

- **Schwellennetz:** Kante, wenn \|r\| > τ (strikt; τ = 0,30 im Raster). **Teilkorrelationsnetz:** Teilkorrelation aus der Inversen der Stichprobenmatrix, Fisher-z mit n − p − 1 Freiheitsgraden, Bonferroni über alle p(p−1)/2 Paare (α = 0,05 im Raster); bei n ≤ p + 1 gibt es die Inverse nicht, das Netz entfällt.
- **Graphical Lasso** (`knz_algorithm.py`, eigener Blockkoordinatenabstieg nach Friedman, Hastie und Tibshirani, Diagonale nicht bestraft): ein Pfad aus 15 Werten von λ_max (größtes \|r\|, leeres Netz) bis 0,05 λ_max, jeweils mit den Werten des vorigen angewärmt. **EBIC** mit dem Verbundwert der bestraften Schätzung und ohne Neuanpassung (γ = 0, 0,5, 1); **bestes λ** = der Pfadwert mit dem höchsten F1 gegen die Wahrheit (kennt die Wahrheit, nur eine Obergrenze).
- **Rangfolge:** die k Paare mit dem größten \|r\|, k = Zahl der echten Kanten (kennt die Wahrheit, nur eine Messlatte). **F1** = 2 · richtig / (2 · richtig + falsch + fehlend); ohne echte und gefundene Kanten gilt F1 = 1 (richtig leer). **Transitivität** = 3 · Dreiecke / verbundene Dreier.
- **Erwartung (Fisher-z-Näherung):** artanh r̂ ist näherungsweise normal mit Mittel artanh ρ und Varianz 1/(n − 3); das Schwellennetz hat in jedem Paar die Wahrscheinlichkeit P(\|artanh r̂\| > artanh τ). Für das Teilkorrelationsnetz gilt dasselbe mit der wahren Teilkorrelation, Varianz 1/(n − p − 1) und der Bonferroni-Schwelle.
- **Experimente:** (1) vier Welten, drei Netze (n = 500), (2) gegen die Datenmenge (n = 50 bis 2 000), (3) mehr Kennzahlen als Wochen (n = 20 bis 60), (4) der λ-Pfad und die Wahl des Abschnitts, (5) der verborgene Faktor (n bis 50 000).
- **Vorgerechnetes Raster** (`tools/make_grid.py`, `data/network_grid.json`, 261 Sekunden auf 10 Prozessen bei der Erzeugung, rechnerabhängig): E1 4 Welten · 20 Datensätze, E2 3 Welten · 5 Datenmengen · 16 Datensätze, E3 3 Welten · 4 Datenmengen · 16 Datensätze (je mit Pfad), E5 Faktor-Welt · 4 Datenmengen · 100 Datensätze (ohne Graphical Lasso); p = 40, feste Seeds.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py` (Raster-Werte gegen die JSON-Datei, die Tabellen Zeile für Zeile in der Rundung der App, die Hilfetexte der Presets über dieselben Funktionen wie in der App). Mittelwerte über alle Datensätze einer Zelle; p = 40.

**Vier Welten, drei Netze** (Experiment 1; n = 500, τ = 0,30, α = 0,05, EBIC γ = 0,5; Welt (echte Kanten; Transitivität der Wahrheit); Zellen: Kanten / davon falsch / F1 (/ Transitivität); „bestes λ“ und „Rangfolge“ kennen die Wahrheit):

| Welt | Schwellennetz | Teilkorrelationsnetz | Graphical Lasso, EBIC | bestes λ (F1) | Rangfolge (F1) |
|---|---|---|---|---|---|
| **Kette** (39; 0,00) | 42,2 / 3,2 / 0,960 / 0,18 | 39,0 / 0,1 / 0,999 | 58,7 / 19,7 / 0,804 | 1,000 | 1,000 |
| **Gitter** (67; 0,00) | 8,9 / 0,0 / 0,234 / 0,00 | 42,7 / 0,0 / 0,777 | 86,3 / 19,4 / 0,875 | 0,954 | 0,936 |
| **Zufallsnetz** (43; 0,04) | 24,4 / 0,0 / 0,724 / 0,05 | 38,4 / 0,1 / 0,941 | 51,8 / 8,8 / 0,909 | 0,986 | 0,969 |
| **Faktor** (0; 0,00) | 737,2 / 737,2 / 0,000 / 0,95 | 0,2 / 0,2 / 0,800 | 492,8 / 492,8 / 0,000 | 1,000 | 1,000 |

**Die Kantenzahl ist vorher bekannt** (erwartet nach der Fisher-z-Näherung, gegen den Mittelwert aus 16 bis 100 Datensätzen, der um bis zu etwa eine Kante abweichen kann; Schwellennetz: Kanten; Teilkorrelationsnetz: echte Kanten / alle Kanten):

| Welt, Wochen | Schwellennetz erwartet | gemessen | Teilkorrelation erwartet (echt / alle) | gemessen (echt / alle) |
|---|---|---|---|---|
| Kette, n = 100 | 53,01 | 53,0 | 9,23 / 9,27 | 9,4 / 9,5 |
| Kette, n = 200 | 47,69 | 47,8 | 35,66 / 35,70 | 35,8 / 35,9 |
| Kette, n = 500 | 43,33 | 42,3 | 39,00 / 39,05 | 39,0 / 39,1 |
| Kette, n = 2000 | 39,30 | 39,4 | 39,00 / 39,05 | 39,0 / 39,1 |
| Gitter, n = 500 | 8,16 | 8,4 | 42,57 / 42,62 | 42,7 / 42,7 |
| Gitter, n = 2000 | 1,10 | 0,8 | 67,00 / 67,05 | 67,0 / 67,0 |
| Zufallsnetz, n = 200 | 24,76 | 24,2 | 14,83 / 14,88 | 13,8 / 13,8 |
| Zufallsnetz, n = 500 | 24,87 | 24,3 | 37,60 / 37,65 | 37,9 / 38,0 |
| Faktor, n = 500 | 728,08 | 730,0 | 0,00 / 0,20 | 0,0 / 0,2 |
| Faktor, n = 5000 | 780,00 | 780,0 | 0,00 / 9,05 | 0,0 / 9,3 |
| Faktor, n = 50000 | 780,00 | 780,0 | 0,00 / 726,38 | 0,0 / 727,0 |

**Kette gegen die Datenmenge** (Experiment 2; F1; beim Schwellennetz in Klammern die Zahl der Kanten bei 39 echten):

| Verfahren | n = 50 | 100 | 200 | 500 | 2 000 |
|---|---|---|---|---|---|
| Schwellennetz (Kanten) | 0,635 (79,6) | 0,840 (53,0) | 0,899 (47,8) | 0,960 (42,3) | 0,995 (39,4) |
| Teilkorrelationsnetz | 0,009 | 0,385 | 0,955 | 0,999 | 0,999 |
| Graphical Lasso, EBIC (γ = 0,5) | 0,000 | 0,859 | 0,812 | 0,800 | 0,740 |
| Graphical Lasso, bestes λ | 0,812 | 0,928 | 0,988 | 1,000 | 1,000 |
| Rangfolge nach \|r\| | 0,803 | 0,921 | 0,981 | 1,000 | 1,000 |

**Gitter gegen die Datenmenge** (F1; 67 echte Kanten):

| Verfahren | n = 50 | 100 | 200 | 500 | 2 000 |
|---|---|---|---|---|---|
| Schwellennetz (Kanten) | 0,391 (52,4) | 0,414 (22,9) | 0,371 (16,0) | 0,222 (8,4) | 0,022 (0,8) |
| Teilkorrelationsnetz | 0,000 | 0,011 | 0,124 | 0,777 | 1,000 |
| Graphical Lasso, EBIC (γ = 0,5) | 0,000 | 0,000 | 0,002 | 0,879 | 0,850 |
| Graphical Lasso, bestes λ | 0,405 | 0,601 | 0,802 | 0,956 | 1,000 |
| Rangfolge nach \|r\| | 0,391 | 0,595 | 0,783 | 0,939 | 1,000 |

**Mehr Kennzahlen als Wochen** (Experiment 3; p = 40, also n ≤ 41 ohne Teilkorrelationsnetz; Schwellennetz: Kanten (F1); die übrigen F1; Teilkorrelationsnetz: Kanten):

| Welt, Wochen | Schwellennetz | Rangfolge | bestes λ | EBIC (γ = 0,5) | Teilkorrelationsnetz |
|---|---|---|---|---|---|
| Kette, n = 20 | 199,8 (0,279) | 0,494 | 0,502 | 0,000 | – |
| Kette, n = 30 | 134,2 (0,409) | 0,660 | 0,667 | 0,000 | – |
| Kette, n = 45 | 85,4 (0,597) | 0,766 | 0,779 | 0,000 | 0,0 |
| Kette, n = 60 | 66,8 (0,717) | 0,846 | 0,862 | 0,000 | 0,7 |
| Gitter, n = 20 | 177,9 (0,233) | 0,230 | 0,256 | 0,000 | – |
| Gitter, n = 30 | 109,1 (0,298) | 0,282 | 0,309 | 0,000 | – |
| Gitter, n = 45 | 61,6 (0,382) | 0,389 | 0,400 | 0,000 | 0,0 |
| Gitter, n = 60 | 41,1 (0,414) | 0,451 | 0,453 | 0,000 | 0,0 |
| Zufallsnetz, n = 20 | 171,8 (0,208) | 0,273 | 0,268 | 0,000 | – |
| Zufallsnetz, n = 30 | 104,4 (0,326) | 0,369 | 0,370 | 0,000 | – |
| Zufallsnetz, n = 45 | 62,4 (0,437) | 0,458 | 0,460 | 0,000 | 0,0 |
| Zufallsnetz, n = 60 | 40,2 (0,542) | 0,545 | 0,555 | 0,000 | 0,1 |

**Wo schneidet man ab?** (Experiment 4; Graphical Lasso mit EBIC für γ = 0, 0,5, 1 und mit dem besten λ; Zellen: Kanten / davon falsch / F1):

| Welt, Wochen | γ = 0 | γ = 0,5 | γ = 1 | bestes λ |
|---|---|---|---|---|
| Kette, n = 500 | 83,6 / 44,6 / 0,640 | 58,7 / 19,7 / 0,804 | 54,2 / 15,2 / 0,839 | 39,0 / 0,0 / 1,000 |
| Gitter, n = 500 | 107,2 / 40,2 / 0,772 | 86,3 / 19,4 / 0,875 | 15,9 / 2,5 / 0,182 | 66,1 / 2,6 / 0,954 |
| Zufallsnetz, n = 500 | 68,3 / 25,3 / 0,775 | 51,8 / 8,8 / 0,909 | 50,2 / 7,2 / 0,924 | 43,7 / 0,9 / 0,986 |
| Kette, n = 100 | 85,2 / 46,2 / 0,637 | 51,4 / 12,7 / 0,859 | 0,0 / 0,0 / 0,000 | 40,8 / 3,8 / 0,928 |
| Kette, n = 2000 | 85,5 / 46,5 / 0,631 | 66,7 / 27,7 / 0,740 | 60,4 / 21,4 / 0,786 | 39,0 / 0,0 / 1,000 |

**Der verborgene Faktor** (Experiment 5; keine direkte Kante, 780 Paare; Kanten (Transitivität)):

| Wochen n | Schwellennetz | Teilkorrelationsnetz |
|---|---|---|
| 200 | 643,8 (0,858) | 0,1 (0,000) |
| 500 | 730,0 (0,945) | 0,2 (0,000) |
| 5000 | 780,0 (1,000) | 9,3 (0,002) |
| 50000 | 780,0 (1,000) | 727,0 (0,931) |

Befunde in Worten:

- **Das Schwellennetz irrt in beide Richtungen.** Es erfindet Kanten (Faktor-Welt 737,2 von 780; in der Kette bei n = 500 im Mittel rund 4 falsche Kanten (erwartet 4,3; die Tabelle zeigt 3,2 aus 20 Datensätzen) und Transitivität rund 0,2 statt 0, weil Paare zwei Schritte entfernt mit wahrem r = 0,25 die Schwelle 0,30 zufällig überschreiten; bei n = 50 sind es 79,6, bei n = 20 199,8 Kanten statt 39) und verfehlt echte (Gitter: nur 8,9 von 67, F1 0,234, weil die wahre Korrelation direkter Nachbarn nur 0,23 bis 0,27 beträgt, die Teilkorrelation 0,2). **Eine feste Schwelle ist keine Eigenschaft der Welt.**
- **Mehr Daten machen das Schwellennetz im Gitter ab n = 100 schlechter:** F1 0,414 bei n = 100, 0,022 bei n = 2 000 (0,8 Kanten). Bei wenig Daten bringt das Rauschen die Paare über 0,3, bei vielen sitzen die echten Kanten genau darunter.
- **Das Teilkorrelationsnetz ist bei n ≤ 50 blind, bei n = 100 findet es nur etwa ein Viertel der Kanten** (Kette n = 100: 9,4 von 39 echten, 9,5 Kanten insgesamt, erwartet 9,27, obwohl die Teilkorrelation 0,4 beträgt, an den beiden Randkanten 0,447) und gewinnt bei vielen alles (Gitter n = 2 000: 67 von 67). Die Macht des Tests, nicht die Struktur, begrenzt, was man sieht.
- **EBIC wählt bei wenig Daten das leere Netz** (Kette n = 50: 0,0 Kanten) **und bei vielen Daten zu viele Kanten** (falsche Kanten in der Kette: 19,7 bei n = 500, 27,7 bei n = 2 000; im Gitter bei n = 500 19,4). Mit γ = 1 ist EBIC im Gitter bei n = 500 zu streng (15,9 Kanten, F1 0,182): **kein γ gewinnt überall.** Das ist ein Befund über diese Regel (Verbundwert der bestraften Schätzung, ohne Neuanpassung), nicht über jede Regel.
- **Das beste λ und die Rangfolge liegen dicht beieinander** (höchstens 0,028 F1 Abstand in 31 Zellen): in diesen Welten bringt der Graphical Lasso bei bekannter Kantenzahl nur einen kleinen Vorsprung vor dem Sortieren nach \|r\| (im Mittel 0,009 F1, in 30 von 31 Zellen nicht schlechter; das beste λ ist das Maximum über 15 Pfadwerte und kennt die Wahrheit); die Schwierigkeit ist der Abschnitt, nicht der Schätzer.
- **Bei n < p erreicht in diesen Welten keines der geprüften Verfahren ein F1 über 0,67:** in der Kette mit n = 20 erreicht die Rangfolge F1 0,494 und das beste λ 0,502 (Schwellennetz 0,279); EBIC wählt das leere Netz.
- **Der verborgene Faktor:** die Teilkorrelation jedes Paares ist nicht null (0,0245). Das Teilkorrelationsnetz hat bei n = 500 0,2 Kanten, bei n = 5 000 9,3 und bei n = 50 000 **727,0 von 780** (Transitivität 0,931), obwohl keine direkte Kante existiert. „Keine Kante“ war eine Frage der Macht, nicht der Wahrheit. Der Graphical Lasso mit EBIC zeichnet bei n = 500 492,8 Kanten.

## Ehrliche Grenzen

- **Gemeinsam normalverteilte, lineare Zusammenhänge.** Teilkorrelation und Graphical Lasso sehen nur lineare, gaußsche Abhängigkeit; eine nichtlineare Kante ohne lineare Teilkorrelation bleibt unsichtbar.
- **Alles Wichtige ist gemessen.** Eine verborgene Ursache macht jedes Paar abhängig; keines der drei Netze trennt das (Experiment 1 und 5).
- **Keine Ursachen und keine Richtungen.** Die Netze sind ungerichtet; der PC-Algorithmus und Pfeilrichtungen gehören der Linie *Kausale Diagramme und Adjustierung* (`strukturlernen-demo`).
- **λ wird aus den Daten gewählt, aber nur eine Regel ist gebaut.** EBIC ohne Neuanpassung ist ein Befund über diese Regel; Kreuzvalidierung und Neuanpassung fehlen. Die Rangfolge und das beste λ kennen die Wahrheit und sind Messlatten, kein Verfahren für echte Daten.
- **Vier Welten sind Beispiele.** Andere Dichten und Stärken verschieben die Zahlen; die Aussagen gelten für diese Welten, p = 40 und die genannten Schwellen.
- **Bonferroni ist streng.** Weniger strenge Verfahren (Falscherkennungsrate) sind nicht gebaut; sie fänden im Teilkorrelationsnetz bei wenig Daten mehr echte Kanten.
- **Erwartet heißt Fisher-z-Näherung.** Bei sehr kleinem n ist sie ungenau (Kette n = 20: erwartet 193,5 Kanten gegen 199,8 gemessen).
- **Quellen:** keine Literaturangaben; die Formeln sind durch unabhängige Rechenwege (unten) und die Messungen gestützt.

## Verifikation

- **Orakel** (`tests/test_oracle_korrelationsnetz.py`): die Teilkorrelation gegen die **Korrelation von Regressionsresiduen** (`numpy.linalg.lstsq`); die Fisher-z-Entscheidung Paar für Paar gegen **`scipy.stats.norm`**; der Graphical Lasso gegen die **Optimalitätsbedingungen** (ohne Bibliothek) und gegen **`sklearn.covariance.graphical_lasso`** (Zielfunktion und Einträge); die Transitivität gegen **`networkx.transitivity`**; EBIC gegen eine Schleifenfassung; die Erwartungen (Fisher-z-Näherung) gegen **scipy Paar für Paar** und gegen **Simulationen mit 100 bis 300 Datensätzen**.
- **Kern** (`tests/test_algorithm.py`, `tests/test_edge_cases.py`): Handrechnungen (Mini-Instanzen in `conftest.py`: drei Kennzahlen A–B–C mit Teilkorrelation 0, drei Kennzahlen mit Faktor 0,2647, zwei Kennzahlen mit r = 0,6 und λ = 0,2 gibt W₁₂ = 0,4), **inklusive Grenzen** (\|r\| = τ ist keine Kante, n − p − 1 = 0 gibt kein Netz, Kantenschwelle der Präzisionsmatrix), kleinste Welt, n < p, λ ≥ λ_max.
- **Eigenschaften** (`tests/test_eigenschaften_korrelationsnetz.py`): Symmetrie und keine Schleifen, Verschachtelung der Schwellen- und Rangnetze, Teilmenge bei kleinerem α, Verschieben und Strecken der Spalten, Umordnung der Kennzahlen, Monotonie der Erwartungen.
- **Raster und Aussagen** (`tests/test_grid.py`, `tests/test_claims.py`): Probe des Erzeugers gegen die gespeicherte (Zellen mit Graphical Lasso mit grober Toleranz, weil die Wahl am Rand je Plattform kippen kann), jede Zahl dieser README gegen das Raster oder die Live-Funktion, die Tabellen Zeile für Zeile in der Rundung der App.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an seinen Grenzen), alle Presets, Adresszeile, jedes Auswahlfeld der Experimente, Layout, Footer, Laufzeit-Importe.
- **Mutationsprüfung:** `tools/mutate_demo.py` des Website-Repos mutiert den Kern `knz_algorithm` (die wörtlichen Kopien `prs_*` sind ausgenommen, sie werden in `pearson-demo` geprüft; Auswertung, Verdikte und Abbildungen prüfen die übrigen Tests): 129 Stellen; 117 durch eigene Tests gefangen, 4 durch wortgleiche Zeilen in Nachbar-Demos gesichert, 5 als gleichwertig festgehalten und 3 ohne Urteil (der Mutant lässt die Testsuite in die Zeitüberschreitung laufen). Gleichwertig: `<` gegen `<=` bei stetigen Zufallszahlen und `>` gegen `>=` beim Bonferroni-Vergleich (Gleichheit wird nie erreicht), `triples > 0` gegen `> 1` (die Summe der Grade mal Grad − 1 ist 0 oder mindestens 2) und das Vorzeichen in der Skala der Konvergenzgrenze (sie ändert nur, wann der Löser aufhört). Der erste Lauf fand 8 Überlebende; drei Lücken schlossen neue Tests: die Zahl der Durchläufe des Lösers, die Kantenschwelle der Präzisionsmatrix und die drei Zweige der Optimalitätsprüfung.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `knz_algorithm.py` | Kern: Welten, Teilkorrelation, Netze, Transitivität, Güte, Graphical Lasso, EBIC, Erwartungen (Fisher-z-Näherung) |
| `knz_constants.py` | Regler-Grenzen, Welten, Raster der Messreihen, Farben, Presets |
| `knz_evaluation.py` | Einzelwelt mit ihren Zeilen, Zellen des Rasters, Verdikte |
| `knz_layout.py` | Fester Kopf (Stylesheet) |
| `knz_pdf_export.py` | PDF-Export der Hauptansicht |
| `knz_presets.py` | Permalink, Presets (Schnellstart mit Hervorhebung), Zufalls-Seed-Button |
| `knz_visualization.py` | Plotly-Abbildungen (Achsen gesperrt, Kanten in Farbe und Linienart) |
| `app.py` | Streamlit-App |
| `data/network_grid.json` | vorgerechnetes Raster (erzeugt von `tools/make_grid.py`) |
| `prs_*.py` | wörtliche Kopien aus `pearson-demo` (SplitMix64, Szenario, Pearson-Kern, Konstanten) |
| `requirements-dev.txt` | Laufzeit plus pytest, scipy, scikit-learn und networkx für die Orakel |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `tests/` | Orakel, Kern, Eigenschaften, Raster, Aussagen, Verdikte, Oberfläche, Layout, Footer, Laufzeit-Importe, Kopie-Test, Mutationszustand |
| `tools/make_grid.py` | rechnet das Raster vor |

## Bewusst nicht umgesetzt

- **Ursachen und Richtungen aus Daten** (PC-Algorithmus, Markov-Äquivalenzklassen): Linie *Kausale Diagramme und Adjustierung*, `strukturlernen-demo`.
- **Nachbarschaftsauswahl** nach Meinshausen und Bühlmann, **Neuanpassung** nach der Auswahl, **Kreuzvalidierung** von λ und **gemeinsamer Graphical Lasso** für mehrere Gruppen.
- **Weniger strenge Mehrfachtest-Verfahren** (Falscherkennungsrate) für das Teilkorrelationsnetz.
- **Nichtlineare oder Rang-Teilkorrelation** und **zeitliche Netze**.
- **Faktormodelle als Schätzer** (Zerlegung in Faktor plus dünnbesetzten Rest): der verborgene Faktor ist hier nur die Welt, die alle drei Netze täuscht.
- **Die Matrix selbst** (Spektrum, Schrumpfung): `korrelationsmatrix-demo`.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Das Raster entsteht neu mit `python tools/make_grid.py` (bei der letzten Erzeugung 261 Sekunden auf zehn Prozessen; je nach Rechner anders); `tests/test_grid.py` rechnet eine feste Probe nach und schlägt an, wenn Erzeuger oder Statistik das Raster überholt haben.

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
