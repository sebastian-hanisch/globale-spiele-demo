# Unsicherheit und Gleichgewicht: globale Spiele – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-globale-spiele-demo.streamlit.app/)** -->

Viertes Stück der Reihe **Koordinationsspiele** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel: **Sammelverkehr über den Hub.** Betriebe legen ihre Sendungen auf einen gemeinsamen Zubringer (mitmachen) oder fahren direkt. Der Zubringer trägt sich, wenn mindestens der Anteil **θ** der Betriebe mitmacht; θ kennt niemand genau, jeder Betrieb hat nur ein **privates, verrauschtes Signal**. Die Demo **zählt** die Gleichgewichte in Schwellenstrategien („mitmachen, wenn das Signal höchstens x* ist“), zeigt, **wo die Eindeutigkeit endet**, was **endlich viele Betriebe** ändern, wie lange die **Beste-Antwort-Iteration** braucht und, im E-Mail-Spiel, warum „jeder weiß es“ weniger ist als gemeinsames Wissen.

Abgrenzung: `koordinationsspiel-demo` (erstes Stück) zeigt die Gleichgewichte und die Schwelle p* der Hirschjagd; hier ist die Auszahlungsschwelle c dieses p*. `gleichgewichtsauswahl-demo` (zweites Stück) zeigt, welche Konvention auf Dauer gewählt wird, `konventionswechsel-demo` (drittes Stück), wie lange ein Wechsel dauert. `komplexe-ansteckung-demo` rechnet Schwellenregeln auf Netzen ohne Auszahlungen und ohne Signale, `korreliert-demo` Koordination über ein **öffentliches** Vermittlersignal (hier sind die Signale **privat**), `myerson-satterthwaite-demo` Bayes-Nash-Gleichgewichte mit privater Information, aber für Handel. **Nicht zu verwechseln mit `globaler-schnitt-demo`** (globaler Schnitt in Graphen). **Eigennutz, Schwellenstrategien und die Beste-Antwort-Regel sind Modellannahmen**, keine Aussagen über Menschen oder echte Terminals.

## Kernfrage

*Wenn Betriebe nur ein verrauschtes Signal über den Zustand haben, wird dann aus „alle machen mit / keiner macht mit“ ein eindeutiges Gleichgewicht, und wo genau endet diese Eindeutigkeit?* Eigenes dieser Demo sind (1) das **Zählen** der Gleichgewichte statt eines postulierten (die Formel β ≥ α²/(2π) steht neben der Zahl, nicht davor), (2) die **Grenze** zwischen „eines“ und „drei“ in zwei Modellen, (3) **endlich viele Betriebe**, (4) die **Runden** der Beste-Antwort-Iteration und ihre Verlangsamung an der Grenze, (5) die **Stabilität** der mittleren Wurzel, (6) das **E-Mail-Spiel** als Beispiel für gemeinsames Wissen. Ohne Unsicherheit (θ allen bekannt) gibt es für jedes θ in (0, 1) zwei Gleichgewichte, „alle“ und „keiner“; diese Zustände haben bei α = 1 und y = ½ die Wahrscheinlichkeit 0,383, bei α = 4 die Wahrscheinlichkeit 0,683.

## Modelle

Zustand θ ~ N(y, 1/α) (Prior, α = Genauigkeit), Signal x = θ + e, e ~ N(0, 1/β), alle Betriebe mit demselben β. Strategie: mitmachen genau dann, wenn x ≤ x*. Kontinuum von Betrieben (Anteil Mitmachender l(θ) = Φ(√β(x* − θ))) oder N Betriebe (die N − 1 anderen zählen).

- **Modell R (Regimewechsel, Hauptmodell):** das Projekt startet, wenn der Anteil Mitmachender mindestens θ erreicht; Mitmachen bringt 1 bei Start und kostet immer c. Mit w = Φ⁻¹(θ̂) (θ̂ = kritischer Zustand, das Projekt startet für θ ≤ θ̂) und z_c = Φ⁻¹(c): **α·Φ(w) − √β·w = α·y + z_c·√(α+β)**, x* = θ̂ + w/√β. In ρ = α/√β und d = (αy + z_c√(α+β))/√β ist G(w) = ρΦ(w) − w = d; drei Lösungen genau dann, wenn ρ > √(2π) und G(−w₀) < d < G(w₀), w₀ = √(2 ln(ρ/√(2π))).
- **Modell L (linear, Umschalter):** Mitmachen bringt (Anteil − θ). Mit m = Anteil bei Indifferenz: **Φ(κ(m − y)) = m**, κ = α√((α+β)/(β(α+2β))), x* = (m(α+β) − αy)/β. Mehrere Lösungen genau dann, wenn κ > √(2π) und 1 − Y(κ) < y < Y(κ), Y = Φ(z₀) − z₀φ(z₀), z₀ = √(2 ln(κ/√(2π))) (Faltenkurve). Bei y = ½ ist die Grenze die positive Wurzel von 4πβ² + (2πα − α²)β − α³ = 0.
- **Endlich viele Betriebe (nur Modell R):** der Start gelingt, wenn (1 + K)/N ≥ θ mit K ~ Bin(N − 1, Φ(√β(x − θ))); die Startwahrscheinlichkeit wird stückweise integriert (Gauß–Legendre je Stück (j/N, (j+1)/N]). Im Modell L ändert N nichts: E[K/(N − 1) | θ] = Φ(√β(x − θ)) für jedes N.
- **E-Mail-Spiel:** Zustand G (Koordination lohnt: beide angreifen bringt M = 1, einsamer Angriff −L) mit Wahrscheinlichkeit ½, sonst B (Angriff bringt −L). In G schickt Spieler 1 eine Nachricht, jede geht mit Wahrscheinlichkeit ε verloren, jede angekommene wird bestätigt; Typ = Zahl der gesendeten Nachrichten (t₁ = ⌈k/2⌉, t₂ = ⌊k/2⌋), Schwelle q = L/(L+M). Typen bis 400, der Rest greift nie an.

Alle Herleitungen sind **eigene Herleitung, durch Zählung und Simulation gestützt**, nicht gegen die Literatur geprüft. Die Modellform des E-Mail-Spiels ist **angelehnt an Rubinstein, nicht gegen die Originalarbeit geprüft**.

## Befunde und Korrekturen gegenüber dem Plan

Alle Zahlen sind in dieser Demo neu gemessen (`tests/test_claims.py`). Das (α, β)-Raster ist mit der Wurzelsuche gezählt (`data/grid.json` mit Erzeuger-Hash), alles andere wird live gerechnet; Simulationen haben feste Zufalls-Seeds (SplitMix64).

| Nr. | Hypothese | Antwort | Beleg |
|---|---|---|---|
| H1 | Die Zahl der Schwellen-Gleichgewichte wird gezählt; die Formel β ≥ α²/(2π) ist hinreichend für Eindeutigkeit | **Ja** | Raster 29 × 31 (α, β) mit 25 Werten (y, c) im Modell R = 22.475 Zellen: oberhalb der Formel 0 von 11.975 mit mehreren Gleichgewichten, unterhalb 5.206 von 10.500 (49,6 %). Modell L (9 Werte y, 8.091 Zellen): oberhalb 0 von 4.311, unterhalb 1.846 von 3.780 (48,8 %). Der Prozentanteil hängt vom (y, c)-Raster ab (`test_h1_*`) |
| H2 | Modell R: die Formel ist genau die Eindeutigkeitsgrenze über alle (y, c), für festes (y, c) nur hinreichend | **Ja** | Alle 420 (α, β)-Zellen, in denen es mehrere gibt, liegen unterhalb der Formel, und jede (α, β)-Zelle unterhalb hat für mindestens ein (y, c) drei Gleichgewichte (bei y = c = ½ immer). Für festes (y, c) sind 5.294 der 10.500 Zellen unterhalb trotzdem eindeutig (`test_h2_*`) |
| H3 | Modell L: die wahre Grenze liegt unter der Formel | **Ja** | Kritisches β bei y = ½ für α = 0,5 / 1 / 2 / 4 / 8 / 16 / 30: 0,0372 / 0,1416 / 0,5268 / 1,9224 / 6,953 / 25,27 / 82,62, das 0,935- / 0,890- / 0,827- / 0,755- / 0,683- / 0,620- / 0,577-Fache der Formel. Geschlossene Form gleich gezählter Grenze auf fünf Stellen (α = 1, 4, 8) (`test_h3_*`) |
| H4 | Die Zählung der Vormessung V3 auf einem x-Gitter y ± 6 verpasst Gleichgewichte außerhalb des Gitters | **Ja** | Zellen α ∈ {1, 2, 4, 6, 10, 20} × β ∈ {0,1 … 400}, y = ½ (60): exakt haben **23** drei, die V3-Zählung findet 14; in 9 Zellen steht dort 1 statt 3: (α; β) = (2; 0,1), (4; 0,1), (6; 0,1), (6; 0,5), (10; 0,1), (10; 0,5), (20; 0,1), (20; 0,5), (20; 1). Die Formel ist wahr und eindeutig in 34, wahr und mehrfach in 0, falsch und eindeutig in 3 ((4; 2), (6; 5), (20; 50)), falsch und mehrfach in 23 Zellen; das Kriterium κ² ≤ 2π trifft 60 von 60. Bei α = 4, β = 0,1, y = ½ liegen die äußeren Wurzeln bei m = 2,07·10⁻¹⁰ und 1 − 2,07·10⁻¹⁰, x* = −20 und 21 (`test_h4_*`, `test_h5_edge_*`) |
| H5 | Der Bereich mit drei Gleichgewichten im Prior y ist ein Intervall um ½ (Faltenkurve); am Prior-Rand ist es eindeutig | **Ja** | 297 Zellen (α = 1, 4, 10; 11 Werte β; y = 0,1 … 0,9): 0 Abweichungen. α = 4: β = 0,1 / 0,25 / 0,5 / 1 hat drei Gleichgewichte für y in [0,2; 0,8] / [0,3; 0,7] / [0,4; 0,6] / nur 0,5 (Raster 0,1). Am Rand (α = 4, β = 1, y = 0,3) genau eines: m = 0,9944, x* = 3,77; für α = 1, β = 100, y = 0,3 liegt x* bei 0,5079 (`test_h5_*`) |
| H6 | Der Aufwand c verschiebt und begrenzt das Band mit drei Gleichgewichten; θ̂ → 1 − c erst bei großem β | **Ja** | Band in c bei α = 8, β = 4: y = 0,5 → (0,4163; 0,5837), y = 0,3 → (0,5989; 0,7496), y = 0,7 → (0,2504; 0,4011); α = 16, β = 4, y = ½ → (0,1896; 0,8104); α = 4, β = 4 (ρ = 2) nie. θ̂ bei α = 1, y = ½, β = 1 / 16 / 256 / 10.000: c = 0,25 → 0,9144 / 0,7778 / 0,7555 / 0,7508 (Grenzwert 0,75); c = 0,75 → 0,0856 / 0,2222 / 0,2445 / 0,2492 (Grenzwert 0,25); c = ½ immer 0,5000 (`test_h6_*`) |
| H7 | Unsicherheit kostet Projekte, gegenüber dem besten Gleichgewicht „alle“ bei bekanntem θ | **Ja, mit dieser Vergleichsbasis** | Verloren (gegenüber „alle machen mit“ bei bekanntem θ; dort gibt es auch „keiner“, und welches gewählt wird, ist offen; bei „keiner“ brächte die Unsicherheit mit P(θ ≤ θ̂) Projekte) P(θ̂ < θ < 1) bei β = 1.024, y = ½: α = 1: 0,092 / 0,191 / 0,291 für c = 0,25 / 0,5 / 0,75; α = 4: 0,142 / 0,341 / 0,540. Das Projekt startet dann mit P(θ ≤ θ̂) = 0,600 / 0,500 / 0,400 (α = 1) bzw. 0,699 / 0,500 / 0,301 (α = 4), gegen P(θ < 1) = 0,691 bzw. 0,841 (`test_h7_*`) |
| H8 | Endliches N: im Modell L kein Effekt; im Modell R höhere Schwellen, bei N ≤ 3 nur ein Gleichgewicht, bei knapp eindeutigem Kontinuum N = 5, 10 mit drei | **Ja** | Zelle α = 8, β = 4, y = c = ½ (Kontinuum: x* = −0,910 / 0,500 / 1,910): N = 2 / 3 / 5 ein Gleichgewicht (x* = 1,989 / 1,978 / 1,961), N = 10 / 25 / 100 / 250 / 1.000 drei, mittleres x* = 0,151 / 0,390 / 0,474 / 0,490 / 0,497. Zelle c = 0,6 (Kontinuum eines): N = 5 und 10 drei, N = 25 / 100 ein (x* = −1,067 / −1,163). Zelle ρ = 2 (eindeutig): x* = 1,409 / 1,267 / 1,070 / 0,864 / 0,675 / 0,548 / 0,520 / 0,505 für N = 2 / 3 / 5 / 10 / 25 / 100 / 250 / 1.000. Orakel: Startwahrscheinlichkeit gegen `scipy.integrate.quad` mit `scipy.stats.binom` (Testtoleranz 10⁻⁹) und gegen Rejection-Sampling des Spiels; Modell L: Erwartungsnutzen bei N = 3 gleich dem Kontinuumswert (Simulation, 4 Standardfehler). Auf einem Raster gezählt, kein Beweis (`test_h8_*`) |
| H9 | Die Beste-Antwort-Runden wachsen an der Grenze; die Iteration endet an den äußeren Wurzeln | **Ja** | Modell R, α = 8, y = c = ½, Grenze β = 10,186: β = 3 / 4 / 6 / 8 / 10 / 10,1 / 10,2 / 10,3 / 11 / 12 / 16 / 100 → 8 / 12 / 31 / 88 / 1.362 / 2.900 / **35.325** / 4.820 / 772 / 385 / 163 / 90 Runden bis auf 10⁻⁶ (Windows; der Test erlaubt 2 %). α = 16: β = 40 / 41 → 2.065 / 12.738. Modell L, α = 4, Grenze 1,9224: β = 1,5 / 1,9 / 1,92 / 1,93 / 2 → 58 / 1.272 / 11.045 / 7.725 / 815. Monoton in allen Zellen, von beiden Enden gleich viele Runden (Spiegelsymmetrie bei y = c = ½) (`test_h9_*`) |
| H10 | E-Mail-Spiel: Angriff nur für q ≤ (1−ε)/(2−ε); gemeinsame p-Überzeugung höchstens bis zur selben Grenze; eine Operator-Runde je Nachrichtenstufe | **Ja** (eine Modellform) | 36 Zellen (ε = 0,02 … 0,5; q bei 0,9 und 1,1 mal Grenze und 0,3 / 0,5 / 0,6 / 0,9): 0 Abweichungen. Bei L = M (q = ½) und ε = 0,1 (Grenze 0,4737) greift nie jemand an, obwohl bis zu 801 Nachrichten ankommen können. Vollaufzählung der 512 reinen Profile (J = 3, ε = 0,1): q = 0,3 und 0,45 je zwei Gleichgewichte, q = 0,5 genau eins. G ist in 800 von 801 Zuständen gemeinsame p-Überzeugung für p knapp unter der Grenze (ε = 0,05 / 0,1 / 0,3: 0,4867 / 0,4732 / 0,4114), in keinem für p knapp darüber (0,4877 / 0,4742 / 0,4122); dann braucht der Operator 801 Runden. Gegenseitiges Wissen der Stufe n = 5 / 10 / 20: 0,5905 / 0,3487 / 0,1216 (Formel (1 − ε)ⁿ), simuliert 0,5905 / 0,3471 / 0,1199 (400.000 Spiele, Zufalls-Seed 21); Auszahlungen im Profil „ab Typ 1“ 0,400 und 0,450 (`test_h10_*`) |
| H11 | Mehrere Schwellen-Gleichgewichte heißen: das mittlere ist instabil | **Ja** (hier gemessen) | Modell R, α = 8, β = 4: Steigung der Antwortfunktion B' = 0,3600 / **1,3314** / 0,3600 an den drei Wurzeln; B' < 1 genau dann, wenn G'(w) < 0 (eigene Herleitung), zweiter Weg: zentrale Differenz der Antwortfunktion. Iteration ab ± 0,001 neben der mittleren Wurzel läuft von ihr weg zur unteren bzw. oberen. Modell L, α = 4, β = 1: 0,3783 / **1,4567** / 0,3783. In jeder Zelle der Stichprobe des Rasters ist die einzige Wurzel stabil, bei drei sind es die äußeren (`test_h11_*`) |
| H12 | Es gibt keine Gleichgewichte außerhalb der Schwellenstrategien | **Offen, nicht gezeigt** | Nur Schwellenstrategien sind gerechnet. Ob andere Gleichgewichte existieren, ist weder gezeigt noch widerlegt (siehe „Ehrliche Grenzen“) |

### Korrekturen gegenüber dem Plan

- **Der Hook „Unsicherheit macht das Gleichgewicht eindeutig“ gilt nur bei hinreichend genauen Signalen**; darunter bleiben bis zu drei Schwellen-Gleichgewichte (H1, H2). Die Demo fragt deshalb „wann“.
- **V3:** „12 von 60 Zellen widersprechen der ‚genau dann‘-Lesart“ stimmt nicht: 3 widersprechen, 9 Zellen waren Gitter-Artefakte (H4). Die Formel gehört (laut Auszug eines Fremdartikels, nicht geprüft) zum Regimewechsel-Modell; in unserer Rechnung ist sie dort genau die Eindeutigkeitsgrenze über alle (y, c), im linearen Modell nur hinreichend.
- **Verlorene Projekte, α = 1, β = 1.024, c = ½:** der Plan nannte 0,192; gemessen 0,191 (= Φ(½) − ½ = 0,1915 abgerundet).
- **Raster:** der Plan zählte 1.500 Zellen (875 oberhalb, 625 unterhalb, davon 277 mit mehreren = 44,3 %); mit dem hier gewählten Raster (29 × 31 × 25) sind es 49,6 %. Die Zahlen sind nicht vergleichbar, der Anteil hängt am Raster. Qualitativ gleich: oberhalb nie mehrere.
- **E5, beide Startpunkte:** bei y = c = ½ sind die Runden von „kaum einer“ und „fast alle“ gleich (Spiegelsymmetrie); der Plan führte beide Richtungen als getrennte Messung. Die Demo zeigt eine Runden-Spalte und sagt es dazu.
- **E-Mail-Grenze für andere Priors:** (1−ε)/(2−ε) gilt für p_G = ½ und ε ≤ ½; allgemein ist die Grenze max(λ₀, (1−ε)/(2−ε)) mit λ₀ = p_G·ε/(1 − p_G + p_G·ε) (Typ 0 von Spieler 2). Eigene Herleitung, in einem Raster aus 5 ε × 4 p_G mit je bis zu 7 Werten q gezählt, über 100 Zellen (`test_email.py`); in der App ist p_G = ½ fest.
- **Rang-Überzeugungen:** Morris, Shin und Yildiz (2016) stützen globale Spiele laut Abstract auf Rang-Überzeugungen, nicht auf gemeinsame p-Überzeugung; der E-Mail-Tab nutzt die gemeinsame p-Überzeugung als eigenes Maß und beansprucht keine Gleichheit mit der Arbeit.
- **Port und Name:** Port 9206 (9202 und 9205 belegt), Präfix `glo`, Linie `koordination`.
- **H11 gemessen, H12 offen** (Entscheidung zum Bau).

## Methodik

- **Kern ohne Gitter:** Modell L hat auf drei Stücken (an den Wendestellen von g, y ± z₀/κ) eine monotone Funktion, Modell R ebenso (±w₀); auf jedem Stück sucht eine Bisektion die Wurzel. Wurzeln dicht an 0 oder 1 (kleines β) werden in m bzw. u = 1 − m mit geometrischer Mitte gesucht; unter 10⁻³⁰⁰ meldet die Rechnung einen Unterlauf (κ·min(y, 1−y) > 37).
- **Endliches N:** Stückintegration statt Gitter über θ (ein Gitter erzeugt an den Sprungstellen der Binomialzahl Scheingleichgewichte); Suchfenster mit Beweis (links davon ist die Startwahrscheinlichkeit größer als c, rechts kleiner); Raster mit 81 bis 241 Punkten, Vorzeichenwechsel, Verfeinerung nach Illinois.
- **Beste-Antwort-Iteration:** Fixpunkt θ̂(x) mit Newton und Klammer [0, 1]; gezählt werden Runden bis auf 10⁻⁶ an die kleinste bzw. größte Wurzel (kein Abbruch über die Schrittweite, der nahe der Grenze täuscht); Obergrenze 200.000 Runden, gemeldet als „abgebrochen“.
- **E-Mail-Spiel:** exakte Typenkette (803 Zustände), bedingte Überzeugungen je Typ (nie unbedingte Erwartungsnutzen mit absoluter Toleranz verglichen: Typen mit Wahrscheinlichkeit unter 10⁻¹⁵ sähen sonst wie Angriffe aus), größtes Gleichgewicht durch Iteration ab „alle greifen an“, gemeinsame p-Überzeugung als größter Fixpunkt des Operators F ↦ F ∩ B₁ᵖ(F) ∩ B₂ᵖ(F).
- **Zufall** nur in der gezeichneten Spielrunde und der Protokollsimulation: SplitMix64 (`glo_rng.py`, wörtliche Kopie aus `ab-test-demo`), Box–Muller für Normalverteilte. Kein scipy zur Laufzeit.
- **Vorgerechnet** (`tools/make_grid.py`, `data/grid.json` mit Erzeuger-Hash über Raster, Modell R, Modell L und Kern): das (α, β)-Raster der Gleichgewichtszahl (E1). Alles andere läuft live; endliches N bis 1.000 kostet rund 4 Sekunden (einmal gemessen, nicht getestet), die engste Iteration (35.325 Runden) rund 0,3 Sekunden.
- **Experimente:** E1 Wo gibt es drei?, E2 Prior und Aufwand verschieben das Band, E3 Was kostet die Unsicherheit?, E4 Endlich viele Betriebe, E5 Runden bis zur Eindeutigkeit, E6 Gemeinsames Wissen (E-Mail-Spiel).

## Ehrliche Grenzen

- **Nur Schwellenstrategien.** Andere Gleichgewichte (nicht-monotone Strategien) sind weder ausgeschlossen noch gesucht; dass es keine gibt, ist nicht gezeigt (H12). Ein Aufzählungsversuch bei kleinem N wurde nicht gemacht.
- **Alle Aussagen gelten für diese Modelle:** gleiche Signalgenauigkeit für alle, normalverteilter Prior und normalverteiltes Rauschen, Eigennutz, Beste-Antwort-Runden als Modellannahme (kein Lernprozess). Ungleiche Genauigkeiten, schiefe Priors und dicke Ränder sind nicht gerechnet.
- **Kontinuum und N Betriebe sind verschiedene Spiele.** Die Zahlen bei endlichem N sind auf einem Raster gezählt: ein Paar dicht an einem Berührpunkt könnte fehlen. In der Zelle α = 8, β = 4, y = c = ½ ist für N = 2 bis 100 der Abstand zwischen Startwahrscheinlichkeit und c abseits der Wurzeln mindestens 5·10⁻⁴ (`test_finite.py`); für die übrigen Zellen ist das nicht eigens geprüft.
- **Randwurzeln:** bei kleinem β liegen äußere Wurzeln des Modells L bei Anteilen von 10⁻¹⁰ und weniger („praktisch nie mitmachen“). Im Eckfall α = 30, β = 0,1, y = ½ liegen sie unter 10⁻³⁰⁰ (als 0 und 1 gemeldet).
- **Die Faustformel im Modell L ist nur hinreichend**, im Modell R genau die Grenze über alle (y, c); für ein festes (y, c) im Modell R ist sie ebenfalls nur hinreichend.
- **Die Rundenzahlen** sind Windows-Messungen; die Tests prüfen sie als Bänder (2 %). Von beiden Enden gleich viele Runden gilt nur bei y = c = ½.
- **E-Mail-Spiel:** eigene Modellform, angelehnt an Rubinstein, nicht gegen die Originalarbeit geprüft; Typen bis 400 abgeschnitten (der Rest greift nie an); p_G fest ½; die Grenze (1−ε)/(2−ε) gilt dafür und für ε ≤ ½. Dass das Operator-Ergebnis „eine Runde je Nachrichtenstufe“ ist, ist für diese Kette gemessen, nicht allgemein.
- **Strategische Substitute:** globale Spiele, in denen Mitmachen mit der Zahl der Mitmachenden weniger lohnt, sind nicht gerechnet.
- Mausklicks auf die Diagramme sind nicht getestet (die Diagramme haben keine Auswahl). Nur an erfundenen Parametern geprüft; keine Messung an Menschen oder Terminals.

## Quellen

Am Abstract bzw. an der Katalogseite geprüft (nicht im Volltext gelesen): Carlsson und van Damme (1993), „Global Games and Equilibrium Selection“, *Econometrica* 61(5), 989–1018 (Abstract: ein globales Spiel ist ein Spiel mit unvollständiger Information, in dem die Auszahlungen zufällig aus einer Klasse von Spielen gezogen werden und jeder Spieler ein verrauschtes Signal sieht; bei 2×2-Spielen führt die iterierte Elimination dominierter Strategien für verschwindendes Rauschen zum Kriterium der Risikodominanz). Morris, Shin und Yildiz (2016), „Common belief foundations of global games“, *Journal of Economic Theory* 163 (Abstract laut Websuche vom 2026-10-27: eindeutige Rationalisierbarkeit bei näherungsweiser gemeinsamer Gewissheit von Rang-Überzeugungen). Harrison und Jara-Moroni (2021), „Global games with strategic substitutes“, *International Economic Review* 62(1), 141–173 (Abstract: binäre Aktionen, N Spieler, bei asymmetrischen Auszahlungen wird ein Gleichgewicht ausgewählt, Wechselschwellen je Spieler verschieden); nur als Hinweis, **nicht gerechnet**. Rubinstein (1989), „The Electronic Mail Game: Strategic Behavior Under ‚Almost Common Knowledge‘“, *American Economic Review* 79(3), 385–391: nur die Kurzbeschreibung des Nachdrucks bei Cambridge University Press (beliebig genaue Näherungen an gemeinsames Wissen haben ganz andere strategische Folgen als gemeinsames Wissen), nicht der AER-Abstract; die Modellform dieser Demo ist angelehnt und nicht gegen die Arbeit geprüft. Die Zahlen und Herleitungen dieser Demo sind eigene Rechnung und stammen aus keiner dieser Arbeiten. Nicht geprüft und nicht verwendet: Morris und Shin (1998, 2003), Monderer und Samet (1989).

## Verifikation

- **Kern gegen Handrechnung** (`tests/conftest.py`, `test_linear.py`, `test_regime.py`, `test_finite.py`, `test_iteration.py`, `test_email.py`): Modell L mit α = 2, β = 1 (κ² = 3, m = ½, kritisches β 0,5268), Modell R mit α = β = 4 (w = 0, θ̂ = x* = ½) und α = 8, β = 4 (Band in d (1,634; 2,366), Rand in c 0,5837), endliches N (N = 2, θ = 0,7, p = 0,3 gibt 0,3; N = 3, θ = ½, p = ½ gibt 0,75), E-Mail-Spiel mit ε = 0,1 (1/1,9 = 0,5263, Grenze 0,4737, Auszahlung 0,4).
- **Orakel** (`tests/test_oracle_globale_spiele.py`): Φ gegen `scipy.special.ndtr`; Modell-L-Wurzeln gegen `scipy.optimize.brentq` auf einem eigenen Gitter beider Ränder und die x-Form der V3-Formel (Identität auf 10⁻¹²); Modell-R-Wurzeln gegen eine unabhängige x-Rechnung (θ̂(x) per Bisektion, dann Indifferenz) in 100 Zufallszellen; Gleichgewichtszahl gegen Band und Faltenkurve; Startwahrscheinlichkeit gegen `scipy.integrate.quad` und `scipy.stats.binom` sowie gegen Rejection-Sampling; Fixpunkt θ̂ gegen `brentq`; E-Mail-Spiel gegen die Vollaufzählung aller 512 reinen Profile mit exakter Bruchrechnung, gemeinsame p-Überzeugung gegen eine Mengenschleife, Protokollsimulation mit numpy-Zufall (ein anderer Strom als SplitMix64).
- **Zweig-Tests:** Zellen mit einer und mit drei Wurzeln in beiden Modellen, Randwurzel (m < 10⁻⁸), Unterlauf, endliches N weicht vom Kontinuum ab (R) und nicht ab (L), jedes Preset hat die behauptete Zahl, E-Mail oberhalb und unterhalb der Grenze. Eine Spalte mit lauter Nullen gäbe es nur, wenn eine Variante nie liefe.
- **Randfälle** (`test_edge_cases.py`): N = 2, β genau an der Formel, q genau an der Angriffsgrenze, Reglerecken beider Modelle, Start im Gleichgewicht, J = 0, ε = 1.
- **Claims** (`tests/test_claims.py`): jede Zahl dieser README. **Verdikte** (`test_verdicts.py`): jede Schwelle und jede Zahl im Text an handgesetzten Werten. **Messreihe** (`test_measurements.py`): Hash, Achsen, jede Zelle gegen das geschlossene Band mit Abstand zu den Berührpunkten, Zweig „oberhalb nie mehrere“.
- **Plattform (CI läuft auf Linux, entwickelt wird auf Windows):** kein Test hängt an einem Löser (kein LAPACK, kein scipy-Wurzelfinder); die Wurzeln sind Bisektionen auf `math.erfc`, Rundenzahlen werden als Bänder geprüft, Zählungen bei endlichem N nur mit Abstand zu Berührpunkten, Simulationen mit festem Strom und Marge 4 Standardfehler, keine Zeit-Assertion. scipy steht nur in `requirements-dev.txt` und nur in den Orakel-Tests (`pytest.importorskip`).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum, alle Regler zugleich je Modell mit N = 2 und N = 1.000, Unterlauf-Eckfall, die drei ▶️-Knöpfe einzeln), Permalink, PDF, Presets samt Hilfetext-Aussagen, Layout, Footer, Laufzeit-Importe.
- **Wie die Zahlen gesichert sind:** jede Zahl hat mindestens einen zweiten Rechenweg (Handrechnung, `scipy`, Aufzählung, Simulation) oder ist eine geschlossene Form, die gegen die Zählung geprüft ist. Ob jede Aussage eines Tests wirklich etwas fängt, ist mit keinem eigenen Verfahren gemessen. Die Aussagenprüfung durch einen unabhängigen Prüfer steht noch aus.

## Tests

318 Tests (`pytest tests`); die Laufzeit hängt stark an der Last des Rechners (ein Lauf unter paralleler Last: rund 260 Sekunden); die langsamsten sind die AppTest-Läufe über alle Regler und die Orakel gegen die Simulation. Zusammensetzung: Kern gegen Handrechnung 73 (`test_core.py` 10, `test_linear.py` 18, `test_regime.py` 10, `test_finite.py` 23, `test_iteration.py` 13), E-Mail-Spiel 19, Orakel 45 (`test_oracle_globale_spiele.py`), Claims 41, Verdikte und Formate 27, Randfälle 21, Messreihe und Experimente 19 (`test_measurements.py` 9, `test_experiments.py` 10), Szenario und SplitMix64-Kopie 14 (`test_scenario.py` 9, `test_rng.py` 5), Abbildungen und PDF 5, Presets 11, Oberfläche 42 (`test_app.py` 23, `test_layout.py` 16, `test_footer.py` 2, `test_runtime_imports.py` 1).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Hauptansicht und sechs Tabs mit je einem Experiment |
| `glo_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets mit Hilfetexten |
| `glo_core.py` | Normalverteilung ohne scipy, Bisektion |
| `glo_linear.py` | Modell L: Wurzeln, kritisches β, Faltenkurve |
| `glo_regime.py` | Modell R: Wurzeln, Band in d und c, Kennzahlen |
| `glo_finite.py` | Endlich viele Betriebe: Binomialzahl, Startwahrscheinlichkeit, Wurzelsuche, beste Antwort |
| `glo_iteration.py` | Beste-Antwort-Iteration, Runden, Steigung der Antwortfunktion |
| `glo_email.py` | E-Mail-Spiel: Typenkette, Gleichgewicht, gemeinsame p-Überzeugung, Protokollsimulation |
| `glo_scenario.py` | Einstellungen und gezeichnete Spielrunde |
| `glo_grid.py` | Raster der Gleichgewichtszahl (E1) |
| `glo_experiments.py` | Die Reihen E1 bis E6, Rasterdatei mit Erzeuger-Hash |
| `glo_evaluation.py` | Ansicht der Hauptseite, Verdikte als reine Funktionen |
| `glo_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `glo_pdf_export.py` | PDF-Export (fpdf2) |
| `glo_layout.py` | Fester Kopf (Stylesheet) |
| `glo_presets.py` | Permalink, Presets (Schnellstart in der Seitenleiste), Zufalls-Seed-Button |
| `glo_rng.py` | SplitMix64 (wörtliche Kopie aus `ab-test-demo`) |
| `data/grid.json` | Vorgerechnetes Raster für E1 mit Erzeuger-Hash |
| `tools/make_grid.py` | Erzeugt die Rasterdatei neu (rund eine halbe Minute) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy (nur Orakel) |
| `tests/` | Handrechnung, Orakel, Claims, Verdikte, Randfälle, Presets, Szenario und Kopie, Messreihe, Abbildungen, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Netze und Ansteckung** (Schwellenregeln auf Netzen): siehe `komplexe-ansteckung-demo`. **Wie lange ein Wechsel dauert:** `konventionswechsel-demo`.
- **Strategische Substitute** und **Bank-Runs** (Märkte und Anreize im Netz).
- **Andere Prior-Verteilungen und ungleiche Signalgenauigkeit.**
- **Gleichgewichte außerhalb der Schwellenstrategien** (H12).
- **Menschenexperimente:** Eigennutz und Beste-Antwort-Regel sind Modellannahmen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

Die Demo läuft mit Python 3.12 oder neuer (CI: 3.12). In der Streamlit Cloud ist die Python-Version in den App-Einstellungen gewählt. <!-- lint-ok: dezimalpunkt -->

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
