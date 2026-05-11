# Agentic Edge Orchestration — Distributed Systems Project

Sistema multi-agente per l'orchestrazione distribuita di risorse edge, basato su Ray Actor Model, Contract Net Protocol (CNP) e CRDT per la consistenza dello stato distribuito.

---

## Struttura del progetto

```
distributed systems project/
├── src/
│   ├── protocol.py            # Tipi di dato e messaggi A2A (CFP, Offer, Accept…)
│   ├── crdt_catalogue.py      # CRDT: LWWRegister, NodeSnapshot, ResourceCatalogue
│   ├── agents.py              # ResourceAgent, TaskAgent, NashTaskAgent (Ray Actors)
│   ├── main.py                # Demo end-to-end (Fase 3 — simulazione base)
│   └── phase4_experiments.py  # Esperimenti con 30 run e intervalli di confidenza
├── results_CI_v4/             # Output degli esperimenti (grafici e raw_results.json)
├── relazione.tex              # Relazione del progetto (LaTeX)
├── state_of_the_art.tex       # Stato dell'arte (LaTeX)
├── report.tex                 # Report sperimentale (LaTeX)
├── refs.bib                   # Bibliografia
└── README.md                  # Questo file
```

---

## Moduli

### `protocol.py`
Definisce le strutture dati condivise tra gli agenti:
- `TaskRequirements` — requisiti di una task (CPU, memoria, latenza massima, durata, priorità, tipo)
- `ResourceOffer` — offerta di un nodo (risorse disponibili, punteggio, latenza stimata)
- `PlacementPolicy` — politica di selezione del nodo: `LATENCY_FIRST`, `ENERGY_FIRST`, `BALANCED`
- Funzioni helper per costruire messaggi CNP: `make_cfp()`, `make_offer()`, `make_accept()`

### `crdt_catalogue.py`
Implementa la consistenza distribuita senza coordinatore centrale:
- `LWWRegister` — registro Last-Write-Wins con clock di Lamport e tiebreak lessicografico
- `NodeSnapshot` — snapshot di un nodo edge (6 registri LWW indipendenti: cpu, memoria, latenza, energia, task attivi, online)
- `ResourceCatalogue` — G-Map CRDT grow-only; merge commutativo, associativo e idempotente; gossip-based

### `agents.py`
Implementa gli attori Ray del sistema:
- `ResourceAgent` — rappresenta un nodo edge fisico; gestisce il proprio stato, risponde ai CFP con offerte, esegue le task accettate, aggiorna il catalogo CRDT via gossip
- `TaskAgent` — agente che piazza una singola task tramite CNP: invia CFP, seleziona la migliore offerta secondo la policy, invia ACCEPT
- `NashTaskAgent` — estende `TaskAgent` con Iterative Best Response (IBR): rinegozia i requisiti fino a raggiungere un equilibrio di Nash (massimo 5 round, rilassamento SLA α=0.20 per round)

### `main.py`
Demo end-to-end della Fase 3. Avvia 4 nodi edge simulati, piazza 8 task con policy diverse e stampa le metriche di placement e lo stato del catalogo CRDT.

### `phase4_experiments.py`
Suite sperimentale completa (Fase 4). Esegue 30 run indipendenti per 5 scenari e produce grafici con intervalli di confidenza al 95%.

---

## Configurazione dei nodi edge

| Nodo        | CPU (core) | Memoria | Latenza | Energy Score |
|-------------|-----------|---------|---------|--------------|
| edge-node-1 | 8.0       | 4096 MB | 15 ms   | 0.3          |
| edge-node-2 | 4.0       | 2048 MB | 40 ms   | 0.5          |
| edge-node-3 | 2.0       | 1024 MB | 80 ms   | 0.2          |
| edge-node-4 | 16.0      | 8192 MB | 25 ms   | 0.8          |

---

## Dipendenze

```bash
pip install ray matplotlib numpy
```

Versione Python consigliata: **3.10+**

> Ray deve essere installato nella stessa versione su tutti i nodi del cluster (se si usa un cluster reale). Per la simulazione locale, un singolo nodo è sufficiente.

---

## Esecuzione

### Demo base (Fase 3)

Avvia la simulazione end-to-end con 4 nodi e 8 task:

```bash
cd src/
python main.py
```

Output atteso:
- Tabella di placement per ogni task (nodo assegnato, policy, score, overhead A2A, latenza stimata)
- Stato di ogni nodo edge dopo il placement (CPU/memoria residua, task attivi)
- Percentuale di convergenza del catalogo CRDT dopo un round di gossip
- File `src/results.json` con i risultati in formato JSON

### Esperimenti completi (Fase 4)

Esegue i 5 scenari sperimentali con 30 run ciascuno:

```bash
cd src/
python phase4_experiments.py
```

Output:
- `results_CI_v4/raw_results.json` — dati grezzi di tutti i run per ogni scenario
- `results_CI_v4/plot_placement_latency.png` — latenza di placement media ± IC 95%
- `results_CI_v4/plot_a2a_overhead.png` — overhead A2A medio ± IC 95%
- `results_CI_v4/plot_sla_violations.png` — tasso di SLA violation ± IC 95%
- `results_CI_v4/plot_crdt_convergence.png` — tempo di convergenza CRDT ± IC 95%
- `results_CI_v4/plot_partition_divergence.png` — divergenza del catalogo durante partizione di rete (S4)
- `results_CI_v4/plot_nash_convergence.png` — round di IBR per task e convergenza all'equilibrio di Nash (S5)
- `results_CI_v4/summary_dashboard.png` — dashboard riepilogativa di tutti gli scenari

> **Tempo di esecuzione stimato:** circa 5–15 minuti su un laptop moderno, in base alle risorse disponibili per Ray.

---

## Scenari sperimentali

| ID | Nome               | Descrizione                                                              |
|----|--------------------|--------------------------------------------------------------------------|
| S1 | Baseline           | 10 task su 4 nodi con pieno utilizzo; misura le metriche nominali        |
| S2 | High Load          | 20 task su 4 nodi (overload); valuta il comportamento in caso di rigetto |
| S3 | Node Failure       | Crash di edge-node-1 durante l'esecuzione; test di fault tolerance        |
| S4 | Network Partition  | Partizione di rete tra i nodi; verifica della consistenza CRDT al ripristino |
| S5 | Nash Equilibrium   | 8 task con `NashTaskAgent`; confronto greedy vs IBR, verifica NE         |

---

## Architettura del sistema

```
TaskAgent (Ray Actor)
    │
    │  CFP (Contract Net Protocol)
    ▼
ResourceAgent × N (Ray Actor)
    │  PROPOSE / REJECT / COUNTER_OFFER
    └──────────────────────────────────►  TaskAgent
                                              │
                                         ACCEPT / INFORM
                                              │
                                         ResourceAgent (selezionato)
                                              │
                                    gossip CRDT ◄──► ResourceAgent × (N-1)
```

La comunicazione A2A è realizzata tramite il message-passing di Ray (`actor.method.remote()`), che emula un layer JSON-RPC 2.0 tra agenti isolati in processi separati. Il catalogo delle risorse è un G-Map CRDT replicato su tutti i nodi e sincronizzato via gossip alla fine di ogni round di negoziazione.

---

## Metriche misurate

- **Placement Latency (ms):** tempo totale di negoziazione CNP dalla CFP all'ACCEPT
- **A2A Overhead (ms):** porzione del placement latency attribuibile alla comunicazione inter-agente
- **SLA Violation Rate (%):** percentuale di task piazzate oltre il vincolo di latenza massima dichiarato
- **CRDT Convergence Time (ms):** tempo per la convergenza completa del catalogo distribuito dopo un round di gossip

Gli intervalli di confidenza sono calcolati con la distribuzione t di Student a 29 gradi di libertà (n=30 run, α=0.05): `IC = t(29, 0.975) × s/√30`.
