# Agentic Edge Orchestration — Distributed Systems Project

Multi-agent system for distributed edge resource orchestration, based on the Ray Actor Model, Contract Net Protocol (CNP), and CRDTs for distributed state consistency.

---

## Project Structure

```
distributed systems project/
├── src/
│   ├── protocol.py            # Data types and A2A messages (CFP, Offer, Accept…)
│   ├── crdt_catalogue.py      # CRDT: LWWRegister, NodeSnapshot, ResourceCatalogue
│   ├── agents.py              # ResourceAgent, TaskAgent, NashTaskAgent (Ray Actors)
│   ├── main.py                # End-to-end demo (Phase 3 — base simulation)
│   └── experiments.py         # Experiments with 30 runs
├── results/                   # Experiment outputs
└── README.md                  # This file
```

---

## Modules

### `protocol.py`
Defines the shared data structures used by all agents:
- `TaskRequirements` — task requirements (CPU, memory, maximum latency, duration, priority, type)
- `ResourceOffer` — node offer (available resources, score, estimated latency)
- `PlacementPolicy` — node selection policy: `LATENCY_FIRST`, `ENERGY_FIRST`, `BALANCED`
- Helper functions for building CNP messages: `make_cfp()`, `make_offer()`, `make_accept()`

### `crdt_catalogue.py`
Implements distributed consistency without a central coordinator:
- `LWWRegister` — Last-Write-Wins register with Lamport clock and lexicographic tiebreak
- `NodeSnapshot` — snapshot of an edge node (6 independent LWW registers: cpu, memory, latency, energy, active tasks, online)
- `ResourceCatalogue` — grow-only G-Map CRDT; commutative, associative, and idempotent merge; gossip-based

### `agents.py`
Implements the Ray actors of the system:
- `ResourceAgent` — represents a physical edge node; manages its own state, responds to CFPs with offers, executes accepted tasks, updates the CRDT catalogue via gossip
- `TaskAgent` — agent that places a single task via CNP: sends CFP, selects the best offer according to the policy, sends ACCEPT
- `NashTaskAgent` — extends `TaskAgent` with Iterative Best Response (IBR): renegotiates requirements until a Nash Equilibrium is reached (maximum 5 rounds, SLA relaxation α=0.20 per round)

### `main.py`
End-to-end demo for Phase 3. Starts 4 simulated edge nodes, places 8 tasks with different policies, and prints placement metrics and the CRDT catalogue state.

### `experiments.py`
Full experimental suite (Phase 4). Runs 30 independent runs for 5 scenarios and produces plots with 95% confidence intervals.

---

## Edge Node Configuration

| Node        | CPU (cores) | Memory  | Latency | Energy Score |
|-------------|-------------|---------|---------|--------------|
| edge-node-1 | 8.0         | 4096 MB | 15 ms   | 0.3          |
| edge-node-2 | 4.0         | 2048 MB | 40 ms   | 0.5          |
| edge-node-3 | 2.0         | 1024 MB | 80 ms   | 0.2          |
| edge-node-4 | 16.0        | 8192 MB | 25 ms   | 0.8          |

---

## Dependencies

```bash
pip install ray matplotlib numpy
```

Recommended Python version: **3.10+**

> Ray must be installed at the same version across all cluster nodes (if using a real cluster). For local simulation, a single node is sufficient.

---

## Running the Project

### Base Demo (Phase 3)

Starts the end-to-end simulation with 4 nodes and 8 tasks:

```bash
cd src/
python main.py
```

Expected output:
- Placement table for each task (assigned node, policy, score, A2A overhead, estimated latency)
- State of each edge node after placement (remaining CPU/memory, active tasks)
- CRDT catalogue convergence percentage after one gossip round
- File `src/results.json` with results in JSON format

### Full Experiments (Phase 4)

Runs the 5 experimental scenarios with 30 runs each:

```bash
cd src/
python experiments.py
```

Output:
- `results/raw_results.json` — raw data from all runs for each scenario
- `results/plot_placement_latency.png` — mean placement latency ± 95% CI
- `results/plot_a2a_overhead.png` — mean A2A overhead ± 95% CI
- `results/plot_sla_violations.png` — SLA violation rate ± 95% CI
- `results/plot_crdt_convergence.png` — CRDT convergence time ± 95% CI
- `results/plot_partition_divergence.png` — catalogue divergence during network partition (S4)
- `results/plot_nash_convergence.png` — IBR rounds per task and Nash Equilibrium convergence (S5)
- `results/summary_dashboard.png` — summary dashboard across all scenarios

---

## Experimental Scenarios

| ID | Name              | Description                                                                   |
|----|-------------------|-------------------------------------------------------------------------------|
| S1 | Baseline          | 10 tasks on 4 nodes at full utilization; measures nominal metrics             |
| S2 | High Load         | 20 tasks on 4 nodes (overload); evaluates behavior under rejection            |
| S3 | Node Failure      | Crash of edge-node-1 during execution; tests fault tolerance                  |
| S4 | Network Partition | Network partition between nodes; verifies CRDT consistency after reconnection |
| S5 | Nash Equilibrium  | 8 tasks with `NashTaskAgent`; greedy vs IBR comparison, NE verification       |

---

## Measured Metrics

- **Placement Latency (ms):** total CNP negotiation time from CFP to ACCEPT
- **A2A Overhead (ms):** portion of placement latency attributable to inter-agent communication
- **SLA Violation Rate (%):** percentage of placed tasks that exceed the declared maximum latency constraint
- **CRDT Convergence Time (ms):** time for full convergence of the distributed catalogue after one gossip round

Confidence intervals are computed using the Student t-distribution with 29 degrees of freedom (n=30 runs, α=0.05): `CI = t(29, 0.975) × s/√30`.
