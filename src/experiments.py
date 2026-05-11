"""
Esegue 4 scenari sperimentali e misura le 4 metriche del progetto:
  1. Placement Latency       — tempo totale per piazzare un task (ms)
  2. A2A Protocol Overhead   — tempo della sola fase di negoziazione (ms)
  3. SLA Violation Rate      — % task piazzati fuori dai requisiti di latenza
  4. CRDT Convergence Time   — tempo perché tutti i nodi si allineino (ms)

Scenari:
  S1 — Baseline          : carico normale, rete stabile
  S2 — High Load         : burst di 20 task simultanei, nodi saturi
  S3 — Node Failure      : un nodo va offline durante la negoziazione
  S4 — Network Partition : cluster diviso in 2 isole, poi riconnessione

Output:
  results/raw_results.json   — dati grezzi di tutti gli esperimenti
  results/plot_*.png         — un grafico per ciascuna metrica
  results/summary.png        — dashboard riepilogativa 2x2

Esecuzione:
  cd src/
  python phase4_experiments.py
"""

import ray
import time
import json
import os
import random
import copy
from typing import List, Dict, Any

import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from protocol import TaskRequirements, PlacementPolicy, MessageType, make_cfp, score_offer
from agents import ResourceAgent, TaskAgent, NashTaskAgent
from crdt_catalogue import ResourceCatalogue

# ─────────────────────────────────────────────
# Configurazione globale
# ─────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_CI_v4")
os.makedirs(RESULTS_DIR, exist_ok=True)

EDGE_NODES = [
    ("edge-node-1",  8.0, 4096,  15.0, 0.3),
    ("edge-node-2",  4.0, 2048,  40.0, 0.5),
    ("edge-node-3",  2.0, 1024,  80.0, 0.2),
    ("edge-node-4", 16.0, 8192,  25.0, 0.8),
]

# Colori per i 4 scenari (consistenti in tutti i grafici)
SCENARIO_COLORS = {
    "S1 Baseline":        "#2E86AB",
    "S2 High Load":       "#E84855",
    "S3 Node Failure":    "#F9A825",
    "S4 Net Partition":   "#43AA8B",
    "S5 Nash Equil.":     "#9B59B6",
}

SCENARIOS = list(SCENARIO_COLORS.keys())

# ─────────────────────────────────────────────
# Numero di run indipendenti per scenario
# ─────────────────────────────────────────────
N_RUNS = 30


# ─────────────────────────────────────────────
# Registry globale degli actor Ray attivi
# Ogni actor creato viene registrato qui e
# viene distrutto esplicitamente dopo ogni run
# per evitare memory leak su esperimenti lunghi.
# ─────────────────────────────────────────────
_actor_registry: list = []


def make_resource_agents():
    agents = []
    for (node_id, cpu, mem, lat, energy) in EDGE_NODES:
        a = ResourceAgent.remote(node_id, cpu, mem, lat, energy)
        agents.append(a)
        _actor_registry.append(a)
    time.sleep(0.3)
    for i, agent in enumerate(agents):
        peers = [a for j, a in enumerate(agents) if j != i]
        agent.register_peers.remote(peers)
    time.sleep(0.1)
    return agents


def kill_registered_actors():
    """
    """
    for actor in _actor_registry:
        try:
            ray.kill(actor, no_restart=True)
        except Exception:
            pass   
    _actor_registry.clear()
    # Piccola pausa per dare a Ray il tempo di liberare le risorse
    time.sleep(0.2)


def run_task_via_agent(task_id: str, cpu: float, mem: float, max_lat: float,
                       policy: PlacementPolicy, resource_agents: list) -> dict:
    req = TaskRequirements(cpu_cores=cpu, memory_mb=mem,
                           max_latency_ms=max_lat, duration_sec=10,
                           priority=2, task_type="generic")
    agent = TaskAgent.remote(task_id, req, policy)
    _actor_registry.append(agent)
    result = ray.get(agent.place.remote(resource_agents))
    # Rimuove dead_agent_indices dal result 
    result.pop("dead_agent_indices", None)
    return result


def gossip_round(agents: list) -> float:
    """
    Esegue un round di gossip CRDT e ritorna il tempo (ms).
    """
    t0 = time.time()
    catalogues = [ray.get(a.get_catalogue_object.remote()) for a in agents]
    for i, agent in enumerate(agents):
        for j, cat in enumerate(catalogues):
            if i != j:
                agent.sync_catalogue.remote(cat)
    time.sleep(0.05)
    return (time.time() - t0) * 1000


def measure_convergence_time(agents: list) -> float:
    """
    Misura il tempo necessario perché tutti i nodi convergano
    eseguendo gossip round successivi fino a convergenza completa.
    Ritorna il tempo totale in ms.
    """
    t0 = time.time()
    for _ in range(5):           # max 5 round di gossip
        gossip_round(agents)
        catalogues = [ray.get(a.get_catalogue_object.remote()) for a in agents]
        # Verifica se tutti i cataloghi sono allineati
        converged = True
        for i in range(len(catalogues)):
            for j in range(i + 1, len(catalogues)):
                if catalogues[i].convergence_diff(catalogues[j]):
                    converged = False
                    break
            if not converged:
                break
        if converged:
            break
    return (time.time() - t0) * 1000


def compute_metrics(task_results: list) -> dict:
    """Calcola le 4 metriche aggregate da una lista di risultati task."""
    placed = [r for r in task_results if r["status"] == "placed"]
    if not placed:
        return {"placement_latency_ms": 0, "a2a_overhead_ms": 0,
                "sla_violation_rate": 1.0, "n_placed": 0, "n_total": len(task_results)}

    sla_violations = sum(1 for r in placed if not r["sla_ok"])

    return {
        "placement_latency_ms": np.mean([r["placement_latency_ms"] for r in placed]),
        "placement_latency_std": np.std([r["placement_latency_ms"] for r in placed]),
        "a2a_overhead_ms":      np.mean([r["a2a_overhead_ms"] for r in placed]),
        "a2a_overhead_std":     np.std([r["a2a_overhead_ms"] for r in placed]),
        "sla_violation_rate":   sla_violations / len(placed),
        "n_placed":             len(placed),
        "n_total":              len(task_results),
    }


# ─────────────────────────────────────────────
# Multi-run: raccolta campioni e CI al 95%
# ─────────────────────────────────────────────

def ci_95(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    s = np.std(values, ddof=1)      # deviazione standard campionaria
    # Approssimazione t per quantile 0.975: valori dalla tavola t
    # n=5→2.776  n=10→2.262  n=15→2.145  n=20→2.093  n=30→2.045  n≥120→1.980
    t_table = {5: 2.776, 10: 2.262, 15: 2.145, 20: 2.093, 25: 2.064,
               30: 2.045, 40: 2.021, 60: 2.000, 120: 1.980}
    # Trova il t più conservativo (primo valore >= n)
    t_val = 1.96
    for df_key in sorted(t_table.keys()):
        if n <= df_key:
            t_val = t_table[df_key]
            break
    return float(t_val * s / np.sqrt(n))


def run_scenario_n_times(scenario_fn, scenario_name: str,
                         n_runs: int = N_RUNS) -> dict:
    """
    Ritorna un dict compatibile con i plot esistenti, con in più:
        - *_ci   : intervallo di confidenza al 95%
        - *_runs : lista dei valori per run (per scatter plot)
    """
    lat_runs   = []
    a2a_runs   = []
    sla_runs   = []
    crdt_runs  = []

    print(f"\n  ── Esecuzione {n_runs} run indipendenti per {scenario_name} ──")
    last_run = None
    for run_idx in range(1, n_runs + 1):
        try:
            m = scenario_fn()
            lat_runs.append(m["placement_latency_ms"])
            a2a_runs.append(m["a2a_overhead_ms"])
            sla_runs.append(m["sla_violation_rate"])
            crdt_runs.append(m["crdt_convergence_ms"])
            last_run = m   # teniamo l'ultima run valida per le metriche extra
            print(f"    run {run_idx:2d}/{n_runs}: "
                  f"lat={m['placement_latency_ms']:.1f}ms  "
                  f"a2a={m['a2a_overhead_ms']:.1f}ms  "
                  f"sla={m['sla_violation_rate']*100:.0f}%  "
                  f"crdt={m['crdt_convergence_ms']:.0f}ms")
        except Exception as e:
            print(f"    run {run_idx:2d}/{n_runs}: ERRORE — {e} (run scartata)")
        finally:
            # ── CLEANUP CRITICO ───────────────────────────────────────────────
            # Distrugge tutti gli actor Ray creati in questa run (ResourceAgent,
            # NashTaskAgent, ecc.)
            kill_registered_actors()

    if last_run is None:
        raise RuntimeError(f"Tutte le run di {scenario_name} sono fallite.")

    result = dict(last_run)   # eredita le chiavi "extra" dall'ultima run valida

    # Sovrascrive le 4 metriche principali con la media delle N run
    result["placement_latency_ms"]  = float(np.mean(lat_runs))
    result["placement_latency_std"] = float(np.std(lat_runs, ddof=1))
    result["placement_latency_ci"]  = ci_95(lat_runs)
    result["placement_latency_runs"] = lat_runs

    result["a2a_overhead_ms"]  = float(np.mean(a2a_runs))
    result["a2a_overhead_std"] = float(np.std(a2a_runs, ddof=1))
    result["a2a_overhead_ci"]  = ci_95(a2a_runs)
    result["a2a_overhead_runs"] = a2a_runs

    result["sla_violation_rate"]     = float(np.mean(sla_runs))
    result["sla_violation_rate_std"] = float(np.std(sla_runs, ddof=1))
    result["sla_violation_rate_ci"]  = ci_95(sla_runs)
    result["sla_violation_rate_runs"] = sla_runs

    result["crdt_convergence_ms"]     = float(np.mean(crdt_runs))
    result["crdt_convergence_ms_std"] = float(np.std(crdt_runs, ddof=1))
    result["crdt_convergence_ms_ci"]  = ci_95(crdt_runs)
    result["crdt_convergence_ms_runs"] = crdt_runs

    result["n_runs"] = len(lat_runs)  

    print(f"  ── {scenario_name}: {len(lat_runs)} run valide ──")
    print(f"     Placement latency : {result['placement_latency_ms']:.2f} ± "
          f"{result['placement_latency_ci']:.2f} ms  (95% CI)")
    print(f"     A2A overhead      : {result['a2a_overhead_ms']:.2f} ± "
          f"{result['a2a_overhead_ci']:.2f} ms  (95% CI)")
    print(f"     SLA violation rate: {result['sla_violation_rate']*100:.1f} ± "
          f"{result['sla_violation_rate_ci']*100:.1f}%  (95% CI)")
    print(f"     CRDT convergence  : {result['crdt_convergence_ms']:.1f} ± "
          f"{result['crdt_convergence_ms_ci']:.1f} ms  (95% CI)")
    return result


# ─────────────────────────────────────────────
# Scenario 1 — Baseline
# ─────────────────────────────────────────────

def scenario_baseline() -> dict:
    """
    Carico normale: 10 task con requisiti randomizzati per scenario,
    campionati da distribuzioni realistiche. La randomizzazione garantisce
    che ogni run sia un esperimento genuinamente diverso, necessario per
    ottenere intervalli di confidenza statisticamente validi.

    Ranges scelti per rappresentare un workload edge realistico:
      cpu  : U[0.5, 4.0] core
      mem  : cpu * U[128, 512] MB
      lat  : U[20, 500] ms  — da real-time a best-effort
      policy: campionata uniformemente
    """
    print("\n[S1] Baseline — carico normale, rete stabile")
    agents = make_resource_agents()

    policies = list(PlacementPolicy)
    tasks = []
    for i in range(10):
        cpu = round(random.choice([0.5, 1.0, 2.0, 4.0]), 1)
        mem = cpu * random.choice([128, 256, 512])
        lat = round(random.uniform(20.0, 500.0), 1)
        pol = random.choice(policies)
        tasks.append((f"t{i+1:02d}", cpu, mem, lat, pol))

    results = []
    for (tid, cpu, mem, lat, pol) in tasks:
        r = run_task_via_agent(tid, cpu, mem, lat, pol, agents)
        results.append(r)
        print(f"  {tid}: {r['status']} on {r['placed_on']} | "
              f"lat={r['placement_latency_ms']:.1f}ms | SLA={'OK' if r['sla_ok'] else 'VIOLATION'}")

    t_conv = measure_convergence_time(agents)
    print(f"  CRDT convergence: {t_conv:.1f}ms")

    metrics = compute_metrics(results)
    metrics["crdt_convergence_ms"] = t_conv
    metrics["task_results"] = results
    return metrics


# ─────────────────────────────────────────────
# Scenario 2 — High Load
# ─────────────────────────────────────────────

def scenario_high_load() -> dict:
    """
    Burst di 20 task: i nodi si saturano progressivamente.
    Aumentano i REJECT e le SLA violations.
    """
    print("\n[S2] High Load — 20 task in burst, nodi saturi")
    agents = make_resource_agents()

    # 20 task con requisiti variabili — alcuni "pesanti" che saturano i nodi
    tasks = []
    for i in range(20):
        cpu  = random.choice([0.5, 1.0, 2.0, 4.0])
        mem  = cpu * 256
        lat  = random.choice([20.0, 50.0, 100.0, 200.0])
        pol  = random.choice(list(PlacementPolicy))
        tasks.append((f"hl{i:02d}", cpu, mem, lat, pol))

    results = []
    for (tid, cpu, mem, lat, pol) in tasks:
        r = run_task_via_agent(tid, cpu, mem, lat, pol, agents)
        results.append(r)
        print(f"  {tid}: {r['status']:6s} | proposals={r['proposals_received']} | "
              f"SLA={'OK' if r['sla_ok'] else 'VIOLATION'}")

    t_conv = measure_convergence_time(agents)
    print(f"  CRDT convergence: {t_conv:.1f}ms")

    metrics = compute_metrics(results)
    metrics["crdt_convergence_ms"] = t_conv
    metrics["task_results"] = results
    return metrics


# ─────────────────────────────────────────────
# Scenario 3 — Node Failure
# ─────────────────────────────────────────────

def scenario_node_failure() -> dict:
    """
    S3 — Crash improvviso di edge-node-4 a metà esperimento.
    Il nodo viene terminato con ray.kill() senza nessuna notifica preventiva
    (nessun mark_offline, nessun gossip). I nodi sopravvissuti rilevano il
    crash al primo CFP senza risposta (RayActorError) e aggiornano il
    catalogo CRDT autonomamente tramite mark_node_offline_external().
    """
    print("\n[S3] Node Failure — crash improvviso di edge-node-4")
    agents = make_resource_agents()

    # Mappa agente -> node_id costruita PRIMA del crash (quando tutti sono vivi)
    agent_id_map = {a: ray.get(a.get_state.remote())["node_id"] for a in agents}

    # ── Prima metà: tutti e 4 i nodi disponibili ──────────────────
    # Task randomizzati: ogni run simula un workload diverso pre-failure
    policies = list(PlacementPolicy)
    tasks_pre = []
    for i in range(5):
        cpu = random.choice([0.5, 1.0, 2.0, 4.0])
        mem = cpu * random.choice([128, 256, 512])
        lat = round(random.uniform(20.0, 300.0), 1)
        pol = random.choice(policies)
        tasks_pre.append((f"f{i+1:02d}", cpu, mem, lat, pol))

    results = []
    print("  [PRE-FAILURE] Tutti i nodi attivi:")
    for (tid, cpu, mem, lat, pol) in tasks_pre:
        r = run_task_via_agent(tid, cpu, mem, lat, pol, agents)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r['placed_on']} | "
              f"SLA={'OK' if r['sla_ok'] else 'VIOLATION'}")

    # ── Crash improvviso: ray.kill senza nessun avviso ────────────
    print("\n  >>> CRASH IMPROVVISO — edge-node-4 terminato con ray.kill() <<<")
    print("  I peer non sono stati notificati — scopriranno il crash al prossimo CFP")
    t_failure = time.time()
    ray.kill(agents[3], no_restart=True)   # nodo morto — nessun gossip preventivo

    active_agents = list(agents)
    crash_detected = False
    t_detect = None

    tasks_post = []
    for i in range(5):
        cpu = random.choice([0.5, 1.0, 2.0, 4.0])
        mem = cpu * random.choice([128, 256, 512])
        lat = round(random.uniform(20.0, 300.0), 1)
        pol = random.choice(list(PlacementPolicy))
        tasks_post.append((f"f{i+6:02d}", cpu, mem, lat, pol))

    print("  [POST-FAILURE] TaskAgent invia CFP a tutti i nodi (crash non ancora noto)...")
    for (tid, cpu, mem, lat, pol) in tasks_post:
        # run_task_via_agent usa TaskAgent.place() che gestisce RayActorError
        # internamente e restituisce dead_agent_indices (lista di indici interi)
        req = TaskRequirements(cpu_cores=cpu, memory_mb=mem,
                               max_latency_ms=lat, duration_sec=10,
                               priority=2, task_type="generic")
        ta = TaskAgent.remote(tid, req, pol)
        _actor_registry.append(ta)
        r = ray.get(ta.place.remote(active_agents))
        r["post_failure"] = True

        dead_indices = r.pop("dead_agent_indices", [])

        # Prima rilevazione del crash tramite dead_agent_indices
        if dead_indices and not crash_detected:
            crash_detected = True
            t_detect = time.time()
            print(f"\n  [FAILURE DETECTED] Crash rilevato dal TaskAgent "
                  f"{(t_detect - t_failure)*1000:.1f}ms dopo il kill — "
                  f"aggiornamento CRDT in corso...")

            for idx in dead_indices:
                dead_agent = active_agents[idx]
                # Usa agent_id_map pre-costruita: nessuna chiamata al nodo morto
                dead_id = agent_id_map.get(dead_agent, "unknown")
                survivors = [a for a in active_agents if a is not dead_agent]
                for survivor in survivors:
                    survivor.mark_node_offline_external.remote(dead_id)
                active_agents = survivors
                print(f"  [CRDT UPDATE] {len(survivors)} nodi sopravvissuti hanno "
                      f"marcato {dead_id} offline nel catalogo CRDT\n")

        r["crash_detected"] = bool(dead_indices)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r.get('placed_on') or '—'} | "
              f"SLA={'OK' if r['sla_ok'] else 'VIOLATION'}"
              + (" | CRASH RILEVATO DAL TaskAgent" if dead_indices else ""))

    # Gossip finale tra i sopravvissuti per convergenza CRDT
    t_conv = measure_convergence_time(active_agents) if len(active_agents) > 1 else 0.0
    print(f"\n  CRDT convergence (nodi sopravvissuti): {t_conv:.1f}ms")

    pre_failure_results = results[:len(tasks_pre)]
    metrics = compute_metrics(pre_failure_results)

    # SLA violation rate su TUTTI i task (pre + post) — riflette l'impatto reale
    all_placed = [r for r in results if r["status"] == "placed"]
    all_violations = sum(1 for r in all_placed if not r["sla_ok"])
    if all_placed:
        metrics["sla_violation_rate"] = all_violations / len(all_placed)

    metrics["crdt_convergence_ms"] = t_conv
    metrics["failure_detected_ms"] = (t_detect - t_failure) * 1000 if t_detect else 0.0
    metrics["task_results"] = results
    return metrics


# ─────────────────────────────────────────────
# Scenario 4 — Network Partition
# ─────────────────────────────────────────────

def scenario_network_partition() -> dict:
    """
    Il cluster viene diviso in 2 isole (partizione):
      Isola A: edge-node-1, edge-node-2
      Isola B: edge-node-3, edge-node-4

    Durante la partizione ogni isola sincronizza solo internamente.
    I due cataloghi CRDT divergono. Dopo la riconnessione, si misura
    il tempo di convergenza (CRDT convergence time).
    """
    print("\n[S4] Network Partition — cluster diviso in 2 isole")
    agents = make_resource_agents()

    island_a = agents[:2]   # edge-node-1, edge-node-2
    island_b = agents[2:]   # edge-node-3, edge-node-4

    # ── Fase partizione: task su isole separate (randomizzati per run) ──
    # I nodi delle isole A e B hanno latenze diverse (node-1=15ms, node-2=40ms,
    # node-3=80ms, node-4=25ms), quindi il workload cambiante produce
    # variabilità reale nelle SLA violations e nella divergenza CRDT.
    policies = list(PlacementPolicy)
    tasks_a = []
    for i in range(3):
        cpu = random.choice([0.5, 1.0, 2.0])
        mem = cpu * random.choice([128, 256, 512])
        lat = round(random.uniform(30.0, 300.0), 1)
        tasks_a.append((f"pa{i+1:02d}", cpu, mem, lat, random.choice(policies)))

    tasks_b = []
    for i in range(3):
        cpu = random.choice([0.5, 1.0, 2.0])
        mem = cpu * random.choice([128, 256, 512])
        lat = round(random.uniform(60.0, 500.0), 1)  # isola B ha nodi più lenti
        tasks_b.append((f"pb{i+1:02d}", cpu, mem, lat, random.choice(policies)))

    print("  [PARTIZIONE ATTIVA]")
    print("  Isola A (node-1, node-2):")
    results = []
    for (tid, cpu, mem, lat, pol) in tasks_a:
        # Sync solo dentro isola A
        gossip_round(island_a)
        r = run_task_via_agent(tid, cpu, mem, lat, pol, island_a)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r['placed_on']}")

    print("  Isola B (node-3, node-4):")
    for (tid, cpu, mem, lat, pol) in tasks_b:
        # Sync solo dentro isola B
        gossip_round(island_b)
        r = run_task_via_agent(tid, cpu, mem, lat, pol, island_b)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r['placed_on']}")

    # ── Verifica divergenza CRDT durante la partizione ────────
    cat_a = ray.get(island_a[0].get_catalogue_object.remote())
    cat_b = ray.get(island_b[0].get_catalogue_object.remote())
    diffs_before = len(cat_a.convergence_diff(cat_b))
    print(f"\n  Divergenza CRDT durante partizione: {diffs_before} entry divergenti")

    # ── Riconnessione: gossip globale ─────────────────────────
    print("  >>> Partizione risolta — gossip globale <<<")
    t_reconnect = time.time()
    t_conv = measure_convergence_time(agents)
    print(f"  CRDT convergence time: {t_conv:.1f}ms")

    # Verifica convergenza
    cat_a_post = ray.get(island_a[0].get_catalogue_object.remote())
    cat_b_post = ray.get(island_b[0].get_catalogue_object.remote())
    diffs_after = len(cat_a_post.convergence_diff(cat_b_post))
    print(f"  Divergenza CRDT dopo gossip: {diffs_after} entry divergenti "
          f"({'CONVERGED' if diffs_after == 0 else 'STILL DIVERGING'})")

    # ── Task post-riconnessione (cluster completo, randomizzati) ──────────
    print("  [POST-RICONNESSIONE] Cluster completo:")
    tasks_post = []
    for i in range(2):
        cpu = random.choice([0.5, 1.0, 2.0])
        mem = cpu * random.choice([128, 256, 512])
        lat = round(random.uniform(30.0, 200.0), 1)
        tasks_post.append((f"pc{i+1:02d}", cpu, mem, lat, random.choice(list(PlacementPolicy))))
    for (tid, cpu, mem, lat, pol) in tasks_post:
        r = run_task_via_agent(tid, cpu, mem, lat, pol, agents)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r['placed_on']}")

    metrics = compute_metrics(results)
    metrics["crdt_convergence_ms"] = t_conv
    metrics["diffs_during_partition"] = diffs_before
    metrics["diffs_after_reconnect"] = diffs_after
    metrics["task_results"] = results
    return metrics


# ─────────────────────────────────────────────
# Scenario 5 — Nash Equilibrium (Greedy vs IBR)
# ─────────────────────────────────────────────

def scenario_s5_nash() -> dict:
    """
    Confronto diretto tra TaskAgent greedy (singolo round) e NashTaskAgent
    (Iterative Best Response) su task con requisiti di latenza inizialmente
    molto stringenti.

    Il TaskAgent greedy fallisce o viola SLA quando i nodi non riescono a
    soddisfare i requisiti. Il NashTaskAgent negozia in piu' round rilassando
    progressivamente i vincoli finche' tutte le 4 condizioni di Nash Equilibrium
    sono soddisfatte, garantendo un'allocazione stabile.

    Metriche aggiuntive rispetto agli altri scenari:
      - nash_rounds_to_convergence : quanti round ha impiegato ogni task
      - nash_winner_utility         : utilita' del nodo vincitore in [0,1]
      - confronto greedy vs nash    : success rate e SLA violation rate
    """
    print("\n[S5] Nash Equilibrium — Greedy vs. Iterative Best Response")


    tasks = []
    cpu_choices = [0.5, 1.0, 2.0, 3.0, 4.0]
    for i in range(8):
        cpu = random.choice(cpu_choices)
        mem = cpu * random.choice([128, 256, 384])
        # Latenze intenzionalmente stringenti: 60% sotto 15ms (spesso impossibile greedy)
        lat = round(random.uniform(8.0, 28.0), 1)
        pol = random.choice([PlacementPolicy.LATENCY_FIRST, PlacementPolicy.BALANCED,
                             PlacementPolicy.ENERGY_FIRST])
        tasks.append((f"n{i+1:02d}", cpu, mem, lat, pol))

    # ── Run GREEDY (TaskAgent standard — singolo round) ──────────────────────
    print("\n  [GREEDY — singolo round, nessuna negoziazione]")
    greedy_agents  = make_resource_agents()
    greedy_results = []
    for (tid, cpu, mem, lat, pol) in tasks:
        r = run_task_via_agent(tid, cpu, mem, lat, pol, greedy_agents)
        greedy_results.append(r)
        lat_str = f"{r['estimated_latency_ms']:.1f}ms" if r["estimated_latency_ms"] else "N/A"
        print(f"    {tid}: {r['status']:6s} | proposals={r['proposals_received']} | "
              f"lat={lat_str} | SLA={'OK' if r['sla_ok'] else 'FAIL'}")

    # ── Run NASH (NashTaskAgent — Iterative Best Response) ───────────────────
    print("\n  [NASH IBR — multi-round, rilassamento progressivo]")
    nash_agents  = make_resource_agents()
    nash_results = []
    for (tid, cpu, mem, lat, pol) in tasks:
        req   = TaskRequirements(cpu_cores=cpu, memory_mb=mem,
                                 max_latency_ms=lat, duration_sec=10,
                                 priority=2, task_type="generic")
        nagent = NashTaskAgent.remote(tid, req, pol,
                                      max_rounds=5, relaxation_factor=0.20)
        _actor_registry.append(nagent)   # registra per il cleanup post-run
        r = ray.get(nagent.place_nash.remote(nash_agents))
        nash_results.append(r)
        converged = r.get("nash_converged", False)
        rounds    = r.get("nash_rounds", "?")
        lat_str   = (f"{r['estimated_latency_ms']:.1f}ms"
                     if r.get("estimated_latency_ms") else "N/A")
        print(f"    {tid}: {r.get('status','?'):6s} | rounds={rounds} | "
              f"Nash={'OK' if converged else 'fallback'} | lat={lat_str}")

    # ── Metriche comparative ──────────────────────────────────────────────────
    greedy_placed = [r for r in greedy_results if r["status"] == "placed"]
    nash_placed   = [r for r in nash_results   if r.get("status") == "placed"]

    greedy_sla_viol = sum(1 for r in greedy_placed if not r["sla_ok"])
    nash_sla_viol   = sum(1 for r in nash_placed
                          if not r.get("sla_ok_original", True))

    rounds_list  = [r.get("nash_rounds", 0) for r in nash_results
                    if r.get("status") == "placed"]
    mean_rounds  = float(np.mean(rounds_list))  if rounds_list  else 0.0
    utilities    = [r.get("nash_winner_utility", 0) for r in nash_placed]
    mean_utility = float(np.mean(utilities)) if utilities else 0.0

    t_conv = measure_convergence_time(nash_agents)

    # Metriche compatibili con il summary table
    nash_latencies = [r.get("placement_latency_ms", 0)
                      for r in nash_results if r.get("status") == "placed"]

    print(f"\n  Greedy: {len(greedy_placed)}/{len(tasks)} piazzati, "
          f"{greedy_sla_viol} SLA violations")
    print(f"  Nash  : {len(nash_placed)}/{len(tasks)} piazzati, "
          f"{nash_sla_viol} SLA violations (su req. originali), "
          f"rounds medi={mean_rounds:.1f}, utility={mean_utility:.3f}")
    print(f"  CRDT convergence: {t_conv:.1f}ms")

    return {
        # Metriche Nash per summary table
        "placement_latency_ms":  float(np.mean(nash_latencies)) if nash_latencies else 0.0,
        "placement_latency_std": float(np.std(nash_latencies))  if nash_latencies else 0.0,
        "a2a_overhead_ms":       0.0,   # inglobato nei round multipli
        "a2a_overhead_std":      0.0,
        "sla_violation_rate":    nash_sla_viol / max(len(nash_placed), 1),
        "n_placed":              len(nash_placed),
        "n_total":               len(tasks),
        "crdt_convergence_ms":   t_conv,
        # Metriche specifiche S5
        "greedy_n_placed":       len(greedy_placed),
        "greedy_n_failed":       len(tasks) - len(greedy_placed),
        "greedy_sla_violations": greedy_sla_viol,
        "greedy_success_rate":   len(greedy_placed) / len(tasks),
        "nash_n_placed":         len(nash_placed),
        "nash_n_failed":         len(tasks) - len(nash_placed),
        "nash_sla_violations":   nash_sla_viol,
        "nash_success_rate":     len(nash_placed) / len(tasks),
        "nash_mean_rounds":      mean_rounds,
        "nash_mean_utility":     mean_utility,
        "task_results_greedy":   greedy_results,
        "task_results_nash":     nash_results,
        "task_labels":           [t[0] for t in tasks],
    }




def plot_runs_scatter(all_metrics: dict):

    metrics_info = [
        ("placement_latency_runs",   "placement_latency_ms",   "Placement Latency (ms)",     "placement_latency_ci"),
        ("a2a_overhead_runs",        "a2a_overhead_ms",        "A2A Overhead (ms)",           "a2a_overhead_ci"),
        ("sla_violation_rate_runs",  "sla_violation_rate",     "SLA Violation Rate",          "sla_violation_rate_ci"),
        ("crdt_convergence_ms_runs", "crdt_convergence_ms",    "CRDT Convergence (ms)",       "crdt_convergence_ms_ci"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Distribuzione delle {N_RUNS} run indipendenti per scenario",
                 fontsize=14, fontweight="bold")

    scenarios = list(all_metrics.keys())
    for ax, (runs_key, mean_key, ylabel, ci_key) in zip(axes.flat, metrics_info):
        for i, s in enumerate(scenarios):
            runs = all_metrics[s].get(runs_key, [])
            mean_val = all_metrics[s].get(mean_key, 0)
            ci_val   = all_metrics[s].get(ci_key, 0)
            color    = SCENARIO_COLORS[s]
            jitter   = np.random.normal(0, 0.08, len(runs))
            ax.scatter([i + j for j in jitter], runs, color=color,
                       alpha=0.5, s=25, zorder=3)
            ax.errorbar(i, mean_val, yerr=ci_val, fmt="D",
                        color=color, markeredgecolor="black",
                        capsize=6, linewidth=2, markersize=7, zorder=4)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, fontsize=7, rotation=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontweight="bold", fontsize=11)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_runs_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")


def plot_placement_latency(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(all_metrics.keys())
    means = [all_metrics[s]["placement_latency_ms"] for s in scenarios]
    # Usa CI al 95% se disponibile, altrimenti std
    errs  = [all_metrics[s].get("placement_latency_ci",
             all_metrics[s].get("placement_latency_std", 0)) for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, means, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(scenarios, means, yerr=errs, fmt="none",
                color="black", capsize=5, linewidth=1.5, zorder=4,
                label="95% CI")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    n_runs = next(iter(all_metrics.values())).get("n_runs", 1)
    ax.set_ylabel("Placement Latency (ms)", fontsize=12)
    ax.set_title(f"Placement Latency per Scenario  (n={n_runs} run, barre = 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_placement_latency.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvato: {path}")


def plot_a2a_overhead(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(all_metrics.keys())
    means = [all_metrics[s]["a2a_overhead_ms"] for s in scenarios]
    errs  = [all_metrics[s].get("a2a_overhead_ci",
             all_metrics[s].get("a2a_overhead_std", 0)) for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, means, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(scenarios, means, yerr=errs, fmt="none",
                color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    n_runs = next(iter(all_metrics.values())).get("n_runs", 1)
    ax.set_ylabel("A2A Overhead (ms)", fontsize=12)
    ax.set_title(f"A2A Protocol Overhead per Scenario  (n={n_runs} run, barre = 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_a2a_overhead.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvato: {path}")


def plot_sla_violations(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(all_metrics.keys())
    rates = [all_metrics[s]["sla_violation_rate"] * 100 for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, rates, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Linea soglia 10%
    ax.axhline(y=10, color="red", linestyle="--", linewidth=1.5,
               label="Soglia SLA (10%)", zorder=4)
    ax.legend(fontsize=10)

    n_runs = next(iter(all_metrics.values())).get("n_runs", 1)
    ax.set_ylabel("SLA Violation Rate (%)", fontsize=12)
    ax.set_title(f"SLA Violation Rate per Scenario  (n={n_runs} run)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(max(rates) * 1.3, 15))
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_sla_violations.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvato: {path}")


def plot_crdt_convergence(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(all_metrics.keys())
    times = [all_metrics[s]["crdt_convergence_ms"] for s in scenarios]
    errs  = [all_metrics[s].get("crdt_convergence_ms_ci", 0) for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, times, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(scenarios, times, yerr=errs, fmt="none",
                color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    n_runs = next(iter(all_metrics.values())).get("n_runs", 1)
    ax.set_ylabel("Convergence Time (ms)", fontsize=12)
    ax.set_title(f"CRDT Convergence Time per Scenario  (n={n_runs} run, barre = 95% CI)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(times) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_crdt_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvato: {path}")


def plot_summary_dashboard(all_metrics: dict):
    """
    Dashboard 2x2 con tutte e 4 le metriche in un'unica figura.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Fase 4 — Riepilogo Metriche per Scenario",
                 fontsize=16, fontweight="bold", y=1.01)

    scenarios = list(all_metrics.keys())
    colors    = [SCENARIO_COLORS[s] for s in scenarios]
    x         = np.arange(len(scenarios))
    bar_w     = 0.5

    n_runs = next(iter(all_metrics.values())).get("n_runs", 1)
    fig.suptitle(f"Fase 4 — Riepilogo Metriche per Scenario  "
                 f"(n={n_runs} run indipendenti, barre = 95% CI)",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── (0,0) Placement Latency ──────────────────────────────
    ax = axes[0][0]
    vals = [all_metrics[s]["placement_latency_ms"] for s in scenarios]
    errs = [all_metrics[s].get("placement_latency_ci",
            all_metrics[s].get("placement_latency_std", 0)) for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("Placement Latency (ms) ± 95% CI", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals)*0.02, f"{v:.1f}", ha="center", fontsize=8)

    # ── (0,1) A2A Overhead ───────────────────────────────────
    ax = axes[0][1]
    vals = [all_metrics[s]["a2a_overhead_ms"] for s in scenarios]
    errs = [all_metrics[s].get("a2a_overhead_ci",
            all_metrics[s].get("a2a_overhead_std", 0)) for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("A2A Protocol Overhead (ms) ± 95% CI", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals)*0.02, f"{v:.1f}", ha="center", fontsize=8)

    # ── (1,0) SLA Violation Rate ─────────────────────────────
    ax = axes[1][0]
    vals = [all_metrics[s]["sla_violation_rate"] * 100 for s in scenarios]
    errs = [all_metrics[s].get("sla_violation_rate_ci", 0) * 100 for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4, zorder=4)
    ax.axhline(y=10, color="red", linestyle="--", linewidth=1.3,
               label="Soglia 10%", zorder=4)
    ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("SLA Violation Rate (%) ± 95% CI", fontweight="bold")
    ax.set_ylim(0, max(max(vals) * 1.35, 15))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    # ── (1,1) CRDT Convergence ───────────────────────────────
    ax = axes[1][1]
    vals = [all_metrics[s]["crdt_convergence_ms"] for s in scenarios]
    errs = [all_metrics[s].get("crdt_convergence_ms_ci", 0) for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", color="black", capsize=4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("CRDT Convergence Time (ms) ± 95% CI", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals)*0.02, f"{v:.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")


def plot_partition_crdt_divergence(s4_metrics: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Durante la partizione", "Dopo il gossip"]
    vals   = [
        s4_metrics.get("diffs_during_partition", 0),
        s4_metrics.get("diffs_after_reconnect", 0),
    ]
    colors = ["#E84855", "#43AA8B"]
    bars = ax.bar(labels, vals, color=colors, width=0.4,
                  edgecolor="white", linewidth=1.2, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                str(v), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Entry divergenti nel catalogo CRDT", fontsize=11)
    ax.set_title("S4 — Divergenza CRDT: Partizione vs. Riconnessione",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.5 + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_partition_divergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Salvato: {path}")


def plot_nash_convergence(s5_metrics: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("S5 — Nash Equilibrium: Greedy vs. Iterative Best Response",
                 fontsize=14, fontweight="bold")

    nash_results = s5_metrics["task_results_nash"]
    task_labels  = s5_metrics["task_labels"]

    # ── (a) Rounds to Nash Equilibrium ───────────────────────────────────────
    rounds       = []
    colors_rounds = []
    for r in nash_results:
        rds    = r.get("nash_rounds", 0)
        status = r.get("status", "failed")
        if status != "placed":
            rounds.append(0)
            colors_rounds.append("#888888")          # grigio = fallito
        elif r.get("nash_converged", False):
            rounds.append(rds)
            if rds == 1:
                colors_rounds.append("#43AA8B")      # verde  = NE immediato
            elif rds == 2:
                colors_rounds.append("#F9A825")      # giallo = 2 round
            else:
                colors_rounds.append("#E84855")      # rosso  = 3+ round
        else:
            rounds.append(rds)
            colors_rounds.append("#9B59B6")          # viola  = fallback

    bars1 = ax1.bar(task_labels, rounds, color=colors_rounds,
                    edgecolor="white", linewidth=1.2, zorder=3)
    for bar, val, r in zip(bars1, rounds, nash_results):
        label = str(val) if r.get("status") == "placed" else "X"
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.05, label,
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    legend_patches = [
        mpatches.Patch(color="#43AA8B", label="NE @ round 1 (equilibrio immediato)"),
        mpatches.Patch(color="#F9A825", label="NE @ round 2"),
        mpatches.Patch(color="#E84855", label="NE @ round 3+"),
        mpatches.Patch(color="#9B59B6", label="Fallback (max rounds esaurito)"),
        mpatches.Patch(color="#888888", label="Fallito (0 proposte)"),
    ]
    ax1.legend(handles=legend_patches, fontsize=7.5, loc="upper right")
    ax1.set_xlabel("Task", fontsize=11)
    ax1.set_ylabel("Round di negoziazione", fontsize=11)
    ax1.set_title("(a) Rounds to Nash Equilibrium per Task",
                  fontsize=12, fontweight="bold")
    ax1.set_ylim(0, max(rounds + [1]) * 1.5 + 1)
    ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax1.set_axisbelow(True)

    # ── (b) Greedy vs Nash comparison ────────────────────────────────────────
    n_tasks         = len(nash_results)
    metrics_labels  = ["Task piazzati", "Fallimenti", "Violazioni SLA*"]
    greedy_vals = [
        s5_metrics["greedy_n_placed"],
        s5_metrics["greedy_n_failed"],
        s5_metrics["greedy_sla_violations"],
    ]
    nash_vals = [
        s5_metrics["nash_n_placed"],
        s5_metrics["nash_n_failed"],
        s5_metrics["nash_sla_violations"],
    ]

    x     = np.arange(len(metrics_labels))
    width = 0.35
    bars_g = ax2.bar(x - width / 2, greedy_vals, width,
                     label="Greedy (1 round)",
                     color="#2E86AB", edgecolor="white", zorder=3)
    bars_n = ax2.bar(x + width / 2, nash_vals,   width,
                     label="Nash IBR (multi-round)",
                     color="#43AA8B", edgecolor="white", zorder=3)

    for bar in list(bars_g) + list(bars_n):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 h + 0.05, str(int(h)),
                 ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics_labels, fontsize=10)
    ax2.set_ylabel("Numero di task", fontsize=11)
    ax2.set_title(f"(b) Greedy vs Nash — {n_tasks} task con SLA stringenti",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(0, max(max(greedy_vals), max(nash_vals), 1) * 1.5 + 1)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)

    # Annotazione riepilogativa
    ax2.text(
        0.98, 0.97,
        f"Rounds medi Nash : {s5_metrics['nash_mean_rounds']:.1f}\n"
        f"Utility media    : {s5_metrics['nash_mean_utility']:.3f}\n"
        f"* SLA calcolate sui requisiti originali",
        transform=ax2.transAxes, fontsize=8.5,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.85),
    )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "plot_nash_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Salvato: {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def sep(c="═", w=65): print(c * w)

def main():
    sep()
    print(f"  FASE 4 — Experimentation & Evaluation  [{N_RUNS} run per scenario]")
    sep()
    ray.init(ignore_reinit_error=True)

    all_metrics: Dict[str, Any] = {}

    sep("─")
    print("SCENARIO 1 — Baseline"); sep("─")
    all_metrics["S1 Baseline"] = run_scenario_n_times(
        scenario_baseline, "S1 Baseline", N_RUNS)

    sep("─")
    print("SCENARIO 2 — High Load"); sep("─")
    all_metrics["S2 High Load"] = run_scenario_n_times(
        scenario_high_load, "S2 High Load", N_RUNS)

    sep("─")
    print("SCENARIO 3 — Node Failure"); sep("─")
    all_metrics["S3 Node Failure"] = run_scenario_n_times(
        scenario_node_failure, "S3 Node Failure", N_RUNS)

    sep("─")
    print("SCENARIO 4 — Network Partition"); sep("─")
    all_metrics["S4 Net Partition"] = run_scenario_n_times(
        scenario_network_partition, "S4 Net Partition", N_RUNS)

    sep("─")
    print("SCENARIO 5 — Nash Equilibrium (Greedy vs IBR)"); sep("─")
    all_metrics["S5 Nash Equil."] = run_scenario_n_times(
        scenario_s5_nash, "S5 Nash Equil.", N_RUNS)

    # ── Riepilogo testuale con CI ────────────────────────────────
    sep()
    print("  RIEPILOGO METRICHE  (media ± 95% CI su run indipendenti)")
    sep()
    print(f"{'Scenario':<22} {'PlacLat ms':<20} {'A2A ms':<18} "
          f"{'SLA viol%':<18} {'CRDT ms'}")
    print(f"{'':22} {'mean ± CI':<20} {'mean ± CI':<18} "
          f"{'mean ± CI':<18} {'mean ± CI'}")
    sep("─")
    for s, m in all_metrics.items():
        lat_ci  = m.get("placement_latency_ci", 0)
        a2a_ci  = m.get("a2a_overhead_ci", 0)
        sla_ci  = m.get("sla_violation_rate_ci", 0) * 100
        crdt_ci = m.get("crdt_convergence_ms_ci", 0)
        print(
            f"{s:<22} "
            f"{m['placement_latency_ms']:.1f} ± {lat_ci:.2f} ms".ljust(20) +
            f"  {m['a2a_overhead_ms']:.1f} ± {a2a_ci:.2f}".ljust(18) +
            f"  {m['sla_violation_rate']*100:.1f} ± {sla_ci:.1f}%".ljust(18) +
            f"  {m['crdt_convergence_ms']:.1f} ± {crdt_ci:.1f}"
        )

    # ── Salva dati grezzi ────────────────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "raw_results.json")

    def to_serializable(obj):
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(i) for i in obj]
        return obj

    with open(raw_path, "w") as f:
        json.dump(to_serializable(all_metrics), f, indent=2)
    print(f"\n  Dati grezzi salvati: {raw_path}")

    # ── Genera grafici ───────────────────────────────────────────
    sep()
    print("  GENERAZIONE GRAFICI")
    sep("─")
    plot_placement_latency(all_metrics)
    plot_a2a_overhead(all_metrics)
    plot_sla_violations(all_metrics)
    plot_crdt_convergence(all_metrics)
    plot_summary_dashboard(all_metrics)
    plot_runs_scatter(all_metrics)
    plot_partition_crdt_divergence(all_metrics["S4 Net Partition"])
    plot_nash_convergence(all_metrics["S5 Nash Equil."])

    sep()
    print(f"  FASE 4 COMPLETATA  ({N_RUNS} run per scenario)")
    print(f"  Grafici in: {RESULTS_DIR}/")
    sep()

    ray.shutdown()


if __name__ == "__main__":
    main()