"""
phase4_experiments.py
=====================
Fase 4 — Experimentation & Evaluation

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
matplotlib.use("Agg")          # backend non-interattivo (funziona senza display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from protocol import TaskRequirements, PlacementPolicy, MessageType, make_cfp, score_offer
from agents import ResourceAgent, TaskAgent, NashTaskAgent
from crdt_catalogue import ResourceCatalogue

# ─────────────────────────────────────────────
# Configurazione globale
# ─────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

EDGE_NODES = [
    ("edge-node-1",  8.0, 4096,  15.0, 0.3),
    ("edge-node-2",  4.0, 2048,  40.0, 0.5),
    ("edge-node-3",  2.0, 1024,  80.0, 0.2),
    ("edge-node-4", 16.0, 8192,  25.0, 0.8),
]


SCENARIO_COLORS = {
    "S1 Baseline":        "#2E86AB",
    "S2 High Load":       "#E84855",
    "S3 Node ffffffffffailure":    "#F9A825",
    "S4 Net Partition":   "#43AA8B",
    "S5 Nash Equil.":     "#9B59B6",
}

SCENARIOS = list(SCENARIO_COLORS.keys())


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_resource_agents():
    """Crea e registra 4 ResourceAgent freschi."""
    agents = []
    for (node_id, cpu, mem, lat, energy) in EDGE_NODES:
        a = ResourceAgent.remote(node_id, cpu, mem, lat, energy)
        agents.append(a)
    time.sleep(0.3)
    for i, agent in enumerate(agents):
        peers = [a for j, a in enumerate(agents) if j != i]
        agent.register_peers.remote(peers)
    time.sleep(0.1)
    return agents


def run_task(task_id: str, cpu: float, mem: float, max_lat: float,
             policy: PlacementPolicy, resource_agents: list) -> dict:
    """
    Esegue la negoziazione per un singolo task e ritorna il risultato
    arricchito con timing granulare per le metriche Fase 4.
    """
    req = TaskRequirements(cpu_cores=cpu, memory_mb=mem,
                           max_latency_ms=max_lat, duration_sec=10,
                           priority=2, task_type="generic")

    # ── misura A2A overhead: solo fase broadcast + raccolta risposte ──
    cfp = make_cfp(f"task-{task_id}", task_id, req)
    t_cfp = time.time()
    response_refs = [a.receive_cfp.remote(cfp) for a in resource_agents]
    responses = ray.get(response_refs)
    t_responses = time.time()
    a2a_overhead_ms = (t_responses - t_cfp) * 1000

    # ── fase scoring + accept (placement decision) ──
    t_placement_start = time.time()
    proposals = []
    for resp in responses:
        if resp and resp.msg_type in (MessageType.PROPOSE, MessageType.COUNTER_OFFER):
            resp.offer.score = score_offer(resp.offer, policy)
            proposals.append(resp)

    status = "failed"
    placed_on = None
    estimated_latency_ms = None
    sla_ok = False

    if proposals:
        best = max(proposals, key=lambda r: r.offer.score)
        winner_idx = next(
            i for i, a in enumerate(resource_agents)
            if ray.get(a.get_state.remote())["node_id"] == best.sender_id
        )
        from protocol import make_accept, make_reject
        accept_msg = make_accept(f"task-{task_id}", best.sender_id,
                                 task_id, cfp.conversation_id)
        ray.get(resource_agents[winner_idx].receive_accept.remote(accept_msg))

        # Reject agli altri
        for resp in proposals:
            if resp.sender_id != best.sender_id:
                loser_idx = next(
                    i for i, a in enumerate(resource_agents)
                    if ray.get(a.get_state.remote())["node_id"] == resp.sender_id
                )
                resource_agents[loser_idx].receive_reject.remote(
                    make_reject(f"task-{task_id}", resp.sender_id,
                                task_id, cfp.conversation_id, "better_offer")
                )

        status = "placed"
        placed_on = best.sender_id
        estimated_latency_ms = best.offer.estimated_latency_ms
        sla_ok = estimated_latency_ms <= max_lat

    t_placement_end = time.time()
    placement_latency_ms = (t_placement_end - t_placement_start) * 1000 + a2a_overhead_ms

    return {
        "task_id":              task_id,
        "status":               status,
        "placed_on":            placed_on,
        "placement_latency_ms": placement_latency_ms,
        "a2a_overhead_ms":      a2a_overhead_ms,
        "estimated_latency_ms": estimated_latency_ms,
        "max_latency_ms":       max_lat,
        "sla_ok":               sla_ok,
        "proposals_received":   len(proposals),
    }


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
# Scenario 1 — Baseline
# ─────────────────────────────────────────────

def scenario_baseline() -> dict:
    """
    Carico normale: 10 task con requisiti misti, rete stabile.
    Serve come riferimento per confrontare gli altri scenari.
    """
    print("\n[S1] Baseline — carico normale, rete stabile")
    agents = make_resource_agents()

    tasks = [
        ("t01", 1.0,  256,  50.0, PlacementPolicy.LATENCY_FIRST),
        ("t02", 2.0,  512, 100.0, PlacementPolicy.ENERGY_FIRST),
        ("t03", 0.5,  128, 200.0, PlacementPolicy.BALANCED),
        ("t04", 4.0, 1024,  30.0, PlacementPolicy.LATENCY_FIRST),
        ("t05", 1.0,  256, 500.0, PlacementPolicy.ENERGY_FIRST),
        ("t06", 2.0,  512,  60.0, PlacementPolicy.BALANCED),
        ("t07", 1.0,  256, 150.0, PlacementPolicy.BALANCED),
        ("t08", 0.5,  128,  20.0, PlacementPolicy.LATENCY_FIRST),
        ("t09", 2.0,  512, 200.0, PlacementPolicy.ENERGY_FIRST),
        ("t10", 1.0,  256, 100.0, PlacementPolicy.BALANCED),
    ]

    results = []
    for (tid, cpu, mem, lat, pol) in tasks:
        r = run_task(tid, cpu, mem, lat, pol, agents)
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
        r = run_task(tid, cpu, mem, lat, pol, agents)
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

def run_task_resilient(task_id: str, cpu: float, mem: float, max_lat: float,
                       policy: PlacementPolicy, resource_agents: list) -> dict:
    """
    Come run_task, ma gestisce RayActorError per singoli agenti morti.
    Restituisce il risultato normale più 'dead_nodes': lista degli id dei
    nodi che non hanno risposto (crash rilevato via eccezione Ray).
    """
    req = TaskRequirements(cpu_cores=cpu, memory_mb=mem,
                           max_latency_ms=max_lat, duration_sec=10,
                           priority=2, task_type="generic")
    cfp = make_cfp(f"task-{task_id}", task_id, req)

    # Manda CFP a tutti — incluso eventuale nodo morto
    t_cfp = time.time()
    refs = [(a, a.receive_cfp.remote(cfp)) for a in resource_agents]

    responses = []
    dead_agents = []   # oggetti agente (non stringhe) — node_id risolto dal chiamante
    for agent, ref in refs:
        try:
            resp = ray.get(ref)          # RayActorError immediato se l'actor è morto
            responses.append(resp)
        except ray.exceptions.RayActorError:
            dead_agents.append(agent)   # salva l'oggetto, NON chiamare get_state sul morto
            print(f"  [CRASH DETECTED] Un agente non risponde — RayActorError ricevuto")

    t_responses = time.time()
    a2a_overhead_ms = (t_responses - t_cfp) * 1000

    # Fase scoring + accept (identica a run_task)
    t_placement_start = time.time()
    proposals = []
    for resp in responses:
        if resp and resp.msg_type in (MessageType.PROPOSE, MessageType.COUNTER_OFFER):
            resp.offer.score = score_offer(resp.offer, policy)
            proposals.append(resp)

    status = "failed"
    placed_on = None
    estimated_latency_ms = None
    sla_ok = False

    if proposals:
        best = max(proposals, key=lambda r: r.offer.score)
        winner_idx = next(
            i for i, a in enumerate(resource_agents)
            if ray.get(a.get_state.remote())["node_id"] == best.sender_id
        )
        from protocol import make_accept, make_reject
        accept_msg = make_accept(f"task-{task_id}", best.sender_id,
                                 task_id, cfp.conversation_id)
        ray.get(resource_agents[winner_idx].receive_accept.remote(accept_msg))

        for resp in proposals:
            if resp.sender_id != best.sender_id:
                loser_idx = next(
                    i for i, a in enumerate(resource_agents)
                    if ray.get(a.get_state.remote())["node_id"] == resp.sender_id
                )
                resource_agents[loser_idx].receive_reject.remote(
                    make_reject(f"task-{task_id}", resp.sender_id,
                                task_id, cfp.conversation_id, "better_offer")
                )

        status = "placed"
        placed_on = best.sender_id
        estimated_latency_ms = best.offer.estimated_latency_ms
        sla_ok = estimated_latency_ms <= max_lat

    t_placement_end = time.time()
    placement_latency_ms = (t_placement_end - t_placement_start) * 1000 + a2a_overhead_ms

    return {
        "task_id":              task_id,
        "status":               status,
        "placed_on":            placed_on,
        "placement_latency_ms": placement_latency_ms,
        "a2a_overhead_ms":      a2a_overhead_ms,
        "estimated_latency_ms": estimated_latency_ms,
        "max_latency_ms":       max_lat,
        "sla_ok":               sla_ok,
        "proposals_received":   len(proposals),
        "dead_agents":          dead_agents,   # oggetti agente, non stringhe
    }


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
    tasks_pre = [
        ("f01", 1.0,  256,  50.0, PlacementPolicy.LATENCY_FIRST),
        ("f02", 2.0,  512, 100.0, PlacementPolicy.BALANCED),
        ("f03", 4.0, 1024,  30.0, PlacementPolicy.LATENCY_FIRST),
        ("f04", 1.0,  256, 200.0, PlacementPolicy.ENERGY_FIRST),
        ("f05", 0.5,  128,  20.0, PlacementPolicy.LATENCY_FIRST),
    ]

    results = []
    print("  [PRE-FAILURE] Tutti i nodi attivi:")
    for (tid, cpu, mem, lat, pol) in tasks_pre:
        r = run_task(tid, cpu, mem, lat, pol, agents)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r['placed_on']} | "
              f"SLA={'OK' if r['sla_ok'] else 'VIOLATION'}")

    # ── Crash improvviso: ray.kill senza nessun avviso ────────────
    print("\n  >>> CRASH IMPROVVISO — edge-node-4 terminato con ray.kill() <<<")
    print("  I peer non sono stati notificati — scopriranno il crash al prossimo CFP")
    t_failure = time.time()
    ray.kill(agents[3], no_restart=True)   # nodo morto — nessun gossip preventivo

    # I peer non sanno ancora che il nodo è morto.
    # Mandiamo i CFP a tutti e 4, incluso il nodo morto.
    active_agents = list(agents)
    crash_detected = False
    t_detect = None

    tasks_post = [
        ("f06", 1.0,  256,  50.0, PlacementPolicy.LATENCY_FIRST),
        ("f07", 2.0,  512, 100.0, PlacementPolicy.BALANCED),
        ("f08", 4.0, 1024,  30.0, PlacementPolicy.LATENCY_FIRST),
        ("f09", 1.0,  256, 200.0, PlacementPolicy.ENERGY_FIRST),
        ("f10", 0.5,  128,  20.0, PlacementPolicy.LATENCY_FIRST),
    ]

    print("  [POST-FAILURE] Invio CFP a tutti i nodi (crash non ancora rilevato)...")
    for (tid, cpu, mem, lat, pol) in tasks_post:
        r =     run_task_resilient(tid, cpu, mem, lat, pol, active_agents)
        r["post_failure"] = True

        # Prima rilevazione del crash
        if r["dead_agents"] and not crash_detected:
            crash_detected = True
            t_detect = time.time()
            print(f"\n  [FAILURE DETECTED] Crash rilevato {(t_detect - t_failure)*1000:.1f}ms "
                  f"dopo il kill — aggiornamento CRDT in corso...")

            for dead_agent in r["dead_agents"]:
                # Usa la mappa pre-costruita: nessuna chiamata al nodo morto
                dead_id = agent_id_map.get(dead_agent, "unknown")
                survivors = [a for a in active_agents if a is not dead_agent]
                # Ogni nodo sopravvissuto aggiorna il proprio catalogo CRDT
                for survivor in survivors:
                    survivor.mark_node_offline_external.remote(dead_id)
                active_agents = survivors
                print(f"  [CRDT UPDATE] {len(survivors)} nodi sopravvissuti hanno "
                      f"marcato {dead_id} offline nel catalogo CRDT\n")

        # Sostituisci ActorHandle (non serializzabile JSON) con booleano
        crash_this_task = bool(r["dead_agents"])
        r["crash_detected"] = crash_this_task
        del r["dead_agents"]

        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r.get('placed_on') or '—'} | "
              f"SLA={'OK' if r['sla_ok'] else 'VIOLATION'}"
              + (" | CRASH RILEVATO" if crash_this_task else ""))

    # Gossip finale tra i sopravvissuti per convergenza CRDT
    t_conv = measure_convergence_time(active_agents) if len(active_agents) > 1 else 0.0
    print(f"\n  CRDT convergence (nodi sopravvissuti): {t_conv:.1f}ms")

    metrics = compute_metrics(results)
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

    # ── Fase partizione: task su isole separate ──────────────
    tasks_a = [
        ("pa01", 1.0, 256,  50.0, PlacementPolicy.LATENCY_FIRST),
        ("pa02", 2.0, 512, 100.0, PlacementPolicy.BALANCED),
        ("pa03", 1.0, 256, 200.0, PlacementPolicy.ENERGY_FIRST),
    ]
    tasks_b = [
        ("pb01", 1.0, 256,  80.0, PlacementPolicy.LATENCY_FIRST),
        ("pb02", 2.0, 512, 150.0, PlacementPolicy.BALANCED),
        ("pb03", 0.5, 128, 500.0, PlacementPolicy.ENERGY_FIRST),
    ]

    print("  [PARTIZIONE ATTIVA]")
    print("  Isola A (node-1, node-2):")
    results = []
    for (tid, cpu, mem, lat, pol) in tasks_a:
        # Sync solo dentro isola A
        gossip_round(island_a)
        r = run_task(tid, cpu, mem, lat, pol, island_a)
        results.append(r)
        print(f"    {tid}: {r['status']:6s} on {r['placed_on']}")

    print("  Isola B (node-3, node-4):")
    for (tid, cpu, mem, lat, pol) in tasks_b:
        # Sync solo dentro isola B
        gossip_round(island_b)
        r = run_task(tid, cpu, mem, lat, pol, island_b)
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

    # ── Task post-riconnessione (cluster completo) ─────────────
    print("  [POST-RICONNESSIONE] Cluster completo:")
    tasks_post = [
        ("pc01", 1.0, 256,  50.0, PlacementPolicy.LATENCY_FIRST),
        ("pc02", 2.0, 512, 100.0, PlacementPolicy.BALANCED),
    ]
    for (tid, cpu, mem, lat, pol) in tasks_post:
        r = run_task(tid, cpu, mem, lat, pol, agents)
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

    # Task con latenze MOLTO stringenti: forzano piu' round di negoziazione.
    # Le latenze base dei nodi sono: node-1=15ms, node-2=40ms,
    # node-3=80ms, node-4=25ms  (piu' jitter fino a +15ms)
    # Quindi requisiti < 20ms rendono difficile trovare NE al primo round.
    tasks = [
        # (id,   cpu,  mem,  max_lat, policy)
        ("n01", 1.0,  256,  12.0, PlacementPolicy.LATENCY_FIRST),  # quasi impossibile
        ("n02", 2.0,  512,  18.0, PlacementPolicy.BALANCED),       # molto stretta
        ("n03", 0.5,  128,  10.0, PlacementPolicy.LATENCY_FIRST),  # impossibile greedy
        ("n04", 4.0, 1024,  20.0, PlacementPolicy.LATENCY_FIRST),  # stretta + pesante
        ("n05", 1.0,  256,  15.0, PlacementPolicy.ENERGY_FIRST),   # stretta
        ("n06", 2.0,  512,  25.0, PlacementPolicy.BALANCED),       # al limite
        ("n07", 0.5,  128,   8.0, PlacementPolicy.LATENCY_FIRST),  # impossibile greedy
        ("n08", 3.0,  768,  22.0, PlacementPolicy.BALANCED),       # stretta + media
    ]

    # ── Run GREEDY (TaskAgent standard — singolo round) ──────────────────────
    print("\n  [GREEDY — singolo round, nessuna negoziazione]")
    greedy_agents  = make_resource_agents()
    greedy_results = []
    for (tid, cpu, mem, lat, pol) in tasks:
        r = run_task(tid, cpu, mem, lat, pol, greedy_agents)
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




def plot_placement_latency(all_metrics: dict):
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = list(all_metrics.keys())
    means = [all_metrics[s]["placement_latency_ms"] for s in scenarios]
    stds  = [all_metrics[s].get("placement_latency_std", 0) for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, means, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(scenarios, means, yerr=stds, fmt="none",
                color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Placement Latency (ms)", fontsize=12)
    ax.set_title("Placement Latency per Scenario", fontsize=14, fontweight="bold")
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
    stds  = [all_metrics[s].get("a2a_overhead_std", 0) for s in scenarios]
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, means, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)
    ax.errorbar(scenarios, means, yerr=stds, fmt="none",
                color="black", capsize=5, linewidth=1.5, zorder=4)

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("A2A Overhead (ms)", fontsize=12)
    ax.set_title("A2A Protocol Overhead per Scenario", fontsize=14, fontweight="bold")
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

    ax.set_ylabel("SLA Violation Rate (%)", fontsize=12)
    ax.set_title("SLA Violation Rate per Scenario", fontsize=14, fontweight="bold")
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
    colors = [SCENARIO_COLORS[s] for s in scenarios]

    bars = ax.bar(scenarios, times, color=colors, width=0.5,
                  edgecolor="white", linewidth=1.2, zorder=3)

    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Convergence Time (ms)", fontsize=12)
    ax.set_title("CRDT Convergence Time per Scenario", fontsize=14, fontweight="bold")
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

    # ── (0,0) Placement Latency ──────────────────────────────
    ax = axes[0][0]
    vals = [all_metrics[s]["placement_latency_ms"] for s in scenarios]
    stds = [all_metrics[s].get("placement_latency_std", 0) for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(x, vals, yerr=stds, fmt="none", color="black", capsize=4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("Placement Latency (ms)", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals)*0.02, f"{v:.1f}", ha="center", fontsize=8)

    # ── (0,1) A2A Overhead ───────────────────────────────────
    ax = axes[0][1]
    vals = [all_metrics[s]["a2a_overhead_ms"] for s in scenarios]
    stds = [all_metrics[s].get("a2a_overhead_std", 0) for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.errorbar(x, vals, yerr=stds, fmt="none", color="black", capsize=4, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("A2A Protocol Overhead (ms)", fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + max(vals)*0.02, f"{v:.1f}", ha="center", fontsize=8)

    # ── (1,0) SLA Violation Rate ─────────────────────────────
    ax = axes[1][0]
    vals = [all_metrics[s]["sla_violation_rate"] * 100 for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.axhline(y=10, color="red", linestyle="--", linewidth=1.3,
               label="Soglia 10%", zorder=4)
    ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("SLA Violation Rate (%)", fontweight="bold")
    ax.set_ylim(0, max(max(vals) * 1.35, 15))
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    # ── (1,1) CRDT Convergence ───────────────────────────────
    ax = axes[1][1]
    vals = [all_metrics[s]["crdt_convergence_ms"] for s in scenarios]
    ax.bar(x, vals, width=bar_w, color=colors, edgecolor="white", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_title("CRDT Convergence Time (ms)", fontweight="bold")
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
    """
    Grafico specifico per S4: mostra la divergenza CRDT
    prima e dopo la riconnessione.
    """
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
    """
    Due subplot:
      (a) Rounds to Nash Equilibrium per task
          Verde  = NE al round 1 (nessun rilassamento necessario)
          Giallo = NE al round 2
          Rosso  = NE al round 3+
          Viola  = fallback (max_rounds esaurito, nessun NE formale)
          Grigio = task fallito (0 proposte in tutti i round)
      (b) Confronto Greedy vs Nash IBR su 3 indicatori:
          task piazzati, fallimenti, violazioni SLA (requisiti originali)
    """
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
    sep(); print("  FASE 4 — Experimentation & Evaluation"); sep()
    ray.init(ignore_reinit_error=True)

    # ── Aggiunge metodo mark_offline_self agli agenti ────────────
    # (necessario per S3 — viene chiamato sull'agente da remoto)

    all_metrics: Dict[str, Any] = {}

    # ── Esecuzione scenari ───────────────────────────────────────
    sep("─")
    print("SCENARIO 1 — Baseline"); sep("─")
    all_metrics["S1 Baseline"] = scenario_baseline()

    sep("─")
    print("SCENARIO 2 — High Load"); sep("─")
    all_metrics["S2 High Load"] = scenario_high_load()

    sep("─")
    print("SCENARIO 3 — Node Failure"); sep("─")
    all_metrics["S3 Node Failure"] = scenario_node_failure()

    sep("─")
    print("SCENARIO 4 — Network Partition"); sep("─")
    all_metrics["S4 Net Partition"] = scenario_network_partition()

    sep("─")
    print("SCENARIO 5 — Nash Equilibrium (Greedy vs IBR)"); sep("─")
    all_metrics["S5 Nash Equil."] = scenario_s5_nash()

    # ── Riepilogo testuale ───────────────────────────────────────
    sep()
    print("  RIEPILOGO METRICHE")
    sep()
    print(f"{'Scenario':<22} {'PlacLat(ms)':<14} {'A2A(ms)':<12} "
          f"{'SLA viol%':<12} {'CRDT conv(ms)'}")
    sep("─")
    for s, m in all_metrics.items():
        print(f"{s:<22} "
              f"{m['placement_latency_ms']:<14.1f} "
              f"{m['a2a_overhead_ms']:<12.1f} "
              f"{m['sla_violation_rate']*100:<12.1f} "
              f"{m['crdt_convergence_ms']:.1f}")

    # ── Salva dati grezzi ────────────────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "raw_results.json")
    # Converti numpy types per JSON serialization
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
    plot_partition_crdt_divergence(all_metrics["S4 Net Partition"])
    plot_nash_convergence(all_metrics["S5 Nash Equil."])

    sep()
    print("  FASE 4 COMPLETATA")
    print(f"  Grafici in: {RESULTS_DIR}/")
    sep()

    ray.shutdown()


if __name__ == "__main__":
    main()