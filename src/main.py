import ray
import time
import json
from typing import List

from protocol import TaskRequirements, PlacementPolicy
from agents import ResourceAgent, TaskAgent
from crdt_catalogue import ResourceCatalogue


# ─────────────────────────────────────────────
# Configurazione Nodi Edge 
# ─────────────────────────────────────────────

EDGE_NODES = [
    # (node_id,     cpu,  mem_mb, latency_ms, energy_score)
    ("edge-node-1",  8.0, 4096,   15.0,  0.3),   # nodo potente, bassa latenza, verde
    ("edge-node-2",  4.0, 2048,   40.0,  0.5),   # nodo medio
    ("edge-node-3",  2.0, 1024,   80.0,  0.2),   # nodo debole, efficiente
    ("edge-node-4", 16.0, 8192,   25.0,  0.8),   # nodo potente ma energivoro
]

# ─────────────────────────────────────────────
# Configurazione Task da Piazzare
# ─────────────────────────────────────────────

TASKS = [
    # (task_id,    cpu, mem_mb, max_lat_ms, duration_s, priority, tipo,        policy)
    ("task-01",    1.0,  256,   50.0,  10, 3, "inference", PlacementPolicy.LATENCY_FIRST),
    ("task-02",    2.0,  512,  100.0,  20, 2, "batch",     PlacementPolicy.ENERGY_FIRST),
    ("task-03",    0.5,  128,  200.0,   5, 1, "generic",   PlacementPolicy.BALANCED),
    ("task-04",    4.0, 1024,   30.0,  15, 3, "streaming", PlacementPolicy.LATENCY_FIRST),
    ("task-05",    1.0,  256,  500.0,  30, 1, "batch",     PlacementPolicy.ENERGY_FIRST),
    ("task-06",    2.0,  512,   60.0,  10, 2, "inference", PlacementPolicy.BALANCED),
    ("task-07",    8.0, 2048,  100.0,  60, 2, "batch",     PlacementPolicy.BALANCED),
    ("task-08",    0.5,  128,   20.0,   5, 3, "inference", PlacementPolicy.LATENCY_FIRST),
]


def print_separator(char="─", width=65):
    print(char * width)


def print_section(title: str):
    print_separator("═")
    print(f"  {title}")
    print_separator("═")


def simulate_crdt_gossip(resource_agents: List[ray.actor.ActorHandle]):
    """
    Simula un round di gossip CRDT tra tutti i nodi.
    """
    print("\n[CRDT Gossip] Sincronizzazione catalogo tra tutti i nodi...")
    catalogues = [ray.get(a.get_catalogue_object.remote()) for a in resource_agents]

    # Ogni nodo manda il proprio catalogo a tutti gli altri
    for i, agent in enumerate(resource_agents):
        for j, other_catalogue in enumerate(catalogues):
            if i != j:
                agent.sync_catalogue.remote(other_catalogue)

    time.sleep(0.1)  
    print("[CRDT Gossip] Round completato.")


def measure_catalogue_convergence(resource_agents: List[ray.actor.ActorHandle]) -> float:
    """
    Misura la convergenza CRDT: quante entry divergono tra i cataloghi dei nodi.
    """
    catalogues = [ray.get(a.get_catalogue_object.remote()) for a in resource_agents]
    total_diffs = 0
    comparisons = 0
    for i in range(len(catalogues)):
        for j in range(i + 1, len(catalogues)):
            diffs = catalogues[i].convergence_diff(catalogues[j])
            total_diffs += len(diffs)
            comparisons += 1

    if comparisons == 0:
        return 100.0
    avg_diffs = total_diffs / comparisons
    n_nodes = len(EDGE_NODES)
    convergence = max(0.0, (1.0 - avg_diffs / n_nodes) * 100.0)
    return convergence


def print_results_table(results: List[dict]):
    print_section("RISULTATI PLACEMENT")
    header = f"{'Task':<12} {'Stato':<10} {'Nodo':<14} {'Policy':<16} {'Score':<8} {'A2A(ms)':<10} {'Lat(ms)':<10}"
    print(header)
    print_separator()

    placed = 0
    failed = 0
    total_a2a_ms = 0
    sla_violations = 0

    for r in results:
        status = r.get("status", "?")
        if status == "placed":
            placed += 1
            a2a_ms = r.get("placement_latency_ms", 0)
            total_a2a_ms += a2a_ms
            lat = r.get("estimated_latency_ms", 0)

            # Controlla SLA: la latenza stimata deve stare nel budget richiesto
            task_cfg = next((t for t in TASKS if t[0] == r["task_id"]), None)
            sla_ok = "✓" if task_cfg and lat <= task_cfg[3] else "✗ SLA!"
            if "SLA!" in sla_ok:
                sla_violations += 1

            print(f"{r['task_id']:<12} {status:<10} "
                  f"{r.get('placed_on','?'):<14} "
                  f"{r.get('policy','?'):<16} "
                  f"{r.get('score', 0):<8.3f} "
                  f"{a2a_ms:<10.1f} "
                  f"{lat:<10.1f} {sla_ok}")
        else:
            failed += 1
            print(f"{r['task_id']:<12} {'FAILED':<10} "
                  f"{'N/A':<14} {r.get('reason','?'):<16}")

    print_separator()
    print(f"\n  Task piazzati:      {placed}/{len(results)}")
    print(f"  Task falliti:       {failed}/{len(results)}")
    print(f"  SLA violations:     {sla_violations}/{placed}")
    if placed > 0:
        print(f"  A2A overhead medio: {total_a2a_ms/placed:.1f}ms")


def print_node_states(resource_agents: List[ray.actor.ActorHandle]):
    print_section("STATO NODI EDGE (dopo placement)")
    header = f"{'Nodo':<14} {'CPU avail':<12} {'MEM avail':<12} {'Task attivi':<13} {'Lat(ms)':<10}"
    print(header)
    print_separator()
    states = [ray.get(a.get_state.remote()) for a in resource_agents]
    for s in states:
        cpu_used_pct = (1 - s['available_cpu'] / dict(zip(
            [n[0] for n in EDGE_NODES], [n[1] for n in EDGE_NODES]
        ))[s['node_id']]) * 100
        print(f"{s['node_id']:<14} "
              f"{s['available_cpu']:.1f} ({100-cpu_used_pct:.0f}% free)  "
              f"{s['available_memory_mb']:.0f}MB{'':<4} "
              f"{s['active_tasks']:<13} "
              f"{s['current_latency_ms']:.0f}ms")


def main():
    #  Avvio Ray 
    print_section("AVVIO SISTEMA — Agentic Edge Orchestration")
    ray.init(ignore_reinit_error=True)
    print(f"Ray avviato. Risorse cluster: {ray.cluster_resources()}")

    #  Crea ResourceAgent 
    print_section("FASE 2 — Inizializzazione Resource Agents")
    resource_agents = []
    for (node_id, cpu, mem, lat, energy) in EDGE_NODES:
        agent = ResourceAgent.remote(node_id, cpu, mem, lat, energy)
        resource_agents.append(agent)
        print(f"  Creato ResourceAgent: {node_id}")

    # Registra i peer 
    for i, agent in enumerate(resource_agents):
        peers = [a for j, a in enumerate(resource_agents) if j != i]
        agent.register_peers.remote(peers)

    time.sleep(0.2)  # attendi inizializzazione

    # Mostra catalogo CRDT iniziale 
    print_section("FASE 2 — Resource Catalogue CRDT (stato iniziale)")
    for agent in resource_agents:
        snap = ray.get(agent.get_catalogue_snapshot.remote())
        print(f"  {snap['owner']}: clock={snap['lamport_clock']}, "
              f"nodi nel catalogo={len(snap['nodes'])}")

    # Esegui placement dei task 
    print_section("FASE 3 — Negoziazione A2A: Placement dei Task")
    all_results = []

    for (task_id, cpu, mem, max_lat, dur, prio, ttype, policy) in TASKS:
        print(f"\n{'─'*55}")
        print(f"  Avvio negoziazione per {task_id} "
              f"[{ttype}, {policy.value}]")

        req = TaskRequirements(
            cpu_cores=cpu,
            memory_mb=mem,
            max_latency_ms=max_lat,
            duration_sec=dur,
            priority=prio,
            task_type=ttype,
        )
        task_agent = TaskAgent.remote(task_id, req, policy)
        result = ray.get(task_agent.place.remote(resource_agents))
        all_results.append(result)

        time.sleep(0.05)

    #  CRDT Gossip Sync 
    print_section("FASE 3 — CRDT Gossip Synchronization")
    t_gossip_start = time.time()
    simulate_crdt_gossip(resource_agents)
    t_gossip_end = time.time()

    convergence = measure_catalogue_convergence(resource_agents)
    print(f"  Convergenza catalogo CRDT: {convergence:.1f}%")
    print(f"  Gossip round completato in: {(t_gossip_end-t_gossip_start)*1000:.1f}ms")

    print_results_table(all_results)
    print_node_states(resource_agents)

    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": time.time(),
            "placements": all_results,
            "crdt_convergence_pct": convergence,
            "gossip_duration_ms": (t_gossip_end - t_gossip_start) * 1000,
        }, f, indent=2)
    print(f"\n  Risultati salvati in: results.json")
    print(f"  (Questi dati verranno usati nella Fase 4 per i grafici)")

    print_section("SIMULAZIONE COMPLETATA")
    ray.shutdown()


if __name__ == "__main__":
    main()
