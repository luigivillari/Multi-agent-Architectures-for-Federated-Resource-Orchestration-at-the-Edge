"""
agents.py
=========
Fase 2 — Agent Modelling: ResourceAgent e TaskAgent come Ray Actors
"""

import ray
import time
import random
from typing import Dict, List, Optional, Tuple

from protocol import (
    A2AMessage, MessageType, TaskRequirements, ResourceOffer,
    PlacementPolicy, score_offer,
    make_propose, make_reject, make_accept, make_counter_offer, make_inform_done
)
from crdt_catalogue import ResourceCatalogue


# ═══════════════════════════════════════════════════════════
#  RESOURCE AGENT
# ═══════════════════════════════════════════════════════════

@ray.remote
class ResourceAgent:
    def __init__(self, node_id: str, total_cpu: float, total_memory_mb: float,
                 latency_ms: float, energy_score: float):

        self.node_id = node_id
        self.total_cpu = total_cpu
        self.total_memory_mb = total_memory_mb
        self.base_latency_ms = latency_ms
        self.energy_score = energy_score

        self.available_cpu = total_cpu
        self.available_memory_mb = total_memory_mb
        self.current_latency_ms = latency_ms

        self.active_tasks: Dict[str, Tuple[float, float, float]] = {}
        self.negotiation_log: List[dict] = []

        self.catalogue = ResourceCatalogue(owner_node_id=node_id)
        self.catalogue.upsert_node(
            node_id=node_id, cpu=self.available_cpu,
            memory_mb=self.available_memory_mb, latency_ms=self.current_latency_ms,
            energy_score=self.energy_score, active_tasks=0, is_online=True
        )
        self.peer_agents: List = []

        print(f"[{node_id}] ResourceAgent avviato: "
              f"CPU={total_cpu}, MEM={total_memory_mb}MB, "
              f"LAT={latency_ms}ms, ENERGY={energy_score:.2f}")

    def register_peers(self, peers: list):
        self.peer_agents = peers

    def get_catalogue_snapshot(self) -> dict:
        return self.catalogue.state_snapshot()

    def get_catalogue_object(self) -> ResourceCatalogue:
        return self.catalogue

    def sync_catalogue(self, remote_catalogue: ResourceCatalogue):
        before_clock = self.catalogue._lamport_clock
        self.catalogue.merge(remote_catalogue)
        after_clock = self.catalogue._lamport_clock
        if after_clock > before_clock:
            print(f"[{self.node_id}] CRDT sync: clock {before_clock} -> {after_clock}")

    def receive_cfp(self, cfp: A2AMessage) -> Optional[A2AMessage]:
        req = cfp.requirements
        start_time = time.time()

        jitter = random.uniform(-5, 15)
        self.current_latency_ms = max(1.0, self.base_latency_ms + jitter)

        can_satisfy_cpu = self.available_cpu >= req.cpu_cores
        can_satisfy_mem = self.available_memory_mb >= req.memory_mb
        can_satisfy_lat = self.current_latency_ms <= req.max_latency_ms

        log_entry = {
            "event": "cfp_received", "node_id": self.node_id,
            "task_id": cfp.task_id, "conv_id": cfp.conversation_id,
            "timestamp": start_time, "available_cpu": self.available_cpu,
            "available_mem": self.available_memory_mb,
            "required_cpu": req.cpu_cores, "required_mem": req.memory_mb,
        }

        if can_satisfy_cpu and can_satisfy_mem and can_satisfy_lat:
            offer = ResourceOffer(
                node_id=self.node_id, available_cpu=self.available_cpu,
                available_memory_mb=self.available_memory_mb,
                estimated_latency_ms=self.current_latency_ms,
                energy_cost_score=self.energy_score,
            )
            response = make_propose(self.node_id, cfp.sender_id,
                                    cfp.task_id, cfp.conversation_id, offer)
            log_entry["response"] = "PROPOSE"

        elif (self.available_cpu >= req.cpu_cores * 0.7 and
              self.available_memory_mb >= req.memory_mb * 0.7):
            offer = ResourceOffer(
                node_id=self.node_id, available_cpu=self.available_cpu,
                available_memory_mb=self.available_memory_mb,
                estimated_latency_ms=self.current_latency_ms,
                energy_cost_score=self.energy_score,
            )
            response = make_counter_offer(self.node_id, cfp.sender_id,
                                          cfp.task_id, cfp.conversation_id, offer)
            log_entry["response"] = "COUNTER_OFFER"

        else:
            reasons = []
            if not can_satisfy_cpu:
                reasons.append(f"CPU: {self.available_cpu:.1f}/{req.cpu_cores} richiesti")
            if not can_satisfy_mem:
                reasons.append(f"MEM: {self.available_memory_mb:.0f}/{req.memory_mb}MB richiesti")
            if not can_satisfy_lat:
                reasons.append(f"LAT: {self.current_latency_ms:.0f}ms > {req.max_latency_ms}ms max")
            response = make_reject(self.node_id, cfp.sender_id,
                                   cfp.task_id, cfp.conversation_id,
                                   " | ".join(reasons))
            log_entry["response"] = "REJECT"
            log_entry["reason"] = " | ".join(reasons)

        self.negotiation_log.append(log_entry)
        return response

    def receive_reject(self, reject_msg: A2AMessage):
        self.negotiation_log.append({
            "event": "proposal_rejected", "node_id": self.node_id,
            "task_id": reject_msg.task_id, "reason": reject_msg.reason,
            "timestamp": time.time(),
        })

    def receive_accept(self, accept_msg: A2AMessage) -> A2AMessage:
        task_id = accept_msg.task_id
        cpu_alloc = min(2.0, self.available_cpu)
        mem_alloc = min(512.0, self.available_memory_mb)

        self.available_cpu -= cpu_alloc
        self.available_memory_mb -= mem_alloc
        self.active_tasks[task_id] = (cpu_alloc, mem_alloc, time.time())

        self.catalogue.upsert_node(
            node_id=self.node_id, cpu=self.available_cpu,
            memory_mb=self.available_memory_mb, latency_ms=self.current_latency_ms,
            energy_score=self.energy_score, active_tasks=len(self.active_tasks),
            is_online=True,
        )

        print(f"[{self.node_id}] ACCEPT ricevuto per task={task_id}. "
              f"CPU rimasta: {self.available_cpu:.1f}, MEM: {self.available_memory_mb:.0f}MB")

        self.negotiation_log.append({
            "event": "task_accepted", "node_id": self.node_id,
            "task_id": task_id, "cpu_allocated": cpu_alloc,
            "mem_allocated": mem_alloc, "timestamp": time.time(),
        })

        return make_inform_done(self.node_id, accept_msg.sender_id,
                                task_id, accept_msg.conversation_id)

    def mark_offline_self(self):
        """Marca questo nodo come offline nel proprio catalogo CRDT."""
        self.catalogue.mark_offline(self.node_id)
        print(f"[{self.node_id}] Marcato OFFLINE nel catalogo CRDT")

    def mark_node_offline_external(self, node_id: str):
        """
        Un nodo sopravvissuto marca un altro nodo come offline nel proprio
        catalogo CRDT locale (usato dopo rilevamento crash via RayActorError).
        """
        self.catalogue.mark_offline(node_id)
        print(f"[{self.node_id}] CRDT aggiornato: {node_id} marcato OFFLINE (crash rilevato)")

    def complete_task(self, task_id: str):
        if task_id in self.active_tasks:
            cpu_alloc, mem_alloc, start_t = self.active_tasks.pop(task_id)
            self.available_cpu = min(self.total_cpu, self.available_cpu + cpu_alloc)
            self.available_memory_mb = min(self.total_memory_mb,
                                           self.available_memory_mb + mem_alloc)
            self.catalogue.upsert_node(
                node_id=self.node_id, cpu=self.available_cpu,
                memory_mb=self.available_memory_mb, latency_ms=self.current_latency_ms,
                energy_score=self.energy_score, active_tasks=len(self.active_tasks),
                is_online=True,
            )
            print(f"[{self.node_id}] Task {task_id} completato in "
                  f"{time.time()-start_t:.2f}s")

    def get_state(self) -> dict:
        return {
            "node_id": self.node_id,
            "available_cpu": self.available_cpu,
            "available_memory_mb": self.available_memory_mb,
            "current_latency_ms": self.current_latency_ms,
            "energy_score": self.energy_score,
            "active_tasks": len(self.active_tasks),
            "total_negotiations": len(self.negotiation_log),
        }

    def get_negotiation_log(self) -> List[dict]:
        return self.negotiation_log


# ═══════════════════════════════════════════════════════════
#  TASK AGENT
# ═══════════════════════════════════════════════════════════

@ray.remote
class TaskAgent:
    def __init__(self, task_id: str, requirements: TaskRequirements,
                 policy: PlacementPolicy = PlacementPolicy.BALANCED):

        self.task_id = task_id
        self.requirements = requirements
        self.policy = policy
        self.status = "pending"
        self.placed_on: Optional[str] = None
        self.placement_latency_ms: Optional[float] = None
        self.received_proposals: Dict[str, A2AMessage] = {}
        self.rejected_by: List[str] = []
        self.event_log: List[dict] = []

        print(f"[TaskAgent:{task_id}] Creato. "
              f"Policy={policy.value}, CPU={requirements.cpu_cores}, "
              f"MEM={requirements.memory_mb}MB, LAT_MAX={requirements.max_latency_ms}ms")

    def place(self, resource_agents: list) -> dict:
        """
        Negoziazione CNP completa via comunicazione A2A tra Actor Ray.

        Questa è la vera interazione Agent-to-Agent: il TaskAgent (Actor)
        invia messaggi CFP agli ResourceAgent (Actor) e riceve le risposte
        attraverso il message-passing di Ray, esattamente come avverrebbe
        su una rete reale tra nodi distribuiti.

        Aggiunto rispetto alla versione originale:
          - Misura separata di a2a_overhead_ms (broadcast CFP → raccolta risposte)
          - Gestione RayActorError per fault tolerance (scenario S3 node failure)
          - dead_agent_indices: indici dei nodi che non hanno risposto (per CRDT update)
          - sla_ok e max_latency_ms nel risultato (per metriche di valutazione)
        """
        from protocol import make_cfp
        t_start = time.time()
        self.status = "negotiating"

        cfp = make_cfp(f"task-{self.task_id}", self.task_id, self.requirements)
        self.event_log.append({"event": "cfp_sent", "timestamp": t_start,
                                "n_agents": len(resource_agents)})
        print(f"[TaskAgent:{self.task_id}] -> CFP inviata a {len(resource_agents)} nodi")

        # ── Mapping node_id → actor costruito UNA SOLA VOLTA prima della fase CFP ──
        # Se fatto dopo (come in get_state post-A2A), aggiunge un ray.get() su tutti
        # i nodi FUORI da a2a_overhead_ms ma DENTRO placement_latency_ms,
        # introducendo varianza artificiale nei CI. Qui invece è parte del setup
        # costante, prima di t_cfp, e non inquina la misurazione di placement_latency.
        # Gestione fault-tolerant: se un nodo è già morto (S3 post-failure),
        # l'ActorDiedError viene ignorato — l'agente sarà escluso anche durante i CFP.
        node_id_to_agent = {}
        for _a in resource_agents:
            try:
                _s = ray.get(_a.get_state.remote())
                node_id_to_agent[_s["node_id"]] = _a
            except (ray.exceptions.RayActorError,
                    ray.exceptions.ActorDiedError):
                pass  # nodo già morto — sarà rilevato come dead_agent durante CFP

        # ── A2A overhead: tempo dal broadcast CFP alla raccolta di tutte le risposte ──
        # Raccoglie individualmente per gestire crash di singoli nodi (RayActorError)
        # senza abortire l'intera negoziazione.
        t_cfp = time.time()
        refs = [agent.receive_cfp.remote(cfp) for agent in resource_agents]
        responses = []
        dead_agent_indices: List[int] = []
        for idx, ref in enumerate(refs):
            try:
                resp = ray.get(ref)
                responses.append(resp)
            except ray.exceptions.RayActorError:
                dead_agent_indices.append(idx)
                print(f"[TaskAgent:{self.task_id}] [CRASH] Nodo {idx} non risponde "
                      f"— RayActorError rilevato durante raccolta CFP")
        t_responses = time.time()
        a2a_overhead_ms = (t_responses - t_cfp) * 1000

        # ── Scoring delle proposte ricevute ──────────────────────────────────────
        proposals = []
        for resp in responses:
            if resp is None:
                continue
            if resp.msg_type in (MessageType.PROPOSE, MessageType.COUNTER_OFFER):
                score = score_offer(resp.offer, self.policy)
                resp.offer.score = score
                proposals.append(resp)
                self.received_proposals[resp.sender_id] = resp
                print(f"[TaskAgent:{self.task_id}] <- {resp.msg_type.value} da "
                      f"{resp.sender_id}: score={score:.3f}, "
                      f"lat={resp.offer.estimated_latency_ms:.0f}ms")
            elif resp.msg_type == MessageType.REJECT:
                self.rejected_by.append(resp.sender_id)
                print(f"[TaskAgent:{self.task_id}] <- REJECT da "
                      f"{resp.sender_id}: {resp.reason}")

        if not proposals:
            self.status = "failed"
            t_end = time.time()
            result = {
                "task_id":              self.task_id,
                "status":               "failed",
                "reason":               "no_proposals",
                "placement_latency_ms": (t_end - t_start) * 1000,
                "a2a_overhead_ms":      a2a_overhead_ms,
                "max_latency_ms":       self.requirements.max_latency_ms,
                "sla_ok":               False,
                "proposals_received":   0,
                "dead_agent_indices":   dead_agent_indices,
            }
            print(f"[TaskAgent:{self.task_id}] x Nessuna proposta ricevuta.")
            return result

        # ── Best response: seleziona l'offerta con score massimo ─────────────────
        best   = max(proposals, key=lambda r: r.offer.score)
        losers = [r for r in proposals if r.sender_id != best.sender_id]
        print(f"[TaskAgent:{self.task_id}] -> Vincitore: {best.sender_id} "
              f"(score={best.offer.score:.3f})")

        # ── ACCEPT / REJECT usando il mapping pre-costruito ─────────────────────
        # Nessun ray.get([get_state.remote()]) post-A2A: il mapping node_id_to_agent
        # è già disponibile. Questo elimina la principale fonte di varianza in
        # placement_latency_ms che non era presente in a2a_overhead_ms.
        accept_msg = make_accept(f"task-{self.task_id}", best.sender_id,
                                 self.task_id, cfp.conversation_id)
        ray.get(node_id_to_agent[best.sender_id].receive_accept.remote(accept_msg))

        for loser_resp in losers:
            reject_msg = make_reject(f"task-{self.task_id}", loser_resp.sender_id,
                                     self.task_id, cfp.conversation_id,
                                     "better_offer_available")
            node_id_to_agent[loser_resp.sender_id].receive_reject.remote(reject_msg)

        t_end = time.time()
        self.status = "placed"
        self.placed_on = best.sender_id
        self.placement_latency_ms = (t_end - t_start) * 1000

        estimated_latency_ms = best.offer.estimated_latency_ms
        sla_ok = estimated_latency_ms <= self.requirements.max_latency_ms

        result = {
            "task_id":              self.task_id,
            "status":               "placed",
            "placed_on":            self.placed_on,
            "policy":               self.policy.value,
            "score":                best.offer.score,
            "estimated_latency_ms": estimated_latency_ms,
            "max_latency_ms":       self.requirements.max_latency_ms,
            "sla_ok":               sla_ok,
            "energy_score":         best.offer.energy_cost_score,
            "placement_latency_ms": self.placement_latency_ms,
            "a2a_overhead_ms":      a2a_overhead_ms,
            "proposals_received":   len(proposals),
            "rejections_received":  len(self.rejected_by),
            "dead_agent_indices":   dead_agent_indices,
        }
        self.event_log.append({"event": "placement_success", **result})
        print(f"[TaskAgent:{self.task_id}] Piazzato su {self.placed_on} "
              f"in {self.placement_latency_ms:.1f}ms | SLA={'OK' if sla_ok else 'VIOLATION'}")
        return result

    def get_status(self) -> dict:
        return {
            "task_id": self.task_id, "status": self.status,
            "placed_on": self.placed_on,
            "placement_latency_ms": self.placement_latency_ms,
            "policy": self.policy.value,
        }


# ═══════════════════════════════════════════════════════════
#  NASH TASK AGENT — Iterative Best Response
# ═══════════════════════════════════════════════════════════

@ray.remote
class NashTaskAgent:
    """
    TaskAgent con negoziazione multi-round basata su Iterative Best Response (IBR).

    Modello di gioco:
      - Giocatori : N ResourceAgent  +  1 NashTaskAgent (coordinatore)
      - Strategia ResourceAgent  : {PROPOSE, COUNTER_OFFER, REJECT}
                                   in base alle risorse disponibili.
                                   Strategia dominante = dichiarazione onesta
                                   (nessun incentivo a mentire).
      - Strategia NashTaskAgent  : seleziona l'offerta con score massimo
                                   (best response del coordinatore).

    Il processo IBR termina quando si verificano TUTTE le condizioni di Nash Equilibrium:
      C1 - Razionalita' individuale  : utility del vincitore > 0
      C2 - Ottimalita' del meccanismo: score del vincitore e' il massimo tra le proposte
      C3 - Stabilita' SLA            : latenza stimata <= max_latency_ms corrente
      C4 - Meccanismo veritiero      : dichiarazione onesta e' dominant strategy
                                       (proprieta' strutturale - sempre True by design)

    Se il NE non e' raggiunto, i requisiti vengono rilassati progressivamente
    (max_latency_ms * (1 + relaxation_factor * round),
     CPU e MEM con floor al 70% del valore originale)
    fino a max_rounds iterazioni, poi fallback al miglior risultato disponibile.
    """

    def __init__(self, task_id: str, requirements: TaskRequirements,
                 policy: PlacementPolicy = PlacementPolicy.BALANCED,
                 max_rounds: int = 5,
                 relaxation_factor: float = 0.20):

        self.task_id           = task_id
        self.requirements      = requirements
        self.policy            = policy
        self.max_rounds        = max_rounds
        self.relaxation_factor = relaxation_factor

        self.status                               = "pending"
        self.placed_on: Optional[str]             = None
        self.placement_latency_ms: Optional[float] = None
        self.nash_rounds_to_convergence: Optional[int] = None
        self.nash_log: List[dict]                 = []
        self.event_log: List[dict]                = []

        print(f"[NashAgent:{task_id}] Creato. Policy={policy.value}, "
              f"max_rounds={max_rounds}, relaxation={relaxation_factor:.0%}")

    # ── Utilita' e verifica Nash ─────────────────────────────────────────────

    def _compute_agent_utility(self, offer: ResourceOffer,
                                req: TaskRequirements) -> float:
        """
        Utilita' del ResourceAgent per questo task.
        Media normalizzata di CPU ratio, MEM ratio e latency ratio.
        Valore in [0, 1]: 1 = risorse abbondanti e latenza perfetta.
        """
        cpu_ratio = min(1.0, offer.available_cpu / max(req.cpu_cores, 0.01))
        mem_ratio = min(1.0, offer.available_memory_mb / max(req.memory_mb, 0.01))
        lat_ratio = max(0.0, 1.0 - offer.estimated_latency_ms
                        / max(req.max_latency_ms, 1.0))
        return round((cpu_ratio + mem_ratio + lat_ratio) / 3.0, 4)

    def _verify_nash_conditions(self, winner_resp: A2AMessage,
                                 all_proposals: list,
                                 current_req: TaskRequirements) -> dict:
        """
        Verifica le 4 condizioni di Nash Equilibrium.

        C1 - Razionalita' individuale:
             Il vincitore ha utility > 0, accetterebbe il task volontariamente.
        C2 - Ottimalita' del meccanismo:
             Il TaskAgent ha scelto l'offerta con score massimo (sua best response).
             Nessun altro agente potrebbe migliorare l'outcome del coordinatore.
        C3 - Stabilita' SLA:
             La latenza stimata del vincitore rispetta il requisito corrente.
             Se C3 = False siamo in COUNTER_OFFER fuori SLA -> allocazione instabile.
        C4 - Meccanismo veritiero (dominant strategy):
             Dichiarare le risorse reali e' sempre optimal per i ResourceAgent.
             Proprieta' strutturale del meccanismo - sempre True by design.
        """
        winner_utility = self._compute_agent_utility(winner_resp.offer, current_req)
        winner_score   = winner_resp.offer.score

        c1 = winner_utility > 0.0
        c2 = all(p.offer.score <= winner_score + 1e-9 for p in all_proposals)
        c3 = (winner_resp.offer.estimated_latency_ms <= current_req.max_latency_ms)
        c4 = True   # garantito dalla struttura del meccanismo d'asta

        return {
            "is_equilibrium":            c1 and c2 and c3 and c4,
            "winner_utility":            winner_utility,
            "winner_score":              winner_score,
            "C1_individual_rationality": c1,
            "C2_mechanism_optimality":   c2,
            "C3_sla_stability":          c3,
            "C4_truthful_mechanism":     c4,
        }

    def _relax_requirements(self, req: TaskRequirements,
                             round_num: int) -> TaskRequirements:
        """
        Rilassa i requisiti per il round successivo.
          - max_latency_ms : aumenta del relaxation_factor * (round+1)
          - cpu_cores      : riduzione lieve, floor al 70% dell'originale
          - memory_mb      : riduzione lieve, floor al 70% dell'originale
        """
        factor  = 1.0 + self.relaxation_factor * (round_num + 1)
        new_lat = req.max_latency_ms * factor

        cpu_floor = self.requirements.cpu_cores * 0.70
        mem_floor = self.requirements.memory_mb * 0.70
        new_cpu   = max(cpu_floor, req.cpu_cores * (1 - self.relaxation_factor * 0.3))
        new_mem   = max(mem_floor, req.memory_mb * (1 - self.relaxation_factor * 0.3))

        return TaskRequirements(
            cpu_cores      = round(new_cpu, 2),
            memory_mb      = round(new_mem, 2),
            max_latency_ms = round(new_lat, 2),
            duration_sec   = req.duration_sec,
            priority       = req.priority,
            task_type      = req.task_type,
        )

    # ── Placement multi-round ────────────────────────────────────────────────

    def place_nash(self, resource_agents: list) -> dict:
        """
        Negoziazione multi-round con Iterative Best Response.

        Ogni round:
          1. Ogni ResourceAgent risponde con la sua best response (offerta onesta)
          2. NashTaskAgent seleziona l'offerta con score massimo (sua best response)
          3. Si verificano le 4 condizioni di Nash Equilibrium
          4. NE raggiunto  -> ACCEPT al vincitore, REJECT agli altri, ritorna risultato
          5. NE non raggiunto -> rilassa i requisiti e ripete

        Se max_rounds esaurito senza NE -> usa il miglior risultato trovato (fallback).
        """
        from protocol import make_cfp, make_accept, make_reject

        t_start = time.time()
        self.status = "negotiating"

        # Copia locale dei requisiti (non altera self.requirements)
        current_req = TaskRequirements(
            cpu_cores      = self.requirements.cpu_cores,
            memory_mb      = self.requirements.memory_mb,
            max_latency_ms = self.requirements.max_latency_ms,
            duration_sec   = self.requirements.duration_sec,
            priority       = self.requirements.priority,
            task_type      = self.requirements.task_type,
        )

        best_available: Optional[dict] = None   # fallback: migliore offerta trovata

        for round_num in range(self.max_rounds):
            print(f"\n[NashAgent:{self.task_id}] Round {round_num + 1}/{self.max_rounds}"
                  f" - CPU>={current_req.cpu_cores}, MEM>={current_req.memory_mb}MB,"
                  f" LAT<={current_req.max_latency_ms:.1f}ms")

            # 1. Broadcast CFP con i requisiti correnti
            cfp   = make_cfp(f"nash-{self.task_id}-r{round_num}",
                             self.task_id, current_req)
            t_cfp = time.time()
            refs  = [a.receive_cfp.remote(cfp) for a in resource_agents]
            resps = ray.get(refs)
            a2a_ms = (time.time() - t_cfp) * 1000

            proposals  = []
            rejections = []
            for resp in resps:
                if resp is None:
                    continue
                if resp.msg_type in (MessageType.PROPOSE, MessageType.COUNTER_OFFER):
                    resp.offer.score = score_offer(resp.offer, self.policy)
                    proposals.append(resp)
                    print(f"  <- {resp.msg_type.value:<15} da {resp.sender_id}: "
                          f"score={resp.offer.score:.3f}, "
                          f"lat={resp.offer.estimated_latency_ms:.1f}ms")
                else:
                    rejections.append(resp)
                    print(f"  <- REJECT           da {resp.sender_id}")

            round_log = {
                "round":        round_num + 1,
                "requirements": {
                    "cpu_cores":      current_req.cpu_cores,
                    "memory_mb":      current_req.memory_mb,
                    "max_latency_ms": current_req.max_latency_ms,
                },
                "proposals":  len(proposals),
                "rejections": len(rejections),
                "a2a_ms":     a2a_ms,
            }

            # 2. Nessuna proposta -> rilassa e riprova
            if not proposals:
                print(f"  Nessuna proposta ricevuta. Rilascio requisiti.")
                round_log["outcome"] = "no_proposals"
                self.nash_log.append(round_log)
                current_req = self._relax_requirements(current_req, round_num)
                continue

            # 3. Best response del TaskAgent: seleziona score massimo
            best = max(proposals, key=lambda r: r.offer.score)

            # Aggiorna il fallback con il miglior risultato visto finora
            if (best_available is None or
                    best.offer.score > best_available["winner_resp"].offer.score):
                best_available = {
                    "winner_resp": best,
                    "proposals":   proposals,
                    "current_req": current_req,
                    "round_num":   round_num,
                }

            # 4. Verifica Nash Equilibrium
            nash = self._verify_nash_conditions(best, proposals, current_req)
            round_log["winner"]       = best.sender_id
            round_log["winner_score"] = best.offer.score
            round_log["nash"]         = nash

            if nash["is_equilibrium"]:
                print(f"  Nash Equilibrium al round {round_num + 1}! "
                      f"Vincitore={best.sender_id}, utility={nash['winner_utility']:.3f}")
                round_log["outcome"] = "nash_equilibrium"
                self.nash_log.append(round_log)
                return self._finalize_placement(
                    best, proposals, resource_agents, current_req,
                    round_num + 1, t_start, cfp.conversation_id,
                    nash_converged=True,
                )
            else:
                failed = [k for k, v in nash.items()
                          if k.startswith("C") and not v]
                print(f"  Non in equilibrio ({', '.join(failed)}). "
                      f"Rilascio requisiti.")
                round_log["outcome"]           = "not_equilibrium"
                round_log["failed_conditions"] = failed
                self.nash_log.append(round_log)
                current_req = self._relax_requirements(current_req, round_num)

        # Max rounds esaurito: usa il best_available come fallback
        print(f"[NashAgent:{self.task_id}] Max rounds esaurito - "
              f"uso best-available (fallback).")

        if best_available:
            return self._finalize_placement(
                best_available["winner_resp"],
                best_available["proposals"],
                resource_agents,
                best_available["current_req"],
                self.max_rounds,
                t_start,
                f"nash-{self.task_id}-fallback",
                nash_converged=False,
            )

        # Fallimento totale
        self.status = "failed"
        return {
            "task_id":              self.task_id,
            "status":               "failed",
            "reason":               "nash_exhausted_no_proposals",
            "nash_rounds":          self.max_rounds,
            "nash_converged":       False,
            "placement_latency_ms": (time.time() - t_start) * 1000,
        }

    def _finalize_placement(self, winner_resp: A2AMessage,
                             all_proposals: list,
                             resource_agents: list,
                             current_req: TaskRequirements,
                             rounds_used: int,
                             t_start: float,
                             conv_id: str,
                             nash_converged: bool) -> dict:
        from protocol import make_accept, make_reject

        winner_id = winner_resp.sender_id
        states    = ray.get([a.get_state.remote() for a in resource_agents])

        # ACCEPT al vincitore
        w_idx      = next(i for i, s in enumerate(states)
                          if s["node_id"] == winner_id)
        accept_msg = make_accept(f"nash-{self.task_id}", winner_id,
                                 self.task_id, conv_id)
        ray.get(resource_agents[w_idx].receive_accept.remote(accept_msg))

        # REJECT ai perdenti
        for loser in all_proposals:
            if loser.sender_id == winner_id:
                continue
            l_idx = next(i for i, s in enumerate(states)
                         if s["node_id"] == loser.sender_id)
            resource_agents[l_idx].receive_reject.remote(
                make_reject(f"nash-{self.task_id}", loser.sender_id,
                            self.task_id, conv_id, "better_offer_selected")
            )

        t_end = time.time()
        self.status                     = "placed"
        self.placed_on                  = winner_id
        self.placement_latency_ms       = (t_end - t_start) * 1000
        self.nash_rounds_to_convergence = rounds_used

        winner_utility = self._compute_agent_utility(winner_resp.offer, current_req)

        result = {
            "task_id":                 self.task_id,
            "status":                  "placed",
            "placed_on":               winner_id,
            "policy":                  self.policy.value,
            "score":                   winner_resp.offer.score,
            "estimated_latency_ms":    winner_resp.offer.estimated_latency_ms,
            "energy_score":            winner_resp.offer.energy_cost_score,
            "placement_latency_ms":    self.placement_latency_ms,
            "proposals_received":      len(all_proposals),
            "nash_converged":          nash_converged,
            "nash_rounds":             rounds_used,
            "nash_winner_utility":     winner_utility,
            "original_max_latency_ms": self.requirements.max_latency_ms,
            "final_max_latency_ms":    current_req.max_latency_ms,
            "sla_ok_original": (winner_resp.offer.estimated_latency_ms
                                <= self.requirements.max_latency_ms),
        }

        print(f"[NashAgent:{self.task_id}] Piazzato su {winner_id} in "
              f"{self.placement_latency_ms:.1f}ms | "
              f"Nash={'OK' if nash_converged else 'fallback'} "
              f"@ round {rounds_used} | utility={winner_utility:.3f}")
        return result

    def get_status(self) -> dict:
        return {
            "task_id":                    self.task_id,
            "status":                     self.status,
            "placed_on":                  self.placed_on,
            "placement_latency_ms":       self.placement_latency_ms,
            "policy":                     self.policy.value,
            "nash_rounds_to_convergence": self.nash_rounds_to_convergence,
            "nash_log":                   self.nash_log,
        }