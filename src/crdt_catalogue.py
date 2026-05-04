"""
crdt_catalogue.py
=================
Fase 2 — Data Model: CRDT Resource Catalogue

Implementa il catalogo delle risorse edge usando strutture CRDT.
Nessun coordinatore centrale: ogni nodo mantiene la propria copia
e le merge sono sempre prive di conflitti (matematicamente garantito).

Struttura:
    ResourceCatalogue  →  G-Map { node_id → NodeSnapshot }
    NodeSnapshot       →  LWW-Register per ogni campo (cpu, ram, latency, ...)
    LWWRegister        →  (value, lamport_timestamp)

Proprietà CRDT garantite:
    - Commutativa:  merge(A, B) == merge(B, A)
    - Associativa:  merge(merge(A,B), C) == merge(A, merge(B,C))
    - Idempotente:  merge(A, A) == A
    → Quindi: qualunque ordine di sincronizzazione converge allo stesso stato.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple
import time
import copy


# ─────────────────────────────────────────────
# LWW-Register (Last Write Wins Register)
# ─────────────────────────────────────────────

@dataclass
class LWWRegister:
    """
    Un registro CRDT che risolve i conflitti tenendo il valore
    con il timestamp di Lamport più alto (Last Write Wins).

    Perché Lamport clock e non wall clock?
    I wall clock (time.time()) non sono sincronizzati tra nodi edge distanti.
    Il Lamport clock è logico: ogni nodo incrementa il proprio contatore
    ad ogni evento, garantendo un ordinamento parziale corretto.
    """
    value: any          # valore corrente
    timestamp: int      # Lamport clock al momento della scrittura
    node_id: str        # nodo che ha scritto (tie-breaking in caso di parità)

    def update(self, new_value, new_ts: int, writer_node: str) -> bool:
        """
        Aggiorna il registro se il nuovo timestamp è maggiore.
        In caso di parità, vince il node_id lessicograficamente maggiore (tie-breaking deterministico).
        Ritorna True se l'update ha avuto effetto.
        """
        if new_ts > self.timestamp:
            self.value = new_value
            self.timestamp = new_ts
            self.node_id = writer_node
            return True
        elif new_ts == self.timestamp and writer_node > self.node_id:
            self.value = new_value
            self.node_id = writer_node
            return True
        return False

    def merge(self, other: LWWRegister) -> LWWRegister:
        """Merge deterministico: vince il timestamp più alto."""
        if other.timestamp > self.timestamp:
            return copy.deepcopy(other)
        elif other.timestamp == self.timestamp and other.node_id > self.node_id:
            return copy.deepcopy(other)
        return copy.deepcopy(self)

    def __repr__(self):
        return f"LWW({self.value!r} @t={self.timestamp} by={self.node_id})"


# ─────────────────────────────────────────────
# NodeSnapshot — stato di un singolo nodo edge
# ─────────────────────────────────────────────

@dataclass
class NodeSnapshot:
    """
    Rappresenta lo stato corrente di un nodo edge.
    Ogni campo è un LWW-Register → aggiornabile in modo CRDT-safe.

    Questo è il "valore" nella G-Map del ResourceCatalogue.
    """
    node_id: str

    # Risorse disponibili (aggiornate dopo ogni allocation/release)
    cpu_available:    LWWRegister = field(default=None)  # core disponibili
    memory_mb:        LWWRegister = field(default=None)  # MB disponibili
    latency_ms:       LWWRegister = field(default=None)  # latenza stimata verso cloud (ms)
    energy_score:     LWWRegister = field(default=None)  # 0.0=efficiente, 1.0=costoso
    active_tasks:     LWWRegister = field(default=None)  # numero task attivi
    is_online:        LWWRegister = field(default=None)  # True/False

    def __post_init__(self):
        """Inizializza i LWW-Register con valori di default se sono stati non forniti."""
        ts = 0
        if self.cpu_available is None:
            self.cpu_available = LWWRegister(0.0, ts, self.node_id)
        if self.memory_mb is None:
            self.memory_mb = LWWRegister(0.0, ts, self.node_id)
        if self.latency_ms is None:
            self.latency_ms = LWWRegister(999.0, ts, self.node_id)
        if self.energy_score is None:
            self.energy_score = LWWRegister(0.5, ts, self.node_id)
        if self.active_tasks is None:
            self.active_tasks = LWWRegister(0, ts, self.node_id)
        if self.is_online is None:
            self.is_online = LWWRegister(False, ts, self.node_id)

    def merge(self, other: NodeSnapshot) -> NodeSnapshot:
        """
        Merge CRDT di due snapshot dello stesso nodo.
        Ogni campo viene mergiato indipendentemente.
        """
        assert self.node_id == other.node_id, "Merge tra nodi diversi non consentito"
        merged = NodeSnapshot(node_id=self.node_id)
        merged.cpu_available = self.cpu_available.merge(other.cpu_available)
        merged.memory_mb     = self.memory_mb.merge(other.memory_mb)
        merged.latency_ms    = self.latency_ms.merge(other.latency_ms)
        merged.energy_score  = self.energy_score.merge(other.energy_score)
        merged.active_tasks  = self.active_tasks.merge(other.active_tasks)
        merged.is_online     = self.is_online.merge(other.is_online)
        return merged

    def to_dict(self) -> dict:
        return {
            "node_id":         self.node_id,
            "cpu_available":   self.cpu_available.value,
            "memory_mb":       self.memory_mb.value,
            "latency_ms":      self.latency_ms.value,
            "energy_score":    self.energy_score.value,
            "active_tasks":    self.active_tasks.value,
            "is_online":       self.is_online.value,
        }

    def __repr__(self):
        return (f"NodeSnapshot({self.node_id}: "
                f"cpu={self.cpu_available.value:.1f} "
                f"mem={self.memory_mb.value:.0f}MB "
                f"lat={self.latency_ms.value:.0f}ms "
                f"online={self.is_online.value})")


# ─────────────────────────────────────────────
# ResourceCatalogue — G-Map CRDT
# ─────────────────────────────────────────────

class ResourceCatalogue:
    """
    Il catalogo distribuito delle risorse: una G-Map (Grow-only Map) CRDT.

    G-Map: le chiavi (node_id) possono solo essere aggiunte, mai rimosse.
    I valori (NodeSnapshot) vengono mergiati con la semantica LWW per campo.

    Ogni ResourceAgent mantiene un'istanza locale di ResourceCatalogue e
    la sincronizza periodicamente con i nodi vicini tramite gossip protocol.

    Lamport Clock:
        Ogni operazione di scrittura locale incrementa il clock logico.
        Durante il merge, il clock locale viene aggiornato al massimo tra
        i due clock (garantisce che eventi successivi al merge abbiano
        timestamp più alti di tutti gli eventi mergiati).
    """

    def __init__(self, owner_node_id: str):
        self.owner_node_id = owner_node_id
        self._entries: Dict[str, NodeSnapshot] = {}
        self._lamport_clock: int = 0
        self._version_vector: Dict[str, int] = {owner_node_id: 0}
        self._merge_history: List[Tuple[str, float]] = []  # (merged_from, timestamp)

    # ── Lamport Clock ─────────────────────────

    def _tick(self) -> int:
        """Incrementa e ritorna il Lamport clock locale."""
        self._lamport_clock += 1
        self._version_vector[self.owner_node_id] = self._lamport_clock
        return self._lamport_clock

    def _update_clock(self, received_ts: int):
        """Aggiorna il clock locale dopo aver ricevuto un messaggio (regola di Lamport)."""
        self._lamport_clock = max(self._lamport_clock, received_ts) + 1
        self._version_vector[self.owner_node_id] = self._lamport_clock

    # ── Operazioni Locali ─────────────────────

    def upsert_node(self, node_id: str,
                    cpu: float, memory_mb: float,
                    latency_ms: float, energy_score: float,
                    active_tasks: int = 0, is_online: bool = True):
        """
        Aggiunge o aggiorna un nodo nel catalogo locale.
        Chiamato dal ResourceAgent ogni volta che le sue risorse cambiano.
        """
        ts = self._tick()

        if node_id not in self._entries:
            # Primo inserimento
            snapshot = NodeSnapshot(node_id=node_id)
            self._entries[node_id] = snapshot

        snap = self._entries[node_id]
        snap.cpu_available.update(cpu, ts, self.owner_node_id)
        snap.memory_mb.update(memory_mb, ts, self.owner_node_id)
        snap.latency_ms.update(latency_ms, ts, self.owner_node_id)
        snap.energy_score.update(energy_score, ts, self.owner_node_id)
        snap.active_tasks.update(active_tasks, ts, self.owner_node_id)
        snap.is_online.update(is_online, ts, self.owner_node_id)

    def get_node(self, node_id: str) -> Optional[NodeSnapshot]:
        """Ritorna lo snapshot di un nodo (None se non trovato)."""
        return self._entries.get(node_id)

    def get_all_nodes(self) -> List[NodeSnapshot]:
        """Ritorna tutti i nodi nel catalogo."""
        return list(self._entries.values())

    def get_online_nodes(self) -> List[NodeSnapshot]:
        """Ritorna solo i nodi marcati come online."""
        return [n for n in self._entries.values() if n.is_online.value]

    def mark_offline(self, node_id: str):
        """Marca un nodo come offline (es. dopo timeout heartbeat)."""
        if node_id in self._entries:
            ts = self._tick()
            self._entries[node_id].is_online.update(False, ts, self.owner_node_id)

    # ── CRDT Merge ────────────────────────────

    def merge(self, remote_catalogue: ResourceCatalogue):
        """
        Merge CRDT con il catalogo di un altro nodo.
        Operazione commutativa, associativa e idempotente.
        Può essere chiamata in qualsiasi ordine con qualsiasi nodo.

        Questo è il cuore del consistency layer:
        dopo il merge, entrambi i nodi convergono allo stesso stato.
        """
        remote_max_ts = max(remote_catalogue._version_vector.values(), default=0)
        self._update_clock(remote_max_ts)

        for node_id, remote_snap in remote_catalogue._entries.items():
            if node_id in self._entries:
                # Merge dei due snapshot campo per campo
                self._entries[node_id] = self._entries[node_id].merge(remote_snap)
            else:
                # Nodo nuovo: aggiungi al catalogo locale
                self._entries[node_id] = copy.deepcopy(remote_snap)

        # Aggiorna il version vector
        for nid, ts in remote_catalogue._version_vector.items():
            self._version_vector[nid] = max(
                self._version_vector.get(nid, 0), ts
            )

        self._merge_history.append(
            (remote_catalogue.owner_node_id, time.time())
        )

    def state_snapshot(self) -> dict:
        """Snapshot serializzabile per il gossip protocol o per logging."""
        return {
            "owner": self.owner_node_id,
            "lamport_clock": self._lamport_clock,
            "version_vector": dict(self._version_vector),
            "nodes": {nid: snap.to_dict() for nid, snap in self._entries.items()}
        }

    def convergence_diff(self, other: ResourceCatalogue) -> List[str]:
        """
        Ritorna i node_id per cui i due cataloghi divergono ancora.
        Usato nella Fase 4 per misurare il CRDT convergence time.
        """
        diffs = []
        all_ids = set(self._entries) | set(other._entries)
        for nid in all_ids:
            a = self._entries.get(nid)
            b = other._entries.get(nid)
            if a is None or b is None:
                diffs.append(nid)
            elif a.to_dict() != b.to_dict():
                diffs.append(nid)
        return diffs

    def __repr__(self):
        online = sum(1 for n in self._entries.values() if n.is_online.value)
        return (f"ResourceCatalogue(owner={self.owner_node_id}, "
                f"nodes={len(self._entries)}, online={online}, "
                f"clock={self._lamport_clock})")
