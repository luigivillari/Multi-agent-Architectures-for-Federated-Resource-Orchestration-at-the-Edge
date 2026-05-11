"""
Definisce tutti i tipi di messaggio scambiati tra ResourceAgent e TaskAgent.

Flusso di negoziazione:
    TaskAgent  --[CFP]-->          ResourceAgent  (broadcast)
    ResourceAgent --[PROPOSE]-->   TaskAgent      (se ha risorse)
    ResourceAgent --[REJECT]-->    TaskAgent      (se non ha risorse)
    TaskAgent  --[ACCEPT]-->       ResourceAgent  (al vincitore)
    TaskAgent  --[REJECT]-->       ResourceAgent  (agli altri)
    ResourceAgent --[COUNTER_OFFER]--> TaskAgent  (negoziazione iterativa)
    ResourceAgent --[INFORM_DONE]--> TaskAgent    (task completato)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any
import time
import uuid


# ─────────────────────────────────────────────
# Tipi di Messaggio (A2A Grammar)
# ─────────────────────────────────────────────

class MessageType(str, Enum):
    CFP           = "CFP"           # Call for Proposals: TaskAgent -> tutti i ResourceAgent
    PROPOSE       = "PROPOSE"       # Offerta: ResourceAgent -> TaskAgent
    ACCEPT        = "ACCEPT"        # Accettazione: TaskAgent -> ResourceAgent vincitore
    REJECT        = "REJECT"        # Rifiuto: TaskAgent -> ResourceAgent non vincitori
                                    #       oppure ResourceAgent -> TaskAgent (non ha risorse)
    COUNTER_OFFER = "COUNTER_OFFER" # Contro-offerta: ResourceAgent -> TaskAgent (negoziazione)
    INFORM_DONE   = "INFORM_DONE"   # Notifica completamento: ResourceAgent -> TaskAgent
    WITHDRAW      = "WITHDRAW"      # Ritiro offerta: ResourceAgent -> TaskAgent


# ─────────────────────────────────────────────
# Strutture Dati dei Messaggi
# ─────────────────────────────────────────────

@dataclass
class TaskRequirements:
    """
    Requisiti di un task — payload del messaggio CFP.
    """
    cpu_cores: float          
    memory_mb: float          
    max_latency_ms: float     
    duration_sec: float      
    priority: int = 1         
    task_type: str = "generic" # "inference", "streaming", "batch", "generic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceOffer:
    """
    Offerta di un ResourceAgent — payload del messaggio PROPOSE o COUNTER_OFFER.
    Il ResourceAgent descrive cosa può offrire e a quale "costo" (scoring).
    """
    node_id: str
    available_cpu: float      # CPU effettivamente disponibile
    available_memory_mb: float
    estimated_latency_ms: float
    energy_cost_score: float  
    score: float = 0.0        # score calcolato dalla policy (più alto = migliore)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class A2AMessage:
    """
    Struttura base di un messaggio A2A.
    Ogni scambio tra agenti è un'istanza di questa classe.

    """
    msg_type:     MessageType
    sender_id:    str                      # ID dell'agente mittente
    receiver_id:  str                      # ID dell'agente destinatario
    conversation_id: str                   # UUID della conversazione di negoziazione
    timestamp:    float = field(default_factory=time.time)
    msg_id:       str   = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Payload opzionale 
    task_id:      Optional[str] = None
    requirements: Optional[TaskRequirements] = None
    offer:        Optional[ResourceOffer] = None
    reason:       Optional[str] = None     # motivo del rifiuto
    metadata:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "msg_type": self.msg_type.value,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "conversation_id": self.conversation_id,
            "timestamp": self.timestamp,
            "msg_id": self.msg_id,
            "task_id": self.task_id,
            "reason": self.reason,
            "metadata": self.metadata,
        }
        if self.requirements:
            d["requirements"] = self.requirements.to_dict()
        if self.offer:
            d["offer"] = self.offer.to_dict()
        return d

    def __repr__(self):
        return (f"A2AMessage({self.msg_type.value} | "
                f"{self.sender_id} → {self.receiver_id} | "
                f"conv={self.conversation_id[:8]} | "
                f"task={self.task_id})")


# ─────────────────────────────────────────────
#  costruzione messaggi
# ─────────────────────────────────────────────

def make_cfp(sender_id: str, task_id: str, requirements: TaskRequirements,
             receivers: str = "*") -> A2AMessage:
    """Crea un Call for Proposals"""
    return A2AMessage(
        msg_type=MessageType.CFP,
        sender_id=sender_id,
        receiver_id=receivers,
        conversation_id=str(uuid.uuid4())[:12],
        task_id=task_id,
        requirements=requirements,
    )

def make_propose(sender_id: str, receiver_id: str, task_id: str,
                 conv_id: str, offer: ResourceOffer) -> A2AMessage:
    """Crea una PROPOSE"""
    return A2AMessage(
        msg_type=MessageType.PROPOSE,
        sender_id=sender_id,
        receiver_id=receiver_id,
        conversation_id=conv_id,
        task_id=task_id,
        offer=offer,
    )

def make_accept(sender_id: str, receiver_id: str,
                task_id: str, conv_id: str) -> A2AMessage:
    """Crea un ACCEPT"""
    return A2AMessage(
        msg_type=MessageType.ACCEPT,
        sender_id=sender_id,
        receiver_id=receiver_id,
        conversation_id=conv_id,
        task_id=task_id,
    )

def make_reject(sender_id: str, receiver_id: str,
                task_id: str, conv_id: str, reason: str = "") -> A2AMessage:
    """Crea un REJECT."""
    return A2AMessage(
        msg_type=MessageType.REJECT,
        sender_id=sender_id,
        receiver_id=receiver_id,
        conversation_id=conv_id,
        task_id=task_id,
        reason=reason,
    )

def make_counter_offer(sender_id: str, receiver_id: str, task_id: str,
                       conv_id: str, offer: ResourceOffer) -> A2AMessage:
    """Crea una COUNTER_OFFER"""
    return A2AMessage(
        msg_type=MessageType.COUNTER_OFFER,
        sender_id=sender_id,
        receiver_id=receiver_id,
        conversation_id=conv_id,
        task_id=task_id,
        offer=offer,
    )

def make_inform_done(sender_id: str, receiver_id: str,
                     task_id: str, conv_id: str) -> A2AMessage:
    """Crea un INFORM_DONE"""
    return A2AMessage(
        msg_type=MessageType.INFORM_DONE,
        sender_id=sender_id,
        receiver_id=receiver_id,
        conversation_id=conv_id,
        task_id=task_id,
    )


# ─────────────────────────────────────────────
# Policy di scoring 
# ─────────────────────────────────────────────

class PlacementPolicy(str, Enum):
    LATENCY_FIRST = "latency_first"   # minimizza latenza
    ENERGY_FIRST  = "energy_first"    # minimizza consumo energetico
    BALANCED      = "balanced"        # compromesso latenza + energia

def score_offer(offer: ResourceOffer, policy: PlacementPolicy) -> float:
    """
    Calcola uno score per un'offerta dato una policy.
    Score più alto = offerta migliore.
    """
    # Normalizza latency: 0ms → 1.0, 500ms → 0.0
    latency_score = max(0.0, 1.0 - offer.estimated_latency_ms / 500.0)

    # Energia invertita: 0.0 (efficiente) → 1.0, 1.0 (costoso) → 0.0
    energy_score = 1.0 - offer.energy_cost_score

    # Disponibilità risorse (CPU relativa all'offerta)
    resource_score = min(1.0, offer.available_cpu / 8.0)

    if policy == PlacementPolicy.LATENCY_FIRST:
        return 0.7 * latency_score + 0.2 * resource_score + 0.1 * energy_score
    elif policy == PlacementPolicy.ENERGY_FIRST:
        return 0.1 * latency_score + 0.2 * resource_score + 0.7 * energy_score
    else:  # BALANCED
        return 0.4 * latency_score + 0.3 * resource_score + 0.3 * energy_score
