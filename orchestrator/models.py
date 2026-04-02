"""
Data models for the Nion Orchestration Engine.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class L2Domain(str, Enum):
    TRACKING_EXECUTION = "TRACKING_EXECUTION"
    COMMUNICATION_COLLABORATION = "COMMUNICATION_COLLABORATION"
    LEARNING_IMPROVEMENT = "LEARNING_IMPROVEMENT"


class CrossCuttingAgent(str, Enum):
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    EVALUATION = "evaluation"


# L3 agents per L2 domain
L3_AGENTS_BY_DOMAIN = {
    L2Domain.TRACKING_EXECUTION: [
        "action_item_extraction",
        "action_item_validation",
        "action_item_tracking",
        "risk_extraction",
        "risk_tracking",
        "issue_extraction",
        "issue_tracking",
        "decision_extraction",
        "decision_tracking",
    ],
    L2Domain.COMMUNICATION_COLLABORATION: [
        "qna",
        "report_generation",
        "message_delivery",
        "meeting_attendance",
    ],
    L2Domain.LEARNING_IMPROVEMENT: [
        "instruction_led_learning",
    ],
}

CROSS_CUTTING_AGENTS = [agent.value for agent in CrossCuttingAgent]


@dataclass
class Message:
    message_id: str
    source: str
    sender_name: str
    sender_role: str
    content: str
    project: Optional[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        if not isinstance(data, dict):
            raise ValueError(f"Message data must be a dict, got {type(data).__name__}")

        sender = data.get("sender") or {}
        if not isinstance(sender, dict):
            sender = {}

        # Normalise project: treat null, empty string, whitespace-only as None
        raw_project = data.get("project")
        project = str(raw_project).strip() if raw_project and str(raw_project).strip() else None

        content = data.get("content") or ""
        if not str(content).strip():
            content = "(no content provided)"

        return cls(
            message_id=str(data.get("message_id") or "MSG-???").strip(),
            source=str(data.get("source") or "unknown").strip().lower(),
            sender_name=str(sender.get("name") or "Unknown").strip(),
            sender_role=str(sender.get("role") or "Unknown").strip(),
            content=str(content).strip(),
            project=project,
        )


@dataclass
class PlannedTask:
    task_id: str
    target: str          # e.g. "L2:TRACKING_EXECUTION" or "L3:knowledge_retrieval"
    purpose: str
    depends_on: List[str] = field(default_factory=list)
    is_cross_cutting: bool = False


@dataclass
class L3Result:
    task_id: str
    agent_name: str
    status: str = "COMPLETED"
    output_lines: List[str] = field(default_factory=list)


@dataclass
class L2Result:
    task_id: str
    domain: str
    l3_results: List[L3Result] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    message: Message
    planned_tasks: List[PlannedTask] = field(default_factory=list)
    l2_results: List[L2Result] = field(default_factory=list)
    cross_cutting_results: List[L3Result] = field(default_factory=list)
