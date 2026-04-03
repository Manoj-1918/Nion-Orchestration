"""
L1 Orchestrator - Ingests messages, reasons about intent,
and generates a task plan delegating to L2 domains and cross-cutting agents.

L1 VISIBILITY: Can only see L2 domains + Cross-Cutting agents.
L1 CANNOT directly address individual L3 agents (except cross-cutting).
"""
import json
import time
import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .models import Message, PlannedTask, L2Domain, CROSS_CUTTING_AGENTS
from .json_utils import extract_json

logger = logging.getLogger(__name__)

VALID_L1_TARGETS = {
    "L2:TRACKING_EXECUTION",
    "L2:COMMUNICATION_COLLABORATION",
    "L2:LEARNING_IMPROVEMENT",
    "L3:knowledge_retrieval",
    "L3:evaluation",
}

FALLBACK_PLAN = [
    PlannedTask("TASK-001", "L3:knowledge_retrieval",
                "Retrieve project context (fallback plan)", [], True),
    PlannedTask("TASK-002", "L2:TRACKING_EXECUTION",
                "Extract action items, risks, issues from message (fallback plan)", [], False),
    PlannedTask("TASK-003", "L2:COMMUNICATION_COLLABORATION",
                "Formulate and deliver response (fallback plan)", ["TASK-001", "TASK-002"], False),
    PlannedTask("TASK-004", "L3:evaluation",
                "Evaluate response before delivery (fallback plan)", ["TASK-003"], True),
]

L1_SYSTEM_PROMPT = """You are the L1 Orchestrator of the Nion AI Program Manager system.

Your job: Analyze an incoming message and produce a structured task plan.

VISIBILITY RULES (STRICT):
- You can ONLY delegate to:
  1. L2 domains: TRACKING_EXECUTION, COMMUNICATION_COLLABORATION, LEARNING_IMPROVEMENT
  2. Cross-cutting L3 agents: knowledge_retrieval, evaluation
- You CANNOT directly delegate to L3 agents inside L2 domains

AVAILABLE TARGETS (use EXACTLY these strings, nothing else):
- L2:TRACKING_EXECUTION
- L2:COMMUNICATION_COLLABORATION
- L2:LEARNING_IMPROVEMENT
- L3:knowledge_retrieval
- L3:evaluation

PLANNING RULES:
1. Start with L3:knowledge_retrieval if project context is needed
2. Use L2:TRACKING_EXECUTION for action items, risks, issues, decisions
3. Use L2:COMMUNICATION_COLLABORATION for Q&A, reports, message delivery
4. Always include L3:evaluation before any final delivery task
5. Use L2:LEARNING_IMPROVEMENT only for explicit instructions/SOPs
6. Express dependencies as a list of task ID strings e.g. ["TASK-001", "TASK-002"]
7. Number tasks sequentially: TASK-001, TASK-002, etc.

IMPORTANT: Respond with ONLY a raw JSON object. No explanation, no markdown, no code fences.

{
  "tasks": [
    {
      "task_id": "TASK-001",
      "target": "L3:knowledge_retrieval",
      "purpose": "Retrieve project context",
      "depends_on": [],
      "is_cross_cutting": true
    }
  ]
}"""


class L1Orchestrator:
    def __init__(self, llm: ChatGoogleGenerativeAI, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def plan(self, message: Message) -> List[PlannedTask]:
        user_prompt = (
            f"Analyze this message and create a task plan:\n\n"
            f"Message ID: {message.message_id}\n"
            f"Source: {message.source}\n"
            f"Sender: {message.sender_name} ({message.sender_role})\n"
            f"Project: {message.project or 'NOT SPECIFIED'}\n"
            f"Content: {message.content}\n\n"
            f"Output ONLY a raw JSON object. No markdown, no code fences, no explanation."
        )

        lc_messages = [
            SystemMessage(content=L1_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke(lc_messages)
                raw = response.content.strip()

                # Robust JSON extraction — handles fences, prose, trailing commas
                data = extract_json(raw)

                if not isinstance(data, dict) or "tasks" not in data:
                    raise ValueError("Response JSON missing 'tasks' key")

                tasks_raw = data["tasks"]
                if not isinstance(tasks_raw, list) or len(tasks_raw) == 0:
                    raise ValueError("'tasks' must be a non-empty list")

                planned_tasks: List[PlannedTask] = []
                seen_ids: set = set()

                for i, t in enumerate(tasks_raw):
                    if not isinstance(t, dict):
                        raise ValueError(f"Task at index {i} is not a dict")

                    task_id = str(t.get("task_id", f"TASK-{i+1:03d}")).strip()
                    target  = str(t.get("target", "")).strip()
                    purpose = str(t.get("purpose", "No purpose specified")).strip()
                    depends_on_raw = t.get("depends_on", [])
                    is_cross_cutting = bool(t.get("is_cross_cutting", False))

                    if target not in VALID_L1_TARGETS:
                        logger.warning(
                            f"[L1] Task {task_id} invalid target '{target}' "
                            "— remapped to L2:COMMUNICATION_COLLABORATION"
                        )
                        target = "L2:COMMUNICATION_COLLABORATION"
                        is_cross_cutting = False

                    if target.startswith("L3:"):
                        is_cross_cutting = True

                    if not isinstance(depends_on_raw, list):
                        depends_on_raw = []
                    depends_on = [str(d).strip() for d in depends_on_raw if d]

                    if task_id in seen_ids:
                        task_id = f"{task_id}-DUP{i}"
                    seen_ids.add(task_id)

                    planned_tasks.append(PlannedTask(
                        task_id=task_id,
                        target=target,
                        purpose=purpose,
                        depends_on=depends_on,
                        is_cross_cutting=is_cross_cutting,
                    ))

                logger.info(f"[L1] Plan: {len(planned_tasks)} tasks (attempt {attempt})")
                return planned_tasks

            except Exception as e:
                last_error = e
                logger.warning(f"[L1] Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)

        logger.error(f"[L1] All attempts failed ({last_error}). Using fallback plan.")
        return FALLBACK_PLAN
