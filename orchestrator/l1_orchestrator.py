"""
L1 Orchestrator - Ingests messages, reasons about intent,
and generates a task plan delegating to L2 domains and cross-cutting agents.

L1 VISIBILITY: Can only see L2 domains + Cross-Cutting agents.
L1 CANNOT directly address individual L3 agents (except cross-cutting).
"""
import json
import re
import time
import logging
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .models import (
    Message, PlannedTask, L2Domain, CROSS_CUTTING_AGENTS
)

logger = logging.getLogger(__name__)

# All valid targets L1 is allowed to produce — enforced at parse time
VALID_L1_TARGETS = {
    "L2:TRACKING_EXECUTION",
    "L2:COMMUNICATION_COLLABORATION",
    "L2:LEARNING_IMPROVEMENT",
    "L3:knowledge_retrieval",
    "L3:evaluation",
}

# Minimal fallback plan when LLM fails entirely
FALLBACK_PLAN = [
    PlannedTask(
        task_id="TASK-001",
        target="L3:knowledge_retrieval",
        purpose="Retrieve project context (fallback plan)",
        depends_on=[],
        is_cross_cutting=True,
    ),
    PlannedTask(
        task_id="TASK-002",
        target="L2:TRACKING_EXECUTION",
        purpose="Extract action items, risks, issues from message (fallback plan)",
        depends_on=[],
        is_cross_cutting=False,
    ),
    PlannedTask(
        task_id="TASK-003",
        target="L2:COMMUNICATION_COLLABORATION",
        purpose="Formulate and deliver response (fallback plan)",
        depends_on=["TASK-001", "TASK-002"],
        is_cross_cutting=False,
    ),
    PlannedTask(
        task_id="TASK-004",
        target="L3:evaluation",
        purpose="Evaluate response before delivery (fallback plan)",
        depends_on=["TASK-003"],
        is_cross_cutting=True,
    ),
]

L1_SYSTEM_PROMPT = """You are the L1 Orchestrator of the Nion AI Program Manager system.

Your job: Analyze an incoming message and produce a structured task plan.

VISIBILITY RULES (STRICT):
- You can ONLY delegate to:
  1. L2 domains: TRACKING_EXECUTION, COMMUNICATION_COLLABORATION, LEARNING_IMPROVEMENT
  2. Cross-cutting L3 agents: knowledge_retrieval, evaluation
- You CANNOT directly delegate to L3 agents inside L2 domains
- L2 domains will internally coordinate their own L3 agents

AVAILABLE TARGETS (use EXACTLY these strings, nothing else):
- L2:TRACKING_EXECUTION — for extracting/tracking action items, risks, issues, decisions
- L2:COMMUNICATION_COLLABORATION — for Q&A, reports, message delivery, meeting transcripts
- L2:LEARNING_IMPROVEMENT — for learning from explicit instructions/SOPs
- L3:knowledge_retrieval — cross-cutting: retrieves project context, stakeholder info
- L3:evaluation — cross-cutting: validates outputs before delivery

PLANNING RULES:
1. Always start with L3:knowledge_retrieval if project context is needed
2. Use L2:TRACKING_EXECUTION when the message contains action items, risks, issues, or decisions
3. Use L2:COMMUNICATION_COLLABORATION for answering questions, generating reports, or sending responses
4. Always include L3:evaluation before any final delivery task
5. Use L2:LEARNING_IMPROVEMENT only if there are explicit instructions or SOPs in the message
6. Express dependencies as a list of task ID strings, e.g. ["TASK-001", "TASK-002"]
7. Number tasks sequentially: TASK-001, TASK-002, etc.

OUTPUT FORMAT — respond ONLY with a raw JSON object. No markdown, no explanation, no code fences:
{
  "tasks": [
    {
      "task_id": "TASK-001",
      "target": "L3:knowledge_retrieval",
      "purpose": "Retrieve project context and stakeholder info",
      "depends_on": [],
      "is_cross_cutting": true
    },
    {
      "task_id": "TASK-002",
      "target": "L2:TRACKING_EXECUTION",
      "purpose": "Extract action items from message",
      "depends_on": [],
      "is_cross_cutting": false
    }
  ]
}
"""


class L1Orchestrator:
    def __init__(self, llm: ChatGoogleGenerativeAI, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    def plan(self, message: Message) -> List[PlannedTask]:
        """
        Analyze the message and produce a validated task plan.
        L1 reasoning: ingestion → intent analysis → planning → delegation.

        Retries up to max_retries times on parse/validation failure.
        Falls back to a safe default plan if all retries are exhausted.
        """
        user_prompt = f"""Analyze this incoming message and create a task plan:

Message ID: {message.message_id}
Source: {message.source}
Sender: {message.sender_name} ({message.sender_role})
Project: {message.project or 'NOT SPECIFIED'}
Content: {message.content}

Create a complete orchestration plan. Be thorough — consider what extraction,
retrieval, processing, communication, and validation steps are needed.
Remember: output ONLY raw JSON, no markdown fences.
"""
        lc_messages = [
            SystemMessage(content=L1_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke(lc_messages)
                raw = response.content.strip()

                # ── Strip markdown code fences ────────────────────
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()

                # ── Parse JSON ────────────────────────────────────
                data = json.loads(raw)

                if not isinstance(data, dict) or "tasks" not in data:
                    raise ValueError("Response JSON missing 'tasks' key")

                tasks_raw = data["tasks"]
                if not isinstance(tasks_raw, list) or len(tasks_raw) == 0:
                    raise ValueError("'tasks' must be a non-empty list")

                # ── Build and validate each task ──────────────────
                planned_tasks: List[PlannedTask] = []
                seen_ids: set = set()

                for i, t in enumerate(tasks_raw):
                    if not isinstance(t, dict):
                        raise ValueError(f"Task at index {i} is not a dict")

                    task_id = str(t.get("task_id", f"TASK-{i+1:03d}")).strip()
                    target = str(t.get("target", "")).strip()
                    purpose = str(t.get("purpose", "No purpose specified")).strip()
                    depends_on_raw = t.get("depends_on", [])
                    is_cross_cutting = bool(t.get("is_cross_cutting", False))

                    # Validate target against allowed L1 visibility
                    if target not in VALID_L1_TARGETS:
                        logger.warning(
                            f"[L1] Task {task_id} has invalid target '{target}' "
                            f"— remapping to L2:COMMUNICATION_COLLABORATION"
                        )
                        target = "L2:COMMUNICATION_COLLABORATION"
                        is_cross_cutting = False

                    # Auto-set is_cross_cutting based on target
                    if target.startswith("L3:"):
                        is_cross_cutting = True

                    # Validate depends_on is a list of strings
                    if not isinstance(depends_on_raw, list):
                        depends_on_raw = []
                    depends_on = [str(d).strip() for d in depends_on_raw if d]

                    # Deduplicate task IDs
                    if task_id in seen_ids:
                        task_id = f"{task_id}-DUP{i}"
                    seen_ids.add(task_id)

                    planned_tasks.append(
                        PlannedTask(
                            task_id=task_id,
                            target=target,
                            purpose=purpose,
                            depends_on=depends_on,
                            is_cross_cutting=is_cross_cutting,
                        )
                    )

                logger.info(f"[L1] Plan generated: {len(planned_tasks)} tasks (attempt {attempt})")
                return planned_tasks

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                last_error = e
                logger.warning(f"[L1] Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)  # exponential-ish backoff

        # All retries exhausted — use safe fallback plan
        logger.error(
            f"[L1] All {self.max_retries} attempts failed (last: {last_error}). "
            "Using fallback plan."
        )
        return FALLBACK_PLAN
