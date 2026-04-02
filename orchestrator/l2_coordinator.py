"""
L2 Coordinator - Receives delegation from L1, coordinates L3 agents,
and aggregates results.

L2 VISIBILITY: Can see its own L3 agents + Cross-Cutting agents.
L2 CANNOT see L3 agents of other domains.
"""
import json
import re
import time
import logging
from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from .models import (
    Message, PlannedTask, L2Result, L3Result,
    L2Domain, L3_AGENTS_BY_DOMAIN
)
from .l3_agents import L3AgentExecutor

logger = logging.getLogger(__name__)

# Enforced ordering rules: lower number = runs earlier
AGENT_ORDER_PRIORITY: Dict[str, int] = {
    # TRACKING_EXECUTION — extract first, validate, then track
    "action_item_extraction": 10,
    "action_item_validation": 20,
    "action_item_tracking":   30,
    "risk_extraction":        10,
    "risk_tracking":          30,
    "issue_extraction":       10,
    "issue_tracking":         30,
    "decision_extraction":    10,
    "decision_tracking":      30,
    # COMMUNICATION_COLLABORATION — formulate before delivering
    "meeting_attendance":     10,
    "qna":                    20,
    "report_generation":      25,
    "message_delivery":       40,
    # LEARNING_IMPROVEMENT
    "instruction_led_learning": 10,
}

L2_SYSTEM_PROMPT_TEMPLATE = """You are the L2 {domain} Coordinator in the Nion orchestration system.

Your job: Given a task from L1, decide which of YOUR L3 agents to invoke.

YOUR AVAILABLE L3 AGENTS (use ONLY these exact names):
{available_agents_numbered}

ORDERING RULES (critical):
- For TRACKING_EXECUTION: always run extraction agents BEFORE tracking agents
  (e.g. action_item_extraction before action_item_tracking)
- For COMMUNICATION_COLLABORATION: run qna or meeting_attendance BEFORE report_generation,
  and always run message_delivery LAST
- Never invoke an agent not in your list above

Invoke 1-3 agents relevant to the task purpose. Do not invoke every agent.

OUTPUT FORMAT - respond ONLY with raw JSON, no markdown, no explanation:
{{
  "agents_to_invoke": [
    {{"sub_task_id": "{task_id}-A", "agent": "exact_agent_name", "reason": "one line reason"}},
    {{"sub_task_id": "{task_id}-B", "agent": "exact_agent_name2", "reason": "one line reason"}}
  ]
}}
"""


class L2Coordinator:
    def __init__(self, llm: ChatGoogleGenerativeAI, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
        self.l3_executor = L3AgentExecutor(llm)

    def _sort_by_order(self, agents: List[Dict]) -> List[Dict]:
        """Sort agents by enforced execution order priority."""
        return sorted(
            agents,
            key=lambda a: AGENT_ORDER_PRIORITY.get(a.get("agent", ""), 50)
        )

    def coordinate(
        self,
        task: PlannedTask,
        domain: L2Domain,
        message: Message,
        context: str = "",
    ) -> L2Result:
        """
        L2 receives a task from L1, selects L3 agents in correct order,
        executes them, and returns aggregated results.
        """
        available_agents = L3_AGENTS_BY_DOMAIN.get(domain, [])
        numbered = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(available_agents))

        system_prompt = L2_SYSTEM_PROMPT_TEMPLATE.format(
            domain=domain.value,
            available_agents_numbered=numbered,
            task_id=task.task_id,
        )

        user_prompt = (
            f"Task from L1 Orchestrator:\n\n"
            f"Task ID: {task.task_id}\n"
            f"Purpose: {task.purpose}\n\n"
            f"Message context:\n"
            f"- Source: {message.source}\n"
            f"- Sender: {message.sender_name} ({message.sender_role})\n"
            f"- Project: {message.project or 'NOT SPECIFIED'}\n"
            f"- Content: {message.content}\n"
        )
        if context:
            user_prompt += f"\nPrior completed context:\n{context}\n"
        user_prompt += "\nSelect the right L3 agents. Output raw JSON only."

        lc_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        agents_to_invoke: List[Dict] = []
        last_error: Exception = RuntimeError("No attempts made")

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.llm.invoke(lc_messages)
                raw = response.content.strip()
                raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
                raw = re.sub(r"\s*```\s*$", "", raw, flags=re.MULTILINE)
                raw = raw.strip()

                data = json.loads(raw)

                if not isinstance(data, dict) or "agents_to_invoke" not in data:
                    raise ValueError("Missing 'agents_to_invoke' key")

                candidates = data["agents_to_invoke"]
                if not isinstance(candidates, list):
                    raise ValueError("'agents_to_invoke' must be a list")

                # Filter to only agents valid for this domain (visibility enforcement)
                valid = []
                for spec in candidates:
                    if not isinstance(spec, dict):
                        continue
                    agent_name = str(spec.get("agent", "")).strip()
                    if agent_name not in available_agents:
                        logger.warning(
                            f"[L2:{domain.value}] Agent '{agent_name}' not in domain — skipped"
                        )
                        continue
                    valid.append(spec)

                if valid:
                    agents_to_invoke = valid
                    break
                raise ValueError("No valid agents after visibility filtering")

            except (json.JSONDecodeError, ValueError, KeyError, TypeError) as e:
                last_error = e
                logger.warning(f"[L2:{domain.value}] Attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)

        # Fallback: use first available agent
        if not agents_to_invoke:
            logger.error(
                f"[L2:{domain.value}] All retries failed ({last_error}). Using fallback agent."
            )
            agents_to_invoke = [{
                "sub_task_id": f"{task.task_id}-A",
                "agent": available_agents[0],
                "reason": "fallback — LLM selection failed",
            }]

        # Enforce correct execution order
        agents_to_invoke = self._sort_by_order(agents_to_invoke)

        # Assign clean sequential sub-task IDs
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for i, spec in enumerate(agents_to_invoke):
            spec["sub_task_id"] = f"{task.task_id}-{letters[i]}"

        # Execute each L3 agent; pass growing context between them
        l3_results: List[L3Result] = []
        running_context = context

        for spec in agents_to_invoke:
            sub_task_id = spec["sub_task_id"]
            agent_name = spec["agent"]
            logger.info(f"[L2:{domain.value}] Running {sub_task_id} → {agent_name}")

            l3_result = self.l3_executor.execute(
                task_id=sub_task_id,
                agent_name=agent_name,
                message=message,
                context=running_context,
            )
            l3_results.append(l3_result)

            # Grow context for next agent in this chain
            if l3_result.output_lines:
                running_context += (
                    f"\n\n[{sub_task_id} | {agent_name}]:\n"
                    + "\n".join(f"  • {line}" for line in l3_result.output_lines)
                )

        return L2Result(
            task_id=task.task_id,
            domain=domain.value,
            l3_results=l3_results,
        )
