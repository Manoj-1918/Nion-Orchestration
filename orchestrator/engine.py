"""
Nion Orchestration Engine
Main engine that ties L1 → L2 → L3 together and formats the output.
"""
import logging
from typing import List, Dict, Set

from langchain_google_genai import ChatGoogleGenerativeAI

from .models import (
    Message, PlannedTask, OrchestrationResult,
    L2Domain, L2Result, L3Result, CROSS_CUTTING_AGENTS
)
from .l1_orchestrator import L1Orchestrator
from .l2_coordinator import L2Coordinator
from .l3_agents import L3AgentExecutor

logger = logging.getLogger(__name__)
SEPARATOR = "=" * 78


class NionOrchestrationEngine:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3,
            #max_output_tokens=8192,   # prevent truncated JSON responses
        )
        self.l1 = L1Orchestrator(self.llm)
        self.l2 = L2Coordinator(self.llm)
        self.l3 = L3AgentExecutor(self.llm)

    # ──────────────────────────────────────────────────────────────
    # Dependency resolution
    # ──────────────────────────────────────────────────────────────
    def _resolve_execution_order(self, tasks: List[PlannedTask]) -> List[PlannedTask]:
        """
        Topological sort of tasks respecting depends_on relationships.
        Tasks with no pending dependencies are executed first.
        Cycles are broken gracefully by logging a warning and continuing.
        """
        task_map: Dict[str, PlannedTask] = {t.task_id: t for t in tasks}
        completed: Set[str] = set()
        ordered: List[PlannedTask] = []
        remaining = list(tasks)

        max_passes = len(tasks) + 1  # guard against infinite loop on cycle
        passes = 0

        while remaining and passes < max_passes:
            passes += 1
            progress = False
            next_remaining = []

            for task in remaining:
                # Check all dependencies are already completed
                unmet = [d for d in task.depends_on if d not in completed]
                if not unmet:
                    ordered.append(task)
                    completed.add(task.task_id)
                    progress = True
                else:
                    next_remaining.append(task)

            remaining = next_remaining

            if not progress:
                # Cycle detected — append remaining tasks as-is and warn
                logger.warning(
                    f"[Engine] Dependency cycle detected among: "
                    f"{[t.task_id for t in remaining]}. Executing in declared order."
                )
                ordered.extend(remaining)
                break

        return ordered

    # ──────────────────────────────────────────────────────────────
    # Context builder — structured, labelled sections
    # ──────────────────────────────────────────────────────────────
    def _build_context(
        self,
        result: OrchestrationResult,
        up_to_task_id: str,
    ) -> str:
        """
        Build a structured context string from all tasks completed before
        up_to_task_id. Each section is clearly labelled by agent/domain
        so downstream LLMs can reason about it effectively.
        """
        sections: List[str] = []
        target_idx = next(
            (i for i, t in enumerate(result.planned_tasks) if t.task_id == up_to_task_id),
            len(result.planned_tasks),
        )
        completed_ids = {t.task_id for t in result.planned_tasks[:target_idx]}

        # Cross-cutting results
        for r in result.cross_cutting_results:
            if r.task_id in completed_ids and r.output_lines:
                sections.append(
                    f"[{r.task_id} | {r.agent_name}]:\n"
                    + "\n".join(f"  • {line}" for line in r.output_lines)
                )

        # L2/L3 results
        for l2r in result.l2_results:
            if l2r.task_id in completed_ids:
                for l3r in l2r.l3_results:
                    if l3r.output_lines:
                        sections.append(
                            f"[{l3r.task_id} | {l2r.domain} → {l3r.agent_name}]:\n"
                            + "\n".join(f"  • {line}" for line in l3r.output_lines)
                        )

        return "\n\n".join(sections) if sections else ""

    # ──────────────────────────────────────────────────────────────
    # Main orchestration pipeline
    # ──────────────────────────────────────────────────────────────
    def run(self, message: Message) -> OrchestrationResult:
        """Execute the full L1 → L2 → L3 orchestration pipeline."""
        result = OrchestrationResult(message=message)

        # ── L1: Plan ──────────────────────────────────────────────
        logger.info(f"[L1] Planning for {message.message_id}...")
        print(f"\n[L1] Orchestrator planning for {message.message_id}...")
        planned_tasks = self.l1.plan(message)
        result.planned_tasks = planned_tasks
        print(f"[L1] Plan: {len(planned_tasks)} tasks")

        # ── Resolve dependency order ───────────────────────────────
        execution_order = self._resolve_execution_order(planned_tasks)
        logger.info(f"[Engine] Execution order: {[t.task_id for t in execution_order]}")

        # ── L2/L3: Execute in dependency order ────────────────────
        for task in execution_order:
            target = task.target
            # Structured context from all previously completed tasks
            context = self._build_context(result, task.task_id)

            if task.is_cross_cutting or target.startswith("L3:"):
                # Cross-cutting agent — L1 delegates directly
                agent_name = target.replace("L3:", "").strip()
                print(f"  [Cross-Cutting] {task.task_id} → L3:{agent_name}")
                logger.info(f"[Engine] Executing cross-cutting: {task.task_id} → {agent_name}")

                l3_result = self.l3.execute(
                    task_id=task.task_id,
                    agent_name=agent_name,
                    message=message,
                    context=context,
                )
                result.cross_cutting_results.append(l3_result)

            elif target.startswith("L2:"):
                domain_str = target.replace("L2:", "").strip()
                try:
                    domain = L2Domain(domain_str)
                except ValueError:
                    logger.warning(f"[Engine] Unknown L2 domain '{domain_str}', skipping {task.task_id}.")
                    print(f"  [WARNING] Unknown domain '{domain_str}' for {task.task_id} — skipped.")
                    continue

                print(f"  [L2] {task.task_id} → L2:{domain.value}")
                logger.info(f"[Engine] Executing L2: {task.task_id} → {domain.value}")

                l2_result = self.l2.coordinate(
                    task=task,
                    domain=domain,
                    message=message,
                    context=context,
                )
                result.l2_results.append(l2_result)

            else:
                logger.warning(f"[Engine] Unrecognised target '{target}' for {task.task_id}, skipping.")

        return result

    def format_output(self, result: OrchestrationResult) -> str:
        """Format the orchestration result into the expected output format."""
        msg = result.message
        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────
        lines += [
            SEPARATOR,
            "NION ORCHESTRATION MAP",
            SEPARATOR,
            f"Message: {msg.message_id}",
            f"From: {msg.sender_name} ({msg.sender_role})",
            f"Project: {msg.project or 'NOT SPECIFIED'}",
            "",
        ]

        # ── L1 Plan ───────────────────────────────────────────────
        lines += [SEPARATOR, "L1 PLAN", SEPARATOR]
        for task in result.planned_tasks:
            task_line = f"[{task.task_id}] → {task.target}"
            if task.is_cross_cutting and "Cross-Cutting" not in task.target:
                task_line += " (Cross-Cutting)"
            lines.append(task_line)
            lines.append(f"  Purpose: {task.purpose}")
            if task.depends_on:
                lines.append(f"  Depends On: {', '.join(task.depends_on)}")
            lines.append("")

        # ── L2/L3 Execution ───────────────────────────────────────
        lines += [SEPARATOR, "L2/L3 EXECUTION", SEPARATOR, ""]

        # Build a lookup: task_id → result (for ordering by planned tasks)
        l2_by_task_id = {r.task_id: r for r in result.l2_results}
        l3_cc_by_task_id = {r.task_id: r for r in result.cross_cutting_results}

        for task in result.planned_tasks:
            if task.task_id in l2_by_task_id:
                l2r = l2_by_task_id[task.task_id]
                lines.append(f"[{l2r.task_id}] L2:{l2r.domain}")
                for l3r in l2r.l3_results:
                    lines.append(f"  └─▶ [{l3r.task_id}] L3:{l3r.agent_name}")
                    lines.append(f"      Status: {l3r.status}")
                    lines.append("      Output:")
                    for ol in l3r.output_lines:
                        lines.append(f"        • {ol}")
                    lines.append("")

            elif task.task_id in l3_cc_by_task_id:
                l3r = l3_cc_by_task_id[task.task_id]
                lines.append(f"[{l3r.task_id}] L3:{l3r.agent_name} (Cross-Cutting)")
                lines.append(f"  Status: {l3r.status}")
                lines.append("  Output:")
                for ol in l3r.output_lines:
                    lines.append(f"    • {ol}")
                lines.append("")

        lines.append(SEPARATOR)
        return "\n".join(lines)
