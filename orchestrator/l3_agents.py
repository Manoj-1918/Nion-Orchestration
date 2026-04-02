"""
L3 Agent Executor - Executes specific tasks and returns results.

Each L3 agent has a specific role. This module simulates their execution
using an LLM to generate realistic, context-aware outputs.
"""
import re
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .models import Message, L3Result

# Per-agent system prompts describing what each agent does
AGENT_PROMPTS = {
    # TRACKING_EXECUTION L3s
    "action_item_extraction": """You are the action_item_extraction L3 agent.
Extract action items from the message. For each action item:
- Assign an ID (AI-001, AI-002, ...)
- Infer the owner if mentioned, else use "?"
- Infer due date if mentioned, else use "?"
- Flag missing fields: [MISSING_OWNER], [MISSING_DUE_DATE]
Output 2-4 bullet points representing action items found.""",

    "action_item_validation": """You are the action_item_validation L3 agent.
Validate that extracted action items have required fields (owner, due date, description).
Report validation status for each. Flag any issues.""",

    "action_item_tracking": """You are the action_item_tracking L3 agent.
Provide a status snapshot of the action items. Show current status (OPEN/IN_PROGRESS/BLOCKED/DONE).""",

    "risk_extraction": """You are the risk_extraction L3 agent.
Extract risks from the message. For each risk:
- Assign an ID (RISK-001, RISK-002, ...)
- Assess Likelihood: LOW/MEDIUM/HIGH
- Assess Impact: LOW/MEDIUM/HIGH
Output 2-3 bullet points representing risks.""",

    "risk_tracking": """You are the risk_tracking L3 agent.
Provide a risk tracking snapshot showing risk status and mitigation steps.""",

    "issue_extraction": """You are the issue_extraction L3 agent.
Extract issues/problems from the message. For each issue:
- Assign an ID (ISS-001, ISS-002, ...)
- Assess severity: LOW/MEDIUM/HIGH/CRITICAL
- Note affected component
Output bullet points for each issue found.""",

    "issue_tracking": """You are the issue_tracking L3 agent.
Provide an issue tracking snapshot showing resolution status.""",

    "decision_extraction": """You are the decision_extraction L3 agent.
Extract decisions (pending or made) from the message. For each decision:
- Assign an ID (DEC-001, DEC-002, ...)
- Identify decision maker (or "?" if unknown)
- Status: PENDING or DECIDED
Output bullet points for each decision.""",

    "decision_tracking": """You are the decision_tracking L3 agent.
Track decisions to implementation. Show decision status and next steps.""",

    # COMMUNICATION_COLLABORATION L3s
    "qna": """You are the qna L3 agent.
Formulate a professional, gap-aware response to the sender's message.
Structure your response with:
- WHAT I KNOW: (facts available)
- WHAT I'VE LOGGED: (items tracked)
- WHAT I NEED: (gaps/missing info)
Then a 1-2 sentence summary of what can/cannot be answered.
Keep it concise and actionable.""",

    "report_generation": """You are the report_generation L3 agent.
Generate a concise status report or summary based on the message context.
Include: Executive Summary, Key Items, Status, Next Steps.""",

    "message_delivery": """You are the message_delivery L3 agent.
Confirm message delivery details:
- Channel: (email/slack/teams based on source)
- Recipient: sender's name
- CC: relevant stakeholders if identifiable
- Delivery Status: SENT
Output as bullet points.""",

    "meeting_attendance": """You are the meeting_attendance L3 agent.
Capture meeting transcript summary and generate meeting minutes.
Include: Attendees (inferred), Key Discussion Points, Action Items, Blockers.""",

    # LEARNING_IMPROVEMENT L3s
    "instruction_led_learning": """You are the instruction_led_learning L3 agent.
Extract explicit instructions, rules, or SOPs from the message.
Store them as learnable rules. Output: Rule ID, Rule Description, Source.""",

    # CROSS-CUTTING agents
    "knowledge_retrieval": """You are the knowledge_retrieval cross-cutting agent.
Retrieve relevant project context. Generate plausible project data such as:
- Project name and ID
- Release dates and milestones
- Team capacity and progress
- Key stakeholders (Engineering Manager, Tech Lead, etc.)
Make the data realistic and consistent with the project ID mentioned.""",

    "evaluation": """You are the evaluation cross-cutting agent.
Validate the response/output before delivery. Check:
- Relevance: PASS/FAIL
- Accuracy: PASS/FAIL
- Tone: PASS/FAIL
- Gaps Acknowledged: PASS/FAIL
- Result: APPROVED or NEEDS_REVISION
Output as bullet points.""",
}


class L3AgentExecutor:
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm

    def execute(
        self,
        task_id: str,
        agent_name: str,
        message: Message,
        context: str = "",
    ) -> L3Result:
        """Execute an L3 agent and return its result."""
        system_prompt = AGENT_PROMPTS.get(
            agent_name,
            f"You are the {agent_name} agent. Process the input and return relevant output as bullet points.",
        )

        user_prompt = f"""Process this message as the {agent_name} agent:

Message ID: {message.message_id}
Source: {message.source}
Sender: {message.sender_name} ({message.sender_role})
Project: {message.project or 'NOT SPECIFIED'}
Content: {message.content}

{f'Additional context:{chr(10)}{context}' if context else ''}

Respond with ONLY bullet points (starting with •), one per line. 
Be specific and realistic. 3-6 bullet points maximum.
"""
        messages_lc = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages_lc)
        raw = response.content.strip()

        # Parse bullet points
        lines = raw.split("\n")
        output_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith("•") or line.startswith("-") or line.startswith("*"):
                # Normalize to bullet
                cleaned = line.lstrip("•-* ").strip()
                if cleaned:
                    output_lines.append(cleaned)

        # Fallback: if LLM didn't use bullets, split by newline
        if not output_lines:
            output_lines = [l.strip() for l in lines if l.strip()][:6]

        return L3Result(
            task_id=task_id,
            agent_name=agent_name,
            status="COMPLETED",
            output_lines=output_lines,
        )
