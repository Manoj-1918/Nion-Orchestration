# Nion Orchestration Engine

A simplified implementation of Nion's AI orchestration system — an AI Program Manager agent that tracks action items, risks, issues, and decisions across projects using a three-tier L1 → L2 → L3 agent hierarchy.

---

## Architecture

```
Input Message (JSON)
        │
        ▼
┌───────────────────┐
│   L1 Orchestrator │  ← Ingests message, reasons about intent, creates task plan
│  (Google Gemini)  │    Visibility: L2 domains + Cross-Cutting agents only
└────────┬──────────┘
         │ delegates
    ┌────┴────────────────────────────────────┐
    │                                         │
    ▼                                         ▼
L2 Domain                           L3 Cross-Cutting Agent
(TRACKING_EXECUTION /               (knowledge_retrieval /
 COMMUNICATION_COLLABORATION /       evaluation)
 LEARNING_IMPROVEMENT)
    │
    ▼
L3 Domain Agents
(action_item_extraction, risk_extraction,
 qna, message_delivery, etc.)
```

### Three-Tier Hierarchy

| Layer | Role | Visibility |
|-------|------|-----------|
| **L1 Orchestrator** | Ingests and parses the message, reasons about intent, generates plan | L2 domains + Cross-Cutting agents only |
| **L2 Coordinator** | Receives directions from L1, coordinates L3 agents, aggregates results | Its own L3 agents + Cross-Cutting agents |
| **L3 Agent** | Executes specific tasks, returns results | None (leaf node) |

### Available Agents

**L2 Domains:**
- `TRACKING_EXECUTION` — extraction and tracking of action items, risks, issues, decisions
- `COMMUNICATION_COLLABORATION` — Q&A, reporting, message delivery, meeting attendance
- `LEARNING_IMPROVEMENT` — learning from explicit instructions and SOPs

**Cross-Cutting Agents (visible to L1 and all L2s):**
- `knowledge_retrieval` — retrieves project context, stakeholder details, historical data
- `evaluation` — validates outputs before delivery

**L3 Agents under TRACKING_EXECUTION:**
- `action_item_extraction`, `action_item_validation`, `action_item_tracking`
- `risk_extraction`, `risk_tracking`
- `issue_extraction`, `issue_tracking`
- `decision_extraction`, `decision_tracking`

**L3 Agents under COMMUNICATION_COLLABORATION:**
- `qna`, `report_generation`, `message_delivery`, `meeting_attendance`

**L3 Agents under LEARNING_IMPROVEMENT:**
- `instruction_led_learning`

---

## Setup

### Prerequisites
- Python 3.9+
- A free Google Gemini API key (no credit card required)

### Step 1: Get a Free API Key
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure API Key
```bash
cp .env.example .env
# Edit .env and paste your API key
```

Your `.env` file should look like:
```
GOOGLE_API_KEY=AIza...your_key_here
```

---

## Running the Code

### Run the default sample input (MSG-001 from the problem statement):
```bash
python main.py
```

### Run with a custom JSON file:
```bash
python main.py my_message.json
```

### Run a specific test case (1–6):
```bash
python main.py --test 1
python main.py --test 4
```

### Run all 6 test cases:
```bash
python main.py --all
```

### Save output to the `outputs/` folder:
```bash
python main.py --save
python main.py --all --save
python main.py --test 2 --save
```

---

## Input Format

```json
{
  "message_id": "MSG-001",
  "source": "email",
  "sender": {
    "name": "Sarah Chen",
    "role": "Product Manager"
  },
  "content": "Your message content here...",
  "project": "PRJ-ALPHA"
}
```

- `source`: `email`, `slack`, `meeting`, or any string
- `project`: can be `null` for ambiguous messages

---

## Output Format

```
==============================================================================
NION ORCHESTRATION MAP
==============================================================================
Message: MSG-001
From: Sarah Chen (Product Manager)
Project: PRJ-ALPHA

==============================================================================
L1 PLAN
==============================================================================
[TASK-001] → L2:TRACKING_EXECUTION
  Purpose: Extract action items from customer request

[TASK-002] → L3:knowledge_retrieval (Cross-Cutting)
  Purpose: Retrieve project context and timeline

[TASK-003] → L2:COMMUNICATION_COLLABORATION
  Purpose: Formulate gap-aware response
  Depends On: TASK-001, TASK-002

...

==============================================================================
L2/L3 EXECUTION
==============================================================================

[TASK-001] L2:TRACKING_EXECUTION
  └─▶ [TASK-001-A] L3:action_item_extraction
      Status: COMPLETED
      Output:
        • AI-001: "Evaluate real-time notifications feature" ...

...
==============================================================================
```

---

## Test Cases

| File | Scenario | Source |
|------|----------|--------|
| `test_cases/MSG-101.json` | Simple status question | Slack |
| `test_cases/MSG-102.json` | Feasibility question (new features) | Email |
| `test_cases/MSG-103.json` | Decision/recommendation request | Email |
| `test_cases/MSG-104.json` | Meeting transcript with multiple speakers | Meeting |
| `test_cases/MSG-105.json` | Urgent escalation (legal threat) | Email |
| `test_cases/MSG-106.json` | Ambiguous request, null project | Slack |

---

## Design Decisions & Assumptions

### 1. LLM Choice: Google Gemini (gemini-2.5-flash)
- Free tier with generous limits (no credit card)
- `gemini-1.5-flash` is fast, capable, and sufficient for orchestration reasoning
- Integrated via LangChain's `ChatGoogleGenerativeAI`
- We need to purchase the api key for accessing the resources needed.

### 2. L1 Reasoning is LLM-Powered
Rather than hardcoded rules, L1 uses an LLM with a carefully crafted system prompt that enforces visibility rules. This means:
- L1 genuinely reasons about message intent
- Task plans vary appropriately by message type
- Visibility rules are enforced in the prompt

### 3. L2 Coordination is Also LLM-Powered
L2 coordinators use an LLM to decide which of their L3 agents to invoke. The prompt only shows the L2's own agents — enforcing the visibility rule at the prompt level.

### 4. Context Propagation
Completed task outputs are accumulated and passed as context to downstream tasks. This allows later agents (e.g., `qna`, `evaluation`) to reason about what was already extracted.

### 5. Visibility Enforcement
- L1's system prompt lists only L2 domains and cross-cutting agents
- L2's system prompt lists only its own L3 agents
- At runtime, the coordinator validates that invoked agents belong to the domain (code-level enforcement)

### 6. Graceful Handling of Ambiguous Messages
For MSG-106 (null project, unknown sender, vague content), the system still produces a valid orchestration map — using `knowledge_retrieval` to attempt context lookup, and `qna` to generate a gap-aware response acknowledging missing information.

### 7. Temperature = 0.3
Low temperature ensures consistent, deterministic outputs while still allowing LLM reasoning flexibility.

---

## Project Structure

```
nion-orchestration/
├── main.py                      # Entry point and CLI
├── orchestrator/
│   ├── __init__.py
│   ├── models.py                # Data models (Message, PlannedTask, L2Result, L3Result)
│   ├── l1_orchestrator.py       # L1: LLM-powered planning
│   ├── l2_coordinator.py        # L2: Domain coordination
│   ├── l3_agents.py             # L3: Agent execution
│   └── engine.py                # Main engine: ties it all together + formats output
├── test_cases/                  # 6 test input JSON files
│   ├── MSG-101.json
│   ├── MSG-102.json
│   ├── MSG-103.json
│   ├── MSG-104.json
│   ├── MSG-105.json
│   └── MSG-106.json
├── outputs/                     # Generated outputs (created on --save)
├── .env                         # Your API key (not committed)
├── .env.example                 # Template
├── requirements.txt
└── README.md
```

---

## Sample Output

See the `outputs/` directory for pre-generated sample outputs for all 6 test cases.
