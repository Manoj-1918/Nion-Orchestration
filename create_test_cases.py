#!/usr/bin/env python3
"""
Run this once to create all test case JSON files.
    python create_test_cases.py
"""
import json
from pathlib import Path

TEST_CASES = {
    "MSG-101.json": {
        "message_id": "MSG-101",
        "source": "slack",
        "sender": {"name": "John Doe", "role": "Engineering Manager"},
        "content": "What's the status of the authentication feature?",
        "project": "PRJ-BETA"
    },
    "MSG-102.json": {
        "message_id": "MSG-102",
        "source": "email",
        "sender": {"name": "Sarah Chen", "role": "Product Manager"},
        "content": "Can we add SSO integration before the December release?",
        "project": "PRJ-ALPHA"
    },
    "MSG-103.json": {
        "message_id": "MSG-103",
        "source": "email",
        "sender": {"name": "Mike Johnson", "role": "VP Engineering"},
        "content": "Should we prioritize security fixes or the new dashboard?",
        "project": "PRJ-GAMMA"
    },
    "MSG-104.json": {
        "message_id": "MSG-104",
        "source": "meeting",
        "sender": {"name": "System", "role": "Meeting Bot"},
        "content": "Dev: I'm blocked on API integration, staging is down. QA: Found 3 critical bugs in payment flow. Designer: Mobile mockups ready by Thursday. Tech Lead: We might need to refactor the auth module.",
        "project": "PRJ-ALPHA"
    },
    "MSG-105.json": {
        "message_id": "MSG-105",
        "source": "email",
        "sender": {"name": "Lisa Wong", "role": "Customer Success Manager"},
        "content": "The client is asking why feature X promised for Q3 is still not delivered. They're threatening to escalate to legal. What happened?",
        "project": "PRJ-DELTA"
    },
    "MSG-106.json": {
        "message_id": "MSG-106",
        "source": "slack",
        "sender": {"name": "Random User", "role": "Unknown"},
        "content": "We need to speed things up",
        "project": None
    },
}

def main():
    test_dir = Path("test_cases")
    test_dir.mkdir(exist_ok=True)

    for filename, data in TEST_CASES.items():
        path = test_dir / filename
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  ✅ Created: test_cases/{filename}")

    print(f"\n✅ All {len(TEST_CASES)} test cases created in test_cases/")
    print("   Now run: python main.py --all --save")

if __name__ == "__main__":
    main()
