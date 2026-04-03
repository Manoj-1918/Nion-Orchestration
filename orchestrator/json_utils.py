"""
Robust JSON extraction utility.
Handles all the ways LLMs wrap or truncate JSON in their responses.
"""
import re
import json
from typing import Any


def extract_json(raw: str) -> Any:
    """
    Aggressively extract and parse a JSON object from raw LLM output.
    Tries multiple strategies in order of preference.
    Raises json.JSONDecodeError if nothing works.
    """
    if not raw or not raw.strip():
        raise json.JSONDecodeError("Empty response from LLM", "", 0)

    original = raw
    raw = raw.strip()

    # Strategy 1: strip all ``` fences (with any language tag, any case)
    cleaned = re.sub(r"```[a-zA-Z]*", "", raw)
    cleaned = cleaned.replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: find first complete { } block using bracket counting
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start:i+1]
                    # Fix trailing commas before } or ]
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    # Strategy 3: try the original raw as-is (no cleaning)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # All strategies failed
    preview = original[:300].replace("\n", " ")
    raise json.JSONDecodeError(
        f"Could not extract JSON. Raw response preview: {preview!r}",
        original, 0
    )
