#!/usr/bin/env python3
"""
Nion Orchestration Engine - Main Entry Point

Usage:
    python main.py                    # runs sample input (MSG-001)
    python main.py input.json         # runs custom JSON file
    python main.py --all              # runs all 6 test cases
    python main.py --test 1           # runs test case N (1-6)
    python main.py --save             # saves output to outputs/ folder
    python main.py --verbose          # enables detailed logging
"""
import sys
import json
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Default sample input (from PDF) ──────────────────────────────────────────
SAMPLE_INPUT = {
    "message_id": "MSG-001",
    "source": "email",
    "sender": {"name": "Sarah Chen", "role": "Product Manager"},
    "content": (
        "The customer demo went great! They loved it but asked if we could add "
        "real-time notifications and a dashboard export feature. They're willing "
        "to pay 20% more and need it in the same timeline. Can we make this work?"
    ),
    "project": "PRJ-ALPHA",
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        print("\n❌  GOOGLE_API_KEY not found.")
        print("    1. Get a free key at: https://aistudio.google.com/app/apikey")
        print("    2. Copy .env.example → .env and paste your key\n")
        sys.exit(1)
    return key


def load_json_file(path: Path) -> dict:
    """Load and validate a JSON message file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(data).__name__}")
    return data


def run_message(engine, data: dict, save: bool = False) -> str:
    """Parse, run and format a single message. Returns the output string."""
    from orchestrator import Message
    message = Message.from_dict(data)
    result = engine.run(message)
    output = engine.format_output(result)
    print(output)

    if save:
        out_path = Path("outputs") / f"{message.message_id}.txt"
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"\n[Saved] → {out_path}")

    return output


def main() -> None:
    args = sys.argv[1:]
    save    = "--save"    in args; args = [a for a in args if a != "--save"]
    verbose = "--verbose" in args; args = [a for a in args if a != "--verbose"]

    setup_logging(verbose)

    try:
        api_key = get_api_key()

        # Import here so setup_logging runs first
        from orchestrator import NionOrchestrationEngine
        engine = NionOrchestrationEngine(api_key=api_key)

        # ── --all: run every test case ────────────────────────────
        if "--all" in args:
            test_dir = Path("test_cases")
            files = sorted(test_dir.glob("*.json"))
            if not files:
                print("❌  No JSON files found in test_cases/")
                sys.exit(1)
            for f in files:
                print(f"\n{'#' * 78}\n# Running: {f.name}\n{'#' * 78}")
                try:
                    data = load_json_file(f)
                    run_message(engine, data, save=save)
                except Exception as e:
                    print(f"\n❌  Failed on {f.name}: {e}\n")
            return

        # ── --test N: run specific test case by number ────────────
        if "--test" in args:
            idx_str = args[args.index("--test") + 1] if args.index("--test") + 1 < len(args) else ""
            if not idx_str.isdigit():
                print("❌  Usage: python main.py --test <number>  (e.g. --test 3)")
                sys.exit(1)
            n = int(idx_str) - 1          # convert 1-based to 0-based
            test_dir = Path("test_cases")
            files = sorted(test_dir.glob("*.json"))
            if n < 0 or n >= len(files):
                print(f"❌  Test case {idx_str} not found. Available: 1–{len(files)}")
                sys.exit(1)
            data = load_json_file(files[n])
            run_message(engine, data, save=save)
            return

        # ── positional file argument ──────────────────────────────
        non_flag = [a for a in args if not a.startswith("--")]
        if non_flag:
            data = load_json_file(Path(non_flag[0]))
            run_message(engine, data, save=save)
            return

        # ── default: sample MSG-001 ───────────────────────────────
        run_message(engine, SAMPLE_INPUT, save=save)

    except FileNotFoundError as e:
        print(f"\n❌  {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌  Input error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌  Unexpected error: {type(e).__name__}: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        else:
            print("    Run with --verbose for full traceback.")
        sys.exit(1)


if __name__ == "__main__":
    main()
