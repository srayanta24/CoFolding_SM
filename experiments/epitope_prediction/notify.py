#!/usr/bin/env python3
"""WhatsApp notifications for milestone events during epitope-prediction training, via
the Hermes agent already installed/configured on this machine (~/.local/bin/hermes,
`send` subcommand — "Pipe text from any shell script to any messaging platform Hermes
is already configured for").

Deliberately used sparingly, only at real milestones (training start, a crash that
needs a restart, final completion) -- not on every routine progress check. Best-effort:
a failed notification never raises, so it can't take down a training run.

Usage (as a library, called around the actual training commands):
    from notify import notify
    notify("Model A ensemble training started (5 members, ~N structures)")
    ...
    notify("Model A training FAILED: <error>, needs restart")

Usage (CLI, for a quick manual test):
    python3 experiments/epitope_prediction/notify.py "test message"
"""

import subprocess
import sys

HERMES_BIN = "/home/srayanta/.local/bin/hermes"
WHATSAPP_TARGET = "whatsapp"  # home channel; re-checked at call time via --list


def whatsapp_target_available() -> bool:
    result = subprocess.run(
        [HERMES_BIN, "send", "--list", "whatsapp"],
        capture_output=True, text=True, timeout=15,
    )
    return "no targets found" not in result.stdout and "no targets found" not in result.stderr


def notify(message: str, subject: str | None = None) -> bool:
    """Best-effort WhatsApp send. Returns False (never raises) on any failure --
    training milestones matter more than notification delivery, so a Hermes/WhatsApp
    hiccup must never crash or block the actual training run."""
    cmd = [HERMES_BIN, "send", "--to", WHATSAPP_TARGET, "-q"]
    if subject:
        cmd += ["--subject", subject]
    cmd.append(message)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"[notify] WhatsApp send failed (exit {result.returncode}): {result.stderr.strip()}", file=sys.stderr)
            return False
        return True
    except Exception as e:
        print(f"[notify] WhatsApp send raised {type(e).__name__}: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "Hermes notify test from epitope_prediction/notify.py"
    if not whatsapp_target_available():
        print("[notify] no WhatsApp target discovered yet -- message the paired number from your phone first", file=sys.stderr)
        sys.exit(1)
    ok = notify(msg)
    print("sent" if ok else "failed")
