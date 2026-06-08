"""
run_workshop.py — Assembly & Developer-In-The-Loop Council
===========================================================
Top-level orchestrator for the Sandbox Workshop MVP.

Pipeline milestones
-------------------
  1. Diagnostic Phase    →  Query the Factory Scaler; halt if floors == 0.
  2. Data Prep Phase     →  Instantiate SnippetLibrary; seed if empty.
  3. Assembly Phase      →  Straight-Line Thinker fuses materials into code.
  4. Council Vote        →  Factory Overseer reviews and returns JSON verdict.
  5. Post-Evaluation     →  Regex-strip markdown fences; branch on vote.

Default target intent
---------------------
  "An interactive choice module designed for children teaching smart
   decision-making and positive moral habits."
"""

import json
import logging
import re
import sys
import textwrap

from core.library import Snippet, SnippetLibrary
from core.scaler import allowed_floors
from agents.node import HiveNode, Persona

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("workshop")

# ---------------------------------------------------------------------------
# Default target intent
# ---------------------------------------------------------------------------
DEFAULT_TARGET_GOAL: str = (
    "An interactive choice module designed for children teaching smart "
    "decision-making and positive moral habits."
)

# ---------------------------------------------------------------------------
# Seed snippets (injected when the library is empty)
# ---------------------------------------------------------------------------
SEED_SNIPPETS: list[Snippet] = [
    Snippet(
        content=textwrap.dedent("""\
            def validate_input(prompt: str, valid_options: list[str]) -> str:
                \"\"\"Repeatedly prompt the user until they enter a valid option.\"\"\"
                while True:
                    choice = input(prompt).strip().lower()
                    if choice in [v.lower() for v in valid_options]:
                        return choice
                    print(f"Invalid choice. Please pick one of: {', '.join(valid_options)}")
        """),
        language="Python",
        classification="Logic",
        domain="Education",
        metadata='{"purpose": "reusable input validation helper"}',
    ),
    Snippet(
        content=textwrap.dedent("""\
            def print_banner(title: str, width: int = 50) -> None:
                \"\"\"Print a centered banner with decorative borders.\"\"\"
                border = "=" * width
                print(border)
                print(title.center(width))
                print(border)
        """),
        language="Python",
        classification="UI",
        domain="Education",
        metadata='{"purpose": "console text layout wrapper"}',
    ),
]


# ---------------------------------------------------------------------------
# Sanitisation utilities
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(
    r"```(?:json|python|text|py)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def strip_code_fences(raw: str) -> str:
    """Remove markdown code-block fences and return the inner content.

    If no fences are detected the original string is returned unchanged.
    """
    match = _FENCE_RE.search(raw)
    if match:
        return match.group(1).strip()
    return raw.strip()


# ---------------------------------------------------------------------------
# Pipeline phases
# ---------------------------------------------------------------------------

def phase_diagnostic() -> int:
    """Phase 1 — Query the Adaptive Factory Scaler.

    Returns the number of allowed floors (0, 1, or 2).
    If 0, the caller should halt execution.
    """
    logger.info("━━━  PHASE 1 · DIAGNOSTIC  ━━━")
    floors = allowed_floors()
    if floors == 0:
        logger.warning(
            "System resources critically constrained. "
            "Workshop execution HALTED to protect stability."
        )
    else:
        logger.info("Allowed floors: %d", floors)
    return floors


def phase_data_prep(
    library: SnippetLibrary,
    language: str = "Python",
    domain: str = "Education",
    limit: int = 5,
) -> list[Snippet]:
    """Phase 2 — Retrieve building materials; seed the library if empty.

    Parameters
    ----------
    library : SnippetLibrary
        An already-connected SnippetLibrary instance.
    """
    logger.info("━━━  PHASE 2 · DATA PREP  ━━━")

    total = library.count(language=language, domain=domain)
    logger.info("Library holds %d record(s) for language=%s, domain=%s", total, language, domain)

    if total == 0:
        logger.info("Library empty — injecting %d seed snippet(s) …", len(SEED_SNIPPETS))
        for seed in SEED_SNIPPETS:
            library.insert(seed)

    snippets = library.get_random_cluster(language=language, domain=domain, limit=limit)
    logger.info("Retrieved %d building material(s) from the Snippet Library.", len(snippets))
    return snippets


def phase_assembly(node: HiveNode, snippets: list[Snippet], target_goal: str) -> str:
    """Phase 3 — Straight-Line Thinker fuses materials into a working module."""
    logger.info("━━━  PHASE 3 · ASSEMBLY  ━━━")

    materials_block = "\n\n# --- MATERIAL ---\n".join(s.content for s in snippets)

    prompt = (
        f"TARGET GOAL:\n{target_goal}\n\n"
        f"AVAILABLE MATERIALS (Python code fragments):\n"
        f"{'=' * 60}\n{materials_block}\n{'=' * 60}\n\n"
        "Fuse these materials with your own glue code into a single, "
        "complete, runnable Python script that fulfils the target goal. "
        "Output ONLY the Python source code."
    )

    raw = node.generate(prompt=prompt, persona=Persona.STRAIGHT_LINE_THINKER)
    assembled = strip_code_fences(raw)
    logger.info("Assembly produced %d characters of code.", len(assembled))
    return assembled


def phase_council_vote(node: HiveNode, assembled_code: str) -> dict:
    """Phase 4 — Factory Overseer reviews the assembled code.

    Returns a dict with keys ``vote`` (``"Approve"`` | ``"Reject"``)
    and ``reason`` (str).
    """
    logger.info("━━━  PHASE 4 · COUNCIL VOTE  ━━━")

    prompt = (
        "Review the following assembled Python module for correctness, "
        "safety, and structural integrity.  Return your verdict as a "
        "JSON object.\n\n"
        f"```python\n{assembled_code}\n```"
    )

    raw = node.generate(prompt=prompt, persona=Persona.FACTORY_OVERSEER)
    clean = strip_code_fences(raw)

    logger.info("Raw Overseer response (cleaned): %s", clean)

    try:
        verdict: dict = json.loads(clean)
    except json.JSONDecodeError:
        logger.error("Failed to parse Overseer JSON — treating as Reject.")
        verdict = {
            "vote": "Reject",
            "reason": f"Unparseable Overseer response: {clean[:200]}",
        }

    logger.info("Council verdict: %s", verdict.get("vote", "UNKNOWN"))
    return verdict


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_workshop(target_goal: str = DEFAULT_TARGET_GOAL) -> None:
    """Execute the full Sandbox Workshop pipeline end-to-end."""
    logger.info("=" * 64)
    logger.info("   SANDBOX WORKSHOP — Pipeline Initiated")
    logger.info("=" * 64)

    # ── Phase 1: Diagnostic ─────────────────────────────────────────────
    floors = phase_diagnostic()
    if floors == 0:
        print("\n⛔  Workshop HALTED — system resources too constrained.\n")
        sys.exit(0)

    # ── Phase 2: Data Prep ──────────────────────────────────────────────
    library = SnippetLibrary().connect()
    try:
        snippets = phase_data_prep(library, language="Python", domain="Education")
        if not snippets:
            logger.error("No materials available even after seeding. Aborting.")
            sys.exit(1)
    finally:
        library.close()

    # ── Initialise Hive Node (local-first, no external config) ──────────
    node = HiveNode()

    # ── Phase 3: Assembly ───────────────────────────────────────────────
    assembled_code = phase_assembly(node, snippets, target_goal)

    # ── Phase 4: Council Vote ───────────────────────────────────────────
    verdict = phase_council_vote(node, assembled_code)

    # ── Phase 5: Post-Evaluation ────────────────────────────────────────
    logger.info("━━━  PHASE 5 · POST-EVALUATION  ━━━")

    vote = verdict.get("vote", "Reject")
    reason = verdict.get("reason", "No reason provided.")

    if vote == "Approve":
        print("\n" + "=" * 64)
        print("  ✅  COUNCIL APPROVED — Compiled Module Output")
        print("=" * 64)
        print(assembled_code)
        print("=" * 64)
        logger.info("Approved module compiled and output successfully.")
    else:
        print("\n" + "=" * 64)
        print("  ❌  COUNCIL REJECTED — Post-Mortem")
        print("=" * 64)
        print(f"  Vote   : {vote}")
        print(f"  Reason : {reason}")
        print("=" * 64)
        logger.info(
            "Rejected. Post-mortem captured. "
            "No further compilation loop triggered."
        )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_workshop(
        target_goal=(
            "An interactive choice module designed for children teaching "
            "smart decision-making and positive moral habits."
        )
    )
