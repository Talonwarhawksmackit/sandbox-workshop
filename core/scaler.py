"""
core/scaler.py — Adaptive Factory Scaler
=========================================
Monitors immediate host CPU load and virtual memory overhead via psutil,
then computes an integer ``allowed_floors`` boundary that governs how many
concurrent assembly pipelines the workshop may run.

Tiers
-----
  0  CRITICAL THROTTLE   CPU > 80 % **or** RAM > 85 %   → pause all work
  1  BALANCED MODE        CPU > 50 % **or** RAM > 70 %   → single-threaded
  2  HIGH PERFORMANCE     otherwise                      → full pipeline
"""

import logging

import psutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------
CPU_CRITICAL: float = 80.0
RAM_CRITICAL: float = 85.0

CPU_BALANCED: float = 50.0
RAM_BALANCED: float = 70.0

# ---------------------------------------------------------------------------
# Sampling interval (seconds) — short burst for a Ryzen-class desktop
# ---------------------------------------------------------------------------
CPU_SAMPLE_INTERVAL: float = 0.2


def allowed_floors() -> int:
    """Sample the host and return the maximum number of assembly floors.

    Returns
    -------
    int
        0 → halt / pause   (resources exhausted)
        1 → sequential     (single-threaded only)
        2 → full pipeline  (parallel processing OK)
    """
    cpu: float = psutil.cpu_percent(interval=CPU_SAMPLE_INTERVAL)
    ram: float = psutil.virtual_memory().percent

    logger.info(
        "System snapshot  ➜  CPU %.1f %%  |  RAM %.1f %%",
        cpu,
        ram,
    )

    # --- Tier 0: CRITICAL THROTTLE -------------------------------------------
    if cpu > CPU_CRITICAL or ram > RAM_CRITICAL:
        logger.warning(
            "⛔  CRITICAL THROTTLE — CPU %.1f %% / RAM %.1f %%.  "
            "Assembly paused to protect system stability.",
            cpu,
            ram,
        )
        return 0

    # --- Tier 1: BALANCED MODE ------------------------------------------------
    if cpu > CPU_BALANCED or ram > RAM_BALANCED:
        logger.info(
            "⚠️  BALANCED MODE — CPU %.1f %% / RAM %.1f %%.  "
            "Enforcing single-threaded sequential execution.",
            cpu,
            ram,
        )
        return 1

    # --- Tier 2: HIGH PERFORMANCE ---------------------------------------------
    logger.info(
        "✅  HIGH PERFORMANCE — CPU %.1f %% / RAM %.1f %%.  "
        "Full pipeline processing enabled.",
        cpu,
        ram,
    )
    return 2
