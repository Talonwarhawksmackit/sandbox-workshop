"""
agents/node.py — Local-First Hive Node Client
===============================================
Hybrid orchestration wrapper supporting both a local Ollama backend and
an optional external OpenAI-compatible API endpoint.

Fallback Architecture
---------------------
  1. If an external API is configured (url + key + model), try it first.
  2. If it is *un*configured, unreachable, rate-limited, or errors out,
     catch the exception, log a warning, and seamlessly fall back to
     the local ``ollama`` Python API using **phi3:mini**.

Default model  : phi3:mini   (local Ollama)
Temperature    : 0.3         (deterministic, focused output)
top_p          : 0.9         (structural integrity)

Personas
--------
  • Straight-Line Thinker — pragmatic systems engineer.
  • Factory Overseer     — architectural auditor returning JSON verdicts.
"""

import logging
from enum import Enum
from typing import Optional

import ollama

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model defaults (local Ollama)
# ---------------------------------------------------------------------------
LOCAL_MODEL: str = "phi3:mini"
DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_TOP_P: float = 0.9


# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

class Persona(str, Enum):
    """Operational personas available for Hive Node execution."""

    STRAIGHT_LINE_THINKER = "straight_line_thinker"
    FACTORY_OVERSEER = "factory_overseer"


PERSONA_SYSTEM_MESSAGES: dict[Persona, str] = {
    Persona.STRAIGHT_LINE_THINKER: (
        "You are the Straight-Line Thinker — a pragmatic, by-the-book systems "
        "engineer.  Your sole objective is to receive a set of code fragments "
        "and a target goal, then fuse them into a single, clean, runnable "
        "Python module.  Focus exclusively on logical flow, correct syntax, "
        "and minimal but effective glue code.  Do NOT add unnecessary comments "
        "or creative flourishes.  Output ONLY the final Python source code."
    ),
    Persona.FACTORY_OVERSEER: (
        "You are the Factory Overseer — an architectural auditor enforcing "
        "strict system standards.  You will receive a block of assembled "
        "Python code.  Analyse it for correctness, safety, and structural "
        "integrity.  Return your verdict as a JSON object with EXACTLY this "
        'format and nothing else:\n\n{"vote": "Approve", "reason": "..."}\n\n'
        "or\n\n"
        '{"vote": "Reject", "reason": "..."}\n\n'
        "Do NOT wrap the JSON in markdown code fences.  Do NOT add any "
        "commentary outside the JSON object."
    ),
}


# ---------------------------------------------------------------------------
# External API configuration (dataclass-style for clarity)
# ---------------------------------------------------------------------------

class ExternalAPIConfig:
    """Optional configuration for an OpenAI-compatible external endpoint.

    Parameters
    ----------
    base_url : str
        The API base URL (e.g. ``"https://api.openai.com/v1"``).
    api_key : str
        Bearer token / API key.
    model : str
        Model identifier on the remote service.
    """

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model

    def __repr__(self) -> str:
        return (
            f"ExternalAPIConfig(base_url={self.base_url!r}, "
            f"model={self.model!r}, api_key=<redacted>)"
        )


# ---------------------------------------------------------------------------
# Hive Node Client
# ---------------------------------------------------------------------------

class HiveNode:
    """Hybrid inference node with local-first fallback.

    Parameters
    ----------
    local_model : str
        Ollama model tag for local execution (default ``phi3:mini``).
    temperature : float
        Sampling temperature (default ``0.3``).
    top_p : float
        Nucleus sampling threshold (default ``0.9``).
    external_config : ExternalAPIConfig | None
        If provided, the node will *attempt* external routing first
        before falling back to the local Ollama backend.
    """

    def __init__(
        self,
        local_model: str = LOCAL_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        external_config: Optional[ExternalAPIConfig] = None,
    ) -> None:
        self.local_model = local_model
        self.temperature = temperature
        self.top_p = top_p
        self.external_config = external_config

        if self.external_config:
            logger.info(
                "HiveNode initialised  ➜  external=%s  local_fallback=%s  "
                "temp=%.2f  top_p=%.2f",
                self.external_config.model,
                self.local_model,
                self.temperature,
                self.top_p,
            )
        else:
            logger.info(
                "HiveNode initialised  ➜  local_only=%s  temp=%.2f  top_p=%.2f",
                self.local_model,
                self.temperature,
                self.top_p,
            )

    # ------------------------------------------------------------------
    # External API attempt
    # ------------------------------------------------------------------

    def _try_external(self, system_msg: str, prompt: str) -> Optional[str]:
        """Attempt generation via the external OpenAI-compatible API.

        Returns the response text on success, or ``None`` on any failure
        (missing config, network error, rate limit, etc.).
        """
        if self.external_config is None:
            return None

        try:
            from openai import OpenAI, APIConnectionError, RateLimitError  # type: ignore[import-untyped]

            client = OpenAI(
                base_url=self.external_config.base_url,
                api_key=self.external_config.api_key,
            )

            response = client.chat.completions.create(
                model=self.external_config.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
            )
            content = response.choices[0].message.content or ""
            logger.info(
                "External API success  ➜  %d chars from %s",
                len(content),
                self.external_config.model,
            )
            return content

        except ImportError:
            logger.warning(
                "openai package not installed — skipping external route."
            )
            return None

        except Exception as exc:  # noqa: BLE001 — intentionally broad
            logger.warning(
                "External API failed (%s: %s) — falling back to local Ollama.",
                type(exc).__name__,
                exc,
            )
            return None

    # ------------------------------------------------------------------
    # Local Ollama execution
    # ------------------------------------------------------------------

    def _run_local(self, system_msg: str, prompt: str) -> str:
        """Execute generation via the local Ollama API (always available)."""
        logger.info("Generating locally with model [%s] …", self.local_model)

        response = ollama.chat(
            model=self.local_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        )

        output: str = response["message"]["content"]
        logger.info(
            "Local generation complete  ➜  %d chars returned.",
            len(output),
        )
        return output

    # ------------------------------------------------------------------
    # Public generation method
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        persona: Persona,
        system_override: Optional[str] = None,
    ) -> str:
        """Send a prompt through the hybrid pipeline and return the response.

        Routing order:
          1. External API (if configured) → on success, return immediately.
          2. Local Ollama (always) → guaranteed offline fallback.

        Parameters
        ----------
        prompt : str
            The user-level prompt / task description.
        persona : Persona
            Which operational persona to activate (sets system message).
        system_override : str, optional
            If provided, replaces the persona's default system message.

        Returns
        -------
        str
            Raw text response from whichever backend succeeded.
        """
        system_msg = system_override or PERSONA_SYSTEM_MESSAGES[persona]

        logger.info(
            "HiveNode.generate()  ➜  persona=[%s]  external=%s",
            persona.value,
            "yes" if self.external_config else "no",
        )

        # --- Try external first -----------------------------------------------
        external_result = self._try_external(system_msg, prompt)
        if external_result is not None:
            return external_result

        # --- Fallback to local Ollama -----------------------------------------
        return self._run_local(system_msg, prompt)
