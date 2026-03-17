"""Configuration loading and validation.

Loads config.yaml into typed dataclasses. API keys come from .env.
All defaults are explicit — nothing is silently assumed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Config sections
# ---------------------------------------------------------------------------

@dataclass
class AssistantConfig:
    id: str
    description: str
    name: str = ""
    dynamic_variables: dict = None  # ElevenLabs: injected into simulate-conversation

    def __post_init__(self):
        if self.dynamic_variables is None:
            self.dynamic_variables = {}


@dataclass
class ScoringConfig:
    should_weight: float = 0.50
    should_not_weight: float = 0.35
    latency_weight: float = 0.15
    latency_threshold_ms: float = 3000.0

    def formula_str(self) -> str:
        """Human-readable scoring formula for LLM prompts."""
        return (
            f"{self.should_weight:.2f} * should_score + "
            f"{self.should_not_weight:.2f} * should_not_score + "
            f"{self.latency_weight:.2f} * latency_score"
        )


@dataclass
class AutoresearchConfig:
    eval_scenarios: int = 8
    improvement_threshold: float = 0.005
    max_experiments: int = 0       # 0 = unlimited


@dataclass
class PipelineConfig:
    attack_rounds: int = 2
    verify_rounds: int = 2
    scenarios_per_round: int = 5
    top_k_elites: int = 2


@dataclass
class ConversationConfig:
    max_turns: int = 12


@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-20250514"
    max_retries: int = 5
    timeout: int = 120


@dataclass
class OutputConfig:
    dir: str = "results"
    save_transcripts: bool = True
    graphs: bool = True


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    assistant: AssistantConfig
    scoring: ScoringConfig
    autoresearch: AutoresearchConfig
    pipeline: PipelineConfig
    conversation: ConversationConfig
    llm: LLMConfig
    output: OutputConfig
    provider: str = "vapi"         # "vapi", "smallest", or "elevenlabs"
    anthropic_api_key: str = ""
    vapi_api_key: str = ""
    smallest_api_key: str = ""
    elevenlabs_api_key: str = ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | None = None) -> Config:
    """Load and validate config from YAML + environment.

    Raises ValueError on missing required fields or invalid values.
    """
    load_dotenv(ROOT / ".env", override=True)

    cfg_path = Path(path) if path else ROOT / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path) as f:
        raw = yaml.safe_load(f) or {}

    # --- Provider ---
    provider = raw.get("provider", "vapi")
    if provider not in ("vapi", "smallest", "elevenlabs"):
        raise ValueError(f"Unknown provider: {provider}. Must be 'vapi', 'smallest', or 'elevenlabs'.")

    # --- API keys (from env only, never from YAML) ---
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    vapi_key = os.environ.get("VAPI_API_KEY", "")
    smallest_key = os.environ.get("SMALLEST_API_KEY", "")
    elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY", "")

    if not anthropic_key:
        raise ValueError("ANTHROPIC_API_KEY not set in .env or environment")
    if provider == "vapi" and not vapi_key:
        raise ValueError("VAPI_API_KEY not set in .env or environment")
    if provider == "smallest" and not smallest_key:
        raise ValueError("SMALLEST_API_KEY not set in .env or environment")
    if provider == "elevenlabs" and not elevenlabs_key:
        raise ValueError("ELEVENLABS_API_KEY not set in .env or environment")

    # --- Assistant (required) ---
    ast = raw.get("assistant", {})
    if not ast.get("id"):
        raise ValueError("assistant.id is required in config.yaml")
    if not ast.get("description"):
        raise ValueError("assistant.description is required in config.yaml")

    # --- Scoring (with validation) ---
    sc = raw.get("scoring", {})
    scoring = ScoringConfig(
        should_weight=sc.get("should_weight", 0.50),
        should_not_weight=sc.get("should_not_weight", 0.35),
        latency_weight=sc.get("latency_weight", 0.15),
        latency_threshold_ms=sc.get("latency_threshold_ms", 3000.0),
    )
    weight_sum = scoring.should_weight + scoring.should_not_weight + scoring.latency_weight
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(
            f"Scoring weights must sum to 1.0, got {weight_sum:.2f} "
            f"({scoring.should_weight} + {scoring.should_not_weight} + {scoring.latency_weight})"
        )

    # --- Optional sections with defaults ---
    ar = raw.get("autoresearch", {})
    pl = raw.get("pipeline", {})
    cv = raw.get("conversation", {})
    lm = raw.get("llm", {})
    out = raw.get("output", {})

    return Config(
        assistant=AssistantConfig(
            id=ast["id"],
            description=ast["description"],
            name=ast.get("name", ""),
            dynamic_variables=ast.get("dynamic_variables", {}),
        ),
        scoring=scoring,
        autoresearch=AutoresearchConfig(
            eval_scenarios=ar.get("eval_scenarios", 8),
            improvement_threshold=ar.get("improvement_threshold", 0.005),
            max_experiments=ar.get("max_experiments", 0),
        ),
        pipeline=PipelineConfig(
            attack_rounds=pl.get("attack_rounds", 2),
            verify_rounds=pl.get("verify_rounds", 2),
            scenarios_per_round=pl.get("scenarios_per_round", 5),
            top_k_elites=pl.get("top_k_elites", 2),
        ),
        conversation=ConversationConfig(
            max_turns=cv.get("max_turns", 12),
        ),
        llm=LLMConfig(
            model=lm.get("model", "claude-sonnet-4-20250514"),
            max_retries=lm.get("max_retries", 5),
            timeout=lm.get("timeout", 120),
        ),
        output=OutputConfig(
            dir=str(ROOT / out.get("dir", "results")),
            save_transcripts=out.get("save_transcripts", True),
            graphs=out.get("graphs", True),
        ),
        provider=provider,
        anthropic_api_key=anthropic_key,
        vapi_api_key=vapi_key,
        smallest_api_key=smallest_key,
        elevenlabs_api_key=elevenlabs_key,
    )
