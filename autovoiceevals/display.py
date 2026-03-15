"""Terminal display formatting.

All print output goes through this module. Business logic modules
never call print() directly — they call functions here instead.
"""

from __future__ import annotations

from .models import EvalResult, ExperimentRecord, Metrics, Scenario


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------

def header(title: str, width: int = 70) -> None:
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def section(title: str, width: int = 70) -> None:
    print(f"\n{'━' * width}")
    print(f"  {title}")
    print(f"{'━' * width}")


def info(msg: str) -> None:
    print(f"  {msg}")


def blank() -> None:
    print()


# ---------------------------------------------------------------------------
# Eval results
# ---------------------------------------------------------------------------

def eval_result_line(result: EvalResult) -> None:
    """Print a single scenario eval result with progress bar."""
    p = "PASS" if result.passed else "FAIL"
    bar = "\u2588" * int(result.score * 20) + "\u2591" * (20 - int(result.score * 20))
    print(
        f"    [{p}] {result.score:.3f} [{bar}] "
        f"CSAT={result.csat_score} {result.persona[:40]}"
    )


def scenario_list(scenarios: list[Scenario]) -> None:
    """Print the list of generated scenarios."""
    for sc in scenarios:
        d = sc.difficulty
        print(f"    [{d}] {sc.persona_name} \u2014 {sc.attack_strategy[:55]}")


# ---------------------------------------------------------------------------
# Experiment (autoresearch)
# ---------------------------------------------------------------------------

def experiment_proposal(
    change_type: str,
    description: str,
    reasoning: str,
    old_len: int,
    new_len: int,
) -> None:
    print(f"  [{change_type}] {description[:70]}")
    if reasoning:
        print(f"  Reasoning: {reasoning[:80]}")
    print(f"  Prompt: {old_len} \u2192 {new_len} chars")


def experiment_result(
    score: float,
    delta: float,
    metrics: Metrics,
    status: str,
    best_score: float,
    prompt_len: int,
    duration: float,
) -> None:
    arrow = "\u25b2" if delta > 0 else "\u25bc" if delta < 0 else "="
    print(
        f"\n  Result: score={score:.3f} ({arrow} {abs(delta):.3f})  "
        f"csat={metrics.avg_csat:.0f}  pass={metrics.n_passed}/{metrics.n_total}"
    )
    print(
        f"  \u2192 {status.upper()}  (best={best_score:.3f}, "
        f"prompt={prompt_len} chars, {duration:.0f}s)\n"
    )


def experiment_skip(reason: str) -> None:
    print(f"  SKIP \u2014 {reason}\n")


# ---------------------------------------------------------------------------
# Summary reports
# ---------------------------------------------------------------------------

def research_final_report(
    experiment_count: int,
    history: list[ExperimentRecord],
    best_score: float,
    original_len: int,
    best_len: int,
    n_failures: int,
) -> None:
    kept = sum(1 for h in history if h.status == "keep")
    discarded = sum(1 for h in history if h.status == "discard")
    skipped = sum(1 for h in history if h.status == "skip")
    baseline = history[0].score if history else 0.0
    delta = best_score - baseline

    print(f"\n  Experiments:  {experiment_count}")
    print(f"  Kept:         {kept}")
    print(f"  Discarded:    {discarded}")
    print(f"  Skipped:      {skipped}")
    print(f"  Baseline:     {baseline:.3f}")
    print(f"  Best score:   {best_score:.3f}")
    print(f"  Improvement:  {'+' if delta >= 0 else ''}{delta:.3f}")
    print(f"  Prompt:       {original_len} \u2192 {best_len} chars")
    print(f"  Failures:     {n_failures} unique modes")


def pipeline_round_summary(label: str, avg_score: float, avg_csat: float, n_failures: int) -> None:
    print(f"\n  {label}: score={avg_score:.3f} csat={avg_csat:.0f} failures={n_failures}")


def pipeline_final_report(
    n_experiments: int,
    n_failures: int,
    n_additions: int,
    attack_score: float,
    attack_csat: float,
    verify_score: float,
    verify_csat: float,
) -> None:
    header("RESULTS")
    print(f"  Experiments: {n_experiments}  |  Unique failures: {n_failures}")
    print(f"  Prompt additions: {n_additions}")
    print(f"\n  BEFORE (attack):  score={attack_score:.3f}  CSAT={attack_csat:.0f}")
    print(f"  AFTER  (verify):  score={verify_score:.3f}  CSAT={verify_csat:.0f}")
    d = verify_score - attack_score
    arrow = "\u25b2" if d > 0 else "\u25bc"
    pct = d / max(attack_score, 0.01) * 100
    print(f"  Change: {arrow} {abs(d):.3f} ({'+' if d > 0 else ''}{pct:.1f}%)")


# ---------------------------------------------------------------------------
# Pipeline scenario detail
# ---------------------------------------------------------------------------

def pipeline_scenario_header(
    index: int,
    scenario_id: str,
    persona: str,
    attack: str,
    voice: dict,
) -> None:
    print(f"\n  [{index:02d}] {scenario_id} | {persona} | {attack[:60]}")
    if voice:
        print(
            f"       voice: accent={voice.get('accent', 'none')}, "
            f"noise={voice.get('background_noise', 'quiet')}, "
            f"pace={voice.get('pace', 'normal')}"
        )


def pipeline_scenario_result(
    composite: float,
    passed: bool,
    csat: int,
    failures: list[str],
    num_turns: int,
    avg_latency_ms: float,
    error: str = "",
) -> None:
    if error:
        print(f"       ERROR: {error}")
    else:
        print(f"       {num_turns} turns, {avg_latency_ms:.0f}ms avg")

    p = "PASS" if passed else "FAIL"
    bar = "\u2588" * int(composite * 20) + "\u2591" * (20 - int(composite * 20))
    print(f"       [{p}] {composite:.3f} [{bar}] CSAT={csat}")

    if failures:
        print(f"       failures: {', '.join(failures[:4])}")
