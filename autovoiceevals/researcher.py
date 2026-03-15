"""Autoresearch loop.

Karpathy's autoresearch pattern applied to voice AI prompt optimization:
one artifact (system prompt), one metric (composite score),
keep/revert binary decision, runs forever.

See program.md for the full protocol.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime

from .config import Config
from .models import EvalResult, ExperimentRecord, Metrics, Scenario
from .scoring import composite_score, aggregate
from .evaluator import Evaluator
from .llm import LLMClient
from . import display


def _build_provider(cfg: Config, llm_client: LLMClient | None = None):
    """Create the voice platform client based on config.provider."""
    if cfg.provider == "smallest":
        from .smallest import SmallestClient
        return SmallestClient(cfg.smallest_api_key, llm_client=llm_client)
    else:
        from .vapi import VapiClient
        return VapiClient(cfg.vapi_api_key)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _eval_scenario(
    provider,
    evaluator: Evaluator,
    cfg: Config,
    assistant_id: str,
    scenario: Scenario,
) -> EvalResult:
    """Run one scenario: conversation -> judge -> score."""
    conv = provider.run_conversation(
        assistant_id, scenario.id,
        scenario.caller_script, cfg.conversation.max_turns,
    )

    try:
        ev = evaluator.evaluate(conv.transcript, scenario)
    except Exception:
        ev = {
            "csat_score": 50, "passed": False, "summary": "Eval failed",
            "agent_should_results": [], "agent_should_not_results": [],
            "issues": [], "failure_modes": ["EVAL_ERROR"],
            "strengths": [], "weaknesses": [],
        }

    sr = ev.get("agent_should_results", [])
    snr = ev.get("agent_should_not_results", [])
    score, s_score, sn_score = composite_score(
        sr, snr, conv.avg_latency_ms, cfg.scoring,
    )

    return EvalResult(
        scenario_id=scenario.id,
        persona=scenario.persona_name,
        score=score,
        csat_score=ev.get("csat_score", 50),
        passed=ev.get("passed", False),
        should_score=s_score,
        should_not_score=sn_score,
        failure_modes=ev.get("failure_modes", []),
        issues=ev.get("issues", []),
        summary=ev.get("summary", ""),
        strengths=ev.get("strengths", []),
        weaknesses=ev.get("weaknesses", []),
        transcript=conv.transcript,
        num_turns=len(conv.turns),
        avg_latency_ms=conv.avg_latency_ms,
    )


def _run_eval_suite(
    provider,
    evaluator: Evaluator,
    cfg: Config,
    assistant_id: str,
    eval_suite: list[Scenario],
) -> list[EvalResult]:
    """Run the full eval suite, printing each result."""
    results: list[EvalResult] = []
    for sc in eval_suite:
        result = _eval_scenario(provider, evaluator, cfg, assistant_id, sc)
        display.eval_result_line(result)
        results.append(result)
    return results


def _json_default(obj):
    """Custom JSON serializer for dataclass objects."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def _save_log(log: dict, out_dir: str) -> None:
    path = os.path.join(out_dir, "autoresearch.json")
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=_json_default)


def _load_resume_state(out_dir: str) -> dict | None:
    log_path = os.path.join(out_dir, "autoresearch.json")
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        return json.load(f)


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------

def run(cfg: Config, resume: bool = False) -> None:
    """Run the autoresearch loop. Runs until Ctrl+C or max_experiments."""

    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.tsv")
    assistant_id = cfg.assistant.id

    threshold = cfg.autoresearch.improvement_threshold
    max_experiments = cfg.autoresearch.max_experiments
    n_eval = cfg.autoresearch.eval_scenarios

    # Build clients (LLM first — needed by SmallestClient for simulated conversations)
    llm = LLMClient(
        cfg.anthropic_api_key,
        model=cfg.llm.model,
        timeout=cfg.llm.timeout,
        max_retries=cfg.llm.max_retries,
    )
    evaluator = Evaluator(llm)
    provider = _build_provider(cfg, llm_client=llm)

    # --- Resume or fresh start ---
    prev_state = _load_resume_state(out_dir) if resume else None

    if prev_state:
        display.header("AutoVoiceEvals \u2014 Autoresearch Mode (RESUMING)")
        eval_suite = [Scenario.from_dict(s) for s in prev_state["eval_suite"]]
        original_prompt = prev_state["original_prompt"]

        history: list[ExperimentRecord] = []
        all_failures: set[str] = set()
        best_score = 0.0
        best_prompt = original_prompt
        last_eval: list[EvalResult] = []

        for exp in prev_state["experiments"]:
            history.append(ExperimentRecord(
                number=exp["experiment"],
                score=exp["score"],
                status=exp["status"],
                description=exp["description"],
                prompt_len=exp.get("prompt_len", 0),
            ))
            if exp["status"] == "keep":
                best_score = exp["score"]
                if exp.get("prompt"):
                    best_prompt = exp["prompt"]
            for r in exp.get("results", []):
                all_failures.update(r.get("failure_modes", []))
            if exp.get("results"):
                last_eval = [EvalResult.from_dict(r) for r in exp["results"]]

        experiment = history[-1].number if history else 0

        # Ensure Vapi has the best prompt
        provider.update_prompt(assistant_id, best_prompt)

        display.info(f"Resumed from experiment {experiment}")
        display.info(f"Best score: {best_score:.3f}")
        display.info(f"Best prompt: {len(best_prompt)} chars")
        display.info(f"Eval suite: {len(eval_suite)} scenarios")
        display.info(f"Failures found: {len(all_failures)}")
        if max_experiments:
            display.info(f"Remaining experiments: {max_experiments - experiment}")

        full_log = prev_state

    else:
        # --- Fresh start ---
        display.header("AutoVoiceEvals \u2014 Autoresearch Mode")
        display.info("Propose \u2192 Eval \u2192 Keep/Revert \u2192 Repeat Forever")

        original_prompt = provider.get_system_prompt(assistant_id)
        best_prompt = original_prompt

        display.blank()
        display.info(f"Assistant: {cfg.assistant.name or assistant_id}")
        display.info(f"Prompt: {len(original_prompt)} chars")
        display.info(f"Eval suite: {n_eval} adversarial scenarios")
        display.info(f"Threshold: {threshold}")
        if max_experiments:
            display.info(f"Max experiments: {max_experiments}")
        else:
            display.info("Max experiments: unlimited (Ctrl+C to stop)")

        # Generate fixed eval suite
        display.blank()
        display.info("Generating eval suite...")
        eval_suite = evaluator.generate_scenarios(
            n_eval, 1, cfg.assistant.description, [], [],
        )
        display.info(f"{len(eval_suite)} scenarios generated:")
        display.scenario_list(eval_suite)

        # Tracking state
        history = []
        all_failures = set()

        with open(results_path, "w") as f:
            f.write(
                "experiment\tscore\tcsat\tpass_rate\t"
                "prompt_len\tstatus\tdescription\n"
            )

        full_log: dict = {
            "meta": {
                "version": "autoresearch-1.0",
                "assistant": cfg.assistant.name or assistant_id,
                "llm": cfg.llm.model,
                "eval_scenarios": n_eval,
                "threshold": threshold,
                "started": datetime.now().isoformat(),
            },
            "eval_suite": eval_suite,
            "original_prompt": original_prompt,
            "experiments": [],
        }

        # --- Baseline ---
        display.section("EXPERIMENT 0: BASELINE")
        display.blank()
        baseline_results = _run_eval_suite(
            provider, evaluator, cfg, assistant_id, eval_suite,
        )
        baseline = aggregate(baseline_results)
        best_score = baseline.avg_score

        for r in baseline_results:
            all_failures.update(r.failure_modes)

        display.blank()
        display.info(
            f"Baseline: score={best_score:.3f}  csat={baseline.avg_csat:.0f}  "
            f"pass={baseline.n_passed}/{baseline.n_total}  "
            f"failures={baseline.unique_failures}"
        )

        # Log baseline
        with open(results_path, "a") as f:
            f.write(
                f"0\t{best_score:.6f}\t{baseline.avg_csat:.1f}\t"
                f"{baseline.pass_rate:.3f}\t{len(best_prompt)}\tkeep\tbaseline\n"
            )

        history.append(ExperimentRecord(
            number=0, score=best_score, status="keep",
            description="baseline", prompt_len=len(best_prompt),
        ))
        full_log["experiments"].append({
            "experiment": 0,
            "timestamp": datetime.now().isoformat(),
            "description": "baseline",
            "score": best_score,
            "csat": baseline.avg_csat,
            "pass_rate": baseline.pass_rate,
            "status": "keep",
            "results": baseline_results,
        })

        last_eval = baseline_results
        experiment = 0

    # --- The loop ---
    if max_experiments:
        display.blank()
        display.info(f"Starting autoresearch loop ({max_experiments} experiments).")
    else:
        display.blank()
        display.info("Starting autoresearch loop. Ctrl+C to stop.")
    display.blank()

    scoring_formula = cfg.scoring.formula_str()

    try:
        while True:
            if max_experiments and experiment >= max_experiments:
                display.info(f"Reached {max_experiments} experiments. Stopping.")
                display.blank()
                break

            experiment += 1
            t0 = time.time()

            display.section(f"EXPERIMENT {experiment}")

            # 1. AI proposes a change
            proposal = evaluator.propose_prompt_change(
                best_prompt, last_eval, history,
                sorted(all_failures), scoring_formula,
            )

            description = proposal.get("description", "unknown")
            change_type = proposal.get("change_type", "?")
            reasoning = proposal.get("reasoning", "")
            new_prompt = proposal.get("improved_prompt", best_prompt)

            display.experiment_proposal(
                change_type, description, reasoning,
                len(best_prompt), len(new_prompt),
            )

            # Skip if no actual change
            if new_prompt.strip() == best_prompt.strip():
                display.experiment_skip("no actual change")
                with open(results_path, "a") as f:
                    f.write(
                        f"{experiment}\t{best_score:.6f}\t0.0\t0.000\t"
                        f"{len(new_prompt)}\tskip\t{description[:80]}\n"
                    )
                history.append(ExperimentRecord(
                    number=experiment, score=best_score,
                    status="skip", description=description,
                    prompt_len=len(new_prompt),
                ))
                continue

            # 2. Apply proposed prompt to Vapi
            if not provider.update_prompt(assistant_id, new_prompt):
                display.experiment_skip("Vapi update failed")
                continue

            # 3. Run eval suite
            display.blank()
            eval_results = _run_eval_suite(
                provider, evaluator, cfg, assistant_id, eval_suite,
            )
            m = aggregate(eval_results)
            new_score = m.avg_score

            for r in eval_results:
                all_failures.update(r.failure_modes)

            # 4. Keep or revert
            delta = new_score - best_score

            if delta > threshold:
                status = "keep"
                best_prompt = new_prompt
                best_score = new_score
                last_eval = eval_results
            elif abs(delta) <= threshold and len(new_prompt) < len(best_prompt) - 20:
                status = "keep"
                description += " (simpler)"
                best_prompt = new_prompt
                best_score = new_score
                last_eval = eval_results
            else:
                status = "discard"
                provider.update_prompt(assistant_id, best_prompt)

            dt = time.time() - t0

            display.experiment_result(
                new_score, delta, m, status,
                best_score, len(best_prompt), dt,
            )

            # 5. Log
            with open(results_path, "a") as f:
                f.write(
                    f"{experiment}\t{new_score:.6f}\t{m.avg_csat:.1f}\t"
                    f"{m.pass_rate:.3f}\t{len(new_prompt)}\t{status}\t"
                    f"{description[:80]}\n"
                )

            history.append(ExperimentRecord(
                number=experiment, score=new_score, status=status,
                description=description, prompt_len=len(new_prompt),
            ))

            full_log["experiments"].append({
                "experiment": experiment,
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "change_type": change_type,
                "reasoning": reasoning,
                "prompt_len": len(new_prompt),
                "score": new_score,
                "delta": delta,
                "csat": m.avg_csat,
                "pass_rate": m.pass_rate,
                "status": status,
                "duration_s": dt,
                "results": eval_results,
                "prompt": new_prompt if status == "keep" else None,
            })

            # Save after every experiment (crash-safe)
            _save_log(full_log, out_dir)

    except KeyboardInterrupt:
        display.header("STOPPED (Ctrl+C)")

    # --- Final report ---
    display.research_final_report(
        experiment, history, best_score,
        len(original_prompt), len(best_prompt), len(all_failures),
    )

    # Restore original prompt
    display.blank()
    display.info("Restoring original prompt on Vapi...")
    provider.update_prompt(assistant_id, original_prompt)

    # Save best prompt
    best_path = os.path.join(out_dir, "best_prompt.txt")
    with open(best_path, "w") as f:
        f.write(best_prompt)
    display.info(f"Best prompt saved: {best_path}")

    # Save final log
    full_log["meta"]["ended"] = datetime.now().isoformat()
    full_log["meta"]["total_experiments"] = experiment
    full_log["meta"]["best_score"] = best_score
    full_log["meta"]["best_prompt_chars"] = len(best_prompt)
    full_log["meta"]["original_prompt_chars"] = len(original_prompt)
    full_log["best_prompt"] = best_prompt
    _save_log(full_log, out_dir)

    display.info(f"Results: {results_path}")
    display.info(f"Full log: {os.path.join(out_dir, 'autoresearch.json')}")
    display.blank()
