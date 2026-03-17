"""Pipeline mode: Attack -> Improve -> Verify.

Single-pass pipeline that generates adversarial attacks, analyzes
failures, rewrites the system prompt, then verifies the improvement.
Useful for a one-time audit rather than iterative optimization.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

from .config import Config
from .models import Scenario
from .scoring import composite_score
from .evaluator import Evaluator
from .vapi import VapiClient  # noqa: F401 — used by provider factory
from .llm import LLMClient
from . import display, graphs


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _run_round(
    label: str,
    phase: str,
    scenarios: list[Scenario],
    cfg: Config,
    vapi,  # VapiClient or SmallestClient
    evaluator: Evaluator,
    all_experiments: list[dict],
    all_failures: set[str],
) -> list[tuple]:
    """Run one round of evaluation.

    Returns list of (scenario, conversation, eval_dict, composite_score)
    tuples for further processing.
    """
    results: list[tuple] = []

    for i, sc in enumerate(scenarios):
        n = len(all_experiments) + 1
        sid = sc.id

        display.pipeline_scenario_header(
            n, sid, sc.persona_name, sc.attack_strategy,
            sc.voice_characteristics,
        )

        conv = vapi.run_conversation(
            cfg.assistant.id, sid,
            sc.caller_script, cfg.conversation.max_turns,
            scenario=sc,
            dynamic_variables=cfg.assistant.dynamic_variables,
        )

        try:
            ev = evaluator.evaluate(conv.transcript, sc)
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

        failures = ev.get("failure_modes", [])
        for fm in failures:
            all_failures.add(fm)

        display.pipeline_scenario_result(
            composite=score,
            passed=ev.get("passed", False),
            csat=ev.get("csat_score", 50),
            failures=failures,
            num_turns=len(conv.turns),
            avg_latency_ms=conv.avg_latency_ms,
            error=conv.error,
        )

        exp = {
            "round": label, "phase": phase, "scenario_id": sid,
            "scenario": sc.to_dict(),
            "difficulty": sc.difficulty,
            "persona": sc.persona_name,
            "attack_strategy": sc.attack_strategy,
            "voice_characteristics": sc.voice_characteristics,
            "score": score, "csat_score": ev.get("csat_score", 50),
            "passed": ev.get("passed", False),
            "should_score": s_score, "should_not_score": sn_score,
            "agent_should_results": sr, "agent_should_not_results": snr,
            "failure_modes": failures,
            "issues": ev.get("issues", []),
            "strengths": ev.get("strengths", []),
            "weaknesses": ev.get("weaknesses", []),
            "summary": ev.get("summary", ""),
            "num_turns": len(conv.turns),
            "avg_latency_ms": conv.avg_latency_ms,
            "transcript": conv.transcript,
        }
        all_experiments.append(exp)
        results.append((sc, conv, ev, score))

    return results


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------

def run(cfg: Config) -> None:
    """Run the full attack -> improve -> verify pipeline."""

    display.header("AutoVoiceEvals \u2014 Pipeline Mode")
    display.info("Phase A: Attack | Phase B: Improve | Phase C: Verify")

    out_dir = cfg.output.dir
    os.makedirs(out_dir, exist_ok=True)

    assistant_id = cfg.assistant.id
    agent_desc = cfg.assistant.description
    n_attack = cfg.pipeline.attack_rounds
    n_verify = cfg.pipeline.verify_rounds
    n_scenarios = cfg.pipeline.scenarios_per_round
    top_k = cfg.pipeline.top_k_elites

    # Build clients
    llm = LLMClient(
        cfg.anthropic_api_key,
        model=cfg.llm.model,
        timeout=cfg.llm.timeout,
        max_retries=cfg.llm.max_retries,
    )
    evaluator = Evaluator(llm)

    if cfg.provider == "smallest":
        from .smallest import SmallestClient
        vapi = SmallestClient(cfg.smallest_api_key, llm_client=llm)
    elif cfg.provider == "elevenlabs":
        from .elevenlabs import ElevenLabsClient
        vapi = ElevenLabsClient(cfg.elevenlabs_api_key)
    else:
        vapi = VapiClient(cfg.vapi_api_key)

    all_experiments: list[dict] = []
    all_failures: set[str] = set()
    all_issues: list[dict] = []
    elite_pool: list[tuple] = []
    round_stats: list[dict] = []

    original_prompt = vapi.get_system_prompt(assistant_id)
    display.blank()
    display.info(f"Assistant: {cfg.assistant.name or assistant_id}")
    display.info(f"Original prompt: {len(original_prompt)} chars")
    total = (n_attack + n_verify) * n_scenarios
    display.info(
        f"Plan: {n_attack} attack + {n_verify} verify rounds "
        f"x {n_scenarios} = {total} experiments"
    )

    # === PHASE A: ATTACK =============================================

    for rnd in range(1, n_attack + 1):
        label = f"A{rnd}"
        display.section(f"PHASE A \u2014 ATTACK {rnd}/{n_attack}")

        prev_f = sorted(all_failures)
        prev_t = [
            e["transcript"]
            for e in sorted(all_experiments, key=lambda x: x["score"])[:2]
        ]

        scenarios: list[Scenario] = []
        if rnd == 1 or not elite_pool:
            scenarios = evaluator.generate_scenarios(
                n_scenarios, rnd, agent_desc, prev_f, prev_t,
            )
        else:
            for j in range(min(top_k * 2, n_scenarios - 1)):
                parent_sc, parent_t, parent_fm, _ = elite_pool[j % len(elite_pool)]
                try:
                    m = evaluator.mutate_scenario(
                        parent_sc, parent_t, parent_fm, f"{label}_M{j+1:02d}",
                    )
                    if m:
                        scenarios.append(m)
                except Exception:
                    pass
            need = n_scenarios - len(scenarios)
            if need > 0:
                scenarios.extend(
                    evaluator.generate_scenarios(
                        need, rnd, agent_desc, prev_f, prev_t,
                    )
                )

        results = _run_round(
            label, "attack", scenarios, cfg,
            vapi, evaluator, all_experiments, all_failures,
        )
        for r in results:
            all_issues.extend(r[2].get("issues", []))

        scores = [r[3] for r in results]
        csats = [r[2].get("csat_score", 50) for r in results]
        s_vals = [e["should_score"] for e in all_experiments if e["round"] == label]
        sn_vals = [e["should_not_score"] for e in all_experiments if e["round"] == label]

        stat = {
            "round": label, "phase": "attack",
            "avg_score": sum(scores) / max(len(scores), 1),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_csat": sum(csats) / max(len(csats), 1),
            "avg_dimensions": {
                "agent_should": sum(s_vals) / max(len(s_vals), 1),
                "agent_should_not": sum(sn_vals) / max(len(sn_vals), 1),
            },
            "unique_failures_cumulative": len(all_failures),
        }
        round_stats.append(stat)

        elite_pool = [
            (r[0], r[1].transcript, r[2].get("failure_modes", []), r[3])
            for r in sorted(results, key=lambda r: r[3])[:top_k]
        ]

        display.pipeline_round_summary(
            label, stat["avg_score"], stat["avg_csat"], len(all_failures),
        )

    # === PHASE B: IMPROVE ============================================

    display.section(
        f"PHASE B \u2014 IMPROVER AGENT "
        f"({len(all_issues)} issues, {len(all_failures)} failure modes)"
    )

    worst_t = [
        e["transcript"]
        for e in sorted(all_experiments, key=lambda x: x["score"])[:3]
    ]
    improvement = evaluator.improve_prompt(
        original_prompt, all_issues, sorted(all_failures), worst_t,
    )

    additions = improvement.get("prompt_additions", [])
    improved_prompt = improvement.get("improved_prompt", original_prompt)

    display.blank()
    display.info(f"Generated {len(additions)} prompt additions:")
    for pa in additions[:6]:
        sev = pa.get("severity", "?").upper()
        desc = pa.get("description", "")[:65]
        print(f"    [{sev}] {desc}")

    display.blank()
    display.info(
        f"Improved prompt: {len(improved_prompt)} chars "
        f"(was {len(original_prompt)})"
    )

    if vapi.update_prompt(assistant_id, improved_prompt):
        display.info("Agent prompt updated.")
    else:
        display.info("WARNING: Agent update failed!")

    # === PHASE C: VERIFY =============================================

    for rnd in range(1, n_verify + 1):
        label = f"C{rnd}"
        display.section(f"PHASE C \u2014 VERIFY {rnd}/{n_verify} (improved prompt)")

        prev_f = sorted(all_failures)
        prev_t = [
            e["transcript"]
            for e in sorted(all_experiments, key=lambda x: x["score"])[:2]
        ]

        scenarios = []
        if rnd == 1 and elite_pool:
            for j, (sc, _, _, _) in enumerate(elite_pool):
                copy = Scenario.from_dict(sc.to_dict())
                copy.id = f"{label}_RE{j+1:02d}"
                scenarios.append(copy)
            need = n_scenarios - len(scenarios)
            if need > 0:
                scenarios.extend(
                    evaluator.generate_scenarios(
                        need, rnd + n_attack, agent_desc, prev_f, prev_t,
                    )
                )
        else:
            for j in range(min(top_k * 2, n_scenarios - 1)):
                parent_sc, parent_t, parent_fm, _ = elite_pool[j % len(elite_pool)]
                try:
                    m = evaluator.mutate_scenario(
                        parent_sc, parent_t, parent_fm, f"{label}_M{j+1:02d}",
                    )
                    if m:
                        scenarios.append(m)
                except Exception:
                    pass
            need = n_scenarios - len(scenarios)
            if need > 0:
                scenarios.extend(
                    evaluator.generate_scenarios(
                        need, rnd + n_attack, agent_desc, prev_f, prev_t,
                    )
                )

        results = _run_round(
            label, "verify", scenarios, cfg,
            vapi, evaluator, all_experiments, all_failures,
        )
        for r in results:
            all_issues.extend(r[2].get("issues", []))

        scores = [r[3] for r in results]
        csats = [r[2].get("csat_score", 50) for r in results]
        s_vals = [e["should_score"] for e in all_experiments if e["round"] == label]
        sn_vals = [e["should_not_score"] for e in all_experiments if e["round"] == label]

        stat = {
            "round": label, "phase": "verify",
            "avg_score": sum(scores) / max(len(scores), 1),
            "min_score": min(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "avg_csat": sum(csats) / max(len(csats), 1),
            "avg_dimensions": {
                "agent_should": sum(s_vals) / max(len(s_vals), 1),
                "agent_should_not": sum(sn_vals) / max(len(sn_vals), 1),
            },
            "unique_failures_cumulative": len(all_failures),
        }
        round_stats.append(stat)

        elite_pool = [
            (r[0], r[1].transcript, r[2].get("failure_modes", []), r[3])
            for r in sorted(results, key=lambda r: r[3])[:top_k]
        ]

        display.pipeline_round_summary(
            label, stat["avg_score"], stat["avg_csat"], len(all_failures),
        )

    # === Cleanup =====================================================

    display.blank()
    display.info("Restoring original prompt...")
    vapi.update_prompt(assistant_id, original_prompt)

    # Save experiment log
    log = {
        "meta": {
            "version": "autovoiceevals-1.0",
            "timestamp": datetime.now().isoformat(),
            "assistant": cfg.assistant.name or assistant_id,
            "llm": cfg.llm.model,
            "total_experiments": len(all_experiments),
            "unique_failures": len(all_failures),
            "prompt_additions": len(additions),
            "original_prompt_chars": len(original_prompt),
            "improved_prompt_chars": len(improved_prompt),
        },
        "prompt_additions": additions,
        "improved_prompt": improved_prompt,
        "round_stats": round_stats,
        "experiments": all_experiments,
    }
    log_path = os.path.join(out_dir, "experiments.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)

    # Graphs
    if cfg.output.graphs:
        paths = graphs.generate_all(round_stats, all_experiments, out_dir)
        for p in paths:
            display.info(f"graph: {p}")

    # Final report
    attack_s = [s for s in round_stats if s["phase"] == "attack"]
    verify_s = [s for s in round_stats if s["phase"] == "verify"]
    a_avg = sum(s["avg_score"] for s in attack_s) / max(len(attack_s), 1)
    v_avg = sum(s["avg_score"] for s in verify_s) / max(len(verify_s), 1)
    a_csat = sum(s["avg_csat"] for s in attack_s) / max(len(attack_s), 1)
    v_csat = sum(s["avg_csat"] for s in verify_s) / max(len(verify_s), 1)

    display.pipeline_final_report(
        len(all_experiments), len(all_failures), len(additions),
        a_avg, a_csat, v_avg, v_csat,
    )
    display.blank()
    display.info(f"Output: {out_dir}/")
    display.blank()
