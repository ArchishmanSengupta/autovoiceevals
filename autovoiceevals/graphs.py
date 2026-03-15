"""Visualization for autovoiceevals results.

Two entry points:
  - generate_all()          — pipeline mode (attack/verify rounds)
  - generate_research()     — autoresearch mode (keep/revert experiments)
"""

from __future__ import annotations

import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import matplotlib.ticker as ticker       # noqa: E402


# ===================================================================
# Autoresearch mode graphs
# ===================================================================

def generate_research(experiments: list[dict], output_dir: str) -> list[str]:
    """Generate all charts for an autoresearch run."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    paths.append(_research_score_progression(experiments, output_dir))
    paths.append(_research_metrics_panel(experiments, output_dir))
    paths.append(_research_keep_discard(experiments, output_dir))
    paths.append(_research_prompt_evolution(experiments, output_dir))

    fm_path = _research_failure_modes(experiments, output_dir)
    if fm_path:
        paths.append(fm_path)

    return paths


def _research_score_progression(experiments: list[dict], output_dir: str) -> str:
    """Score progression — Karpathy style.

    Small gray dots for discards, large green dots for keeps,
    step-line for running best, clean annotations on keeps only.
    """
    n_total = len(experiments) - 1  # exclude baseline from count
    n_kept = sum(1 for e in experiments if e.get("status") == "keep") - 1

    fig, ax = plt.subplots(figsize=(16, 7))

    # --- Discarded: small, light gray, de-emphasized ---
    for e in experiments:
        if e.get("status") != "keep":
            ax.scatter(
                e["experiment"], e["score"],
                color="#cccccc", s=40, zorder=2, alpha=0.7,
            )

    # --- Running best step-line (green, solid) ---
    keep_nums = []
    keep_scores = []
    for e in experiments:
        if e.get("status") == "keep":
            keep_nums.append(e["experiment"])
            keep_scores.append(e["score"])

    # Extend the step line to the last experiment
    last_exp = experiments[-1]["experiment"] if experiments else 0
    step_x = []
    step_y = []
    for i, (n, s) in enumerate(zip(keep_nums, keep_scores)):
        step_x.append(n)
        step_y.append(s)
        # Extend horizontally to next keep (or end)
        if i < len(keep_nums) - 1:
            step_x.append(keep_nums[i + 1])
            step_y.append(s)
        else:
            step_x.append(last_exp)
            step_y.append(s)

    ax.plot(step_x, step_y, "-", color="#2ecc71", lw=2.5, alpha=0.8,
            zorder=3, label="Running best")

    # --- Kept: large green dots ---
    first_keep = True
    for e in experiments:
        if e.get("status") == "keep":
            ax.scatter(
                e["experiment"], e["score"],
                color="#2ecc71", s=120, edgecolors="white", lw=2,
                zorder=5, label="Kept" if first_keep else None,
            )
            first_keep = False

    # --- Discard label (just once for legend) ---
    ax.scatter([], [], color="#cccccc", s=40, label="Discarded")

    # --- Annotate keeps with short descriptions ---
    kept_exps = [e for e in experiments if e.get("status") == "keep"]
    # Alternate annotation positions to avoid overlap
    for i, e in enumerate(kept_exps):
        desc = e.get("description", "")
        # Shorten intelligently
        if len(desc) > 40:
            desc = desc[:37] + "..."
        if e["experiment"] == 0:
            desc = "baseline"

        # Alternate above/below to reduce clutter
        y_offset = 18 if i % 2 == 0 else -22

        ax.annotate(
            desc,
            xy=(e["experiment"], e["score"]),
            xytext=(8, y_offset),
            textcoords="offset points",
            fontsize=7.5,
            color="#27ae60",
            fontstyle="italic",
            arrowprops=dict(
                arrowstyle="-",
                color="#27ae60",
                lw=0.7,
                alpha=0.6,
            ),
        )

    # --- Axes ---
    scores = [e["score"] for e in experiments]
    y_min = min(scores) - 0.03
    y_max = max(max(scores) + 0.03, 1.0)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(-0.5, last_exp + 0.5)

    ax.set_xlabel("Experiment #", fontsize=13)
    ax.set_ylabel("Composite Score (higher is better)", fontsize=13)
    ax.set_title(
        f"AutoVoiceEvals: {n_total} Experiments, "
        f"{n_kept} Kept Improvements",
        fontsize=15, fontweight="bold",
    )

    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    p = os.path.join(output_dir, "01_score_progression.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _research_metrics_panel(experiments: list[dict], output_dir: str) -> str:
    """2x2 panel: score, CSAT, pass rate, prompt length over experiments."""
    nums = [e["experiment"] for e in experiments]
    scores = [e["score"] for e in experiments]
    csats = [e.get("csat", 0) for e in experiments]
    pass_rates = [e.get("pass_rate", 0) * 100 for e in experiments]
    prompt_lens = [e.get("prompt_len", 0) for e in experiments]
    statuses = [e.get("status", "keep") for e in experiments]

    colors = ["#2ecc71" if s == "keep" else "#e74c3c" for s in statuses]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Score
    ax = axes[0, 0]
    ax.bar(nums, scores, color=colors, width=0.7, edgecolor="white", lw=1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Composite Score", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # CSAT
    ax = axes[0, 1]
    ax.bar(nums, csats, color=colors, width=0.7, edgecolor="white", lw=1)
    ax.set_ylabel("CSAT (0-100)", fontsize=11)
    ax.set_title("Customer Satisfaction", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Pass rate
    ax = axes[1, 0]
    ax.bar(nums, pass_rates, color=colors, width=0.7, edgecolor="white", lw=1)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_title("Pass Rate", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis="y")

    # Prompt length
    ax = axes[1, 1]
    ax.bar(nums, prompt_lens, color=colors, width=0.7, edgecolor="white", lw=1)
    ax.set_ylabel("Chars", fontsize=11)
    ax.set_xlabel("Experiment", fontsize=11)
    ax.set_title("Prompt Length", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "AutoVoiceEvals — All Metrics  (green = keep, red = discard)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    p = os.path.join(output_dir, "02_metrics_panel.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def _research_keep_discard(experiments: list[dict], output_dir: str) -> str:
    """Keep/discard decision map — horizontal bar showing each experiment."""
    exps = [e for e in experiments if e["experiment"] > 0]
    if not exps:
        # Only baseline, nothing to show
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "Only baseline — no experiments to show",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        p = os.path.join(output_dir, "03_keep_discard.png")
        plt.savefig(p, dpi=150)
        plt.close()
        return p

    nums = [e["experiment"] for e in exps]
    deltas = [e.get("delta", 0) for e in exps]
    statuses = [e.get("status", "discard") for e in exps]
    colors = ["#2ecc71" if s == "keep" else "#e74c3c" if s == "discard" else "#f39c12"
              for s in statuses]

    fig, ax = plt.subplots(figsize=(14, max(4, len(exps) * 0.45)))

    bars = ax.barh(range(len(exps)), deltas, color=colors, height=0.6,
                   edgecolor="white", lw=1.5)

    for i, (exp, delta, status) in enumerate(zip(exps, deltas, statuses)):
        desc = exp.get("description", "")[:45]
        sign = "+" if delta > 0 else ""
        ax.text(
            max(delta, 0) + 0.003 if delta >= 0 else min(delta, 0) - 0.003,
            i, f"{sign}{delta:.3f}  {desc}",
            va="center", ha="left" if delta >= 0 else "right",
            fontsize=8, color="#2c3e50",
        )

    ax.axvline(x=0, color="#2c3e50", lw=1)
    ax.set_yticks(range(len(exps)))
    ax.set_yticklabels([f"exp {n}" for n in nums], fontsize=9)
    ax.set_xlabel("Score Delta", fontsize=12)
    ax.set_title("AutoVoiceEvals — Keep / Discard Decisions", fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    p = os.path.join(output_dir, "03_keep_discard.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _research_prompt_evolution(experiments: list[dict], output_dir: str) -> str:
    """Prompt length over time, showing only the 'active' prompt after keeps."""
    keeps = [e for e in experiments if e.get("status") == "keep"]
    if len(keeps) < 2:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Not enough keeps to show prompt evolution",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        p = os.path.join(output_dir, "04_prompt_evolution.png")
        plt.savefig(p, dpi=150)
        plt.close()
        return p

    nums = [e["experiment"] for e in keeps]
    lens = [e.get("prompt_len", 0) for e in keeps]
    scores = [e["score"] for e in keeps]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(len(keeps)), lens, color="#3498db", width=0.5,
            edgecolor="white", lw=1.5, alpha=0.7, label="Prompt length")
    ax2.plot(range(len(keeps)), scores, "o-", color="#e74c3c", lw=2.5,
             ms=10, label="Score", zorder=5)

    for i, (n, length, score) in enumerate(zip(nums, lens, scores)):
        ax1.text(i, length + 20, f"{length}", ha="center", fontsize=9, color="#2c3e50")
        ax2.text(i, score + 0.01, f"{score:.3f}", ha="center", fontsize=9, color="#c0392b")

    ax1.set_xlabel("Kept Experiment", fontsize=12)
    ax1.set_ylabel("Prompt Length (chars)", fontsize=12, color="#3498db")
    ax2.set_ylabel("Score", fontsize=12, color="#e74c3c")
    ax1.set_xticks(range(len(keeps)))
    ax1.set_xticklabels([f"exp {n}" for n in nums], fontsize=9)
    ax2.set_ylim(0, 1.1)
    ax1.set_title("AutoVoiceEvals — Prompt Evolution (keeps only)",
                   fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "04_prompt_evolution.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _research_failure_modes(experiments: list[dict], output_dir: str) -> str | None:
    """Top failure modes discovered across all experiments."""
    fm_counts: Counter = Counter()
    for exp in experiments:
        for r in exp.get("results", []):
            for fm in r.get("failure_modes", []):
                fm_counts[fm] += 1

    if not fm_counts:
        return None

    # Top 15
    top = fm_counts.most_common(15)
    labels = [t[0] for t in reversed(top)]
    counts = [t[1] for t in reversed(top)]

    fig, ax = plt.subplots(figsize=(12, max(5, len(top) * 0.4)))
    bars = ax.barh(range(len(labels)), counts, color="#8e44ad", height=0.6,
                   edgecolor="white", lw=1.5)

    for i, c in enumerate(counts):
        ax.text(c + 0.3, i, str(c), va="center", fontsize=10, color="#2c3e50")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Occurrences", fontsize=12)
    ax.set_title("AutoVoiceEvals — Failure Modes Discovered", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    p = os.path.join(output_dir, "05_failure_modes.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


# ===================================================================
# Pipeline mode graphs (unchanged)
# ===================================================================

def generate_all(
    round_stats: list[dict],
    experiments: list[dict],
    output_dir: str,
) -> list[str]:
    """Generate all charts for a pipeline run."""
    os.makedirs(output_dir, exist_ok=True)
    paths: list[str] = []

    paths.append(_score_comparison(round_stats, output_dir))
    paths.append(_cumulative_failures(round_stats, output_dir))
    paths.append(_experiment_scatter(experiments, output_dir))

    issues_path = _issue_breakdown(experiments, output_dir)
    if issues_path:
        paths.append(issues_path)

    return paths


def _score_comparison(round_stats: list[dict], output_dir: str) -> str:
    labels = [s["round"] for s in round_stats]
    scores = [s["avg_score"] for s in round_stats]
    csats = [s["avg_csat"] for s in round_stats]
    phases = [s["phase"] for s in round_stats]
    colors = ["#e74c3c" if p == "attack" else "#2ecc71" for p in phases]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.bar(labels, scores, color=colors, width=0.6, edgecolor="white", lw=2)
    for i, (_, s) in enumerate(zip(labels, scores)):
        ax1.text(i, s + 0.02, f"{s:.3f}", ha="center", fontsize=12, fontweight="bold")
    n_attack = sum(1 for p in phases if p == "attack")
    ax1.axvline(x=n_attack - 0.5, color="#2c3e50", ls="--", lw=2, label="Prompt improved")
    ax1.set_ylabel("Avg Agent Score", fontsize=13)
    ax1.set_title(
        "AutoVoiceEvals: Attack \u2192 Improve \u2192 Verify\n"
        "(Red = before, Green = after prompt improvement)",
        fontsize=14, fontweight="bold",
    )
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(labels, csats, color=colors, width=0.6, edgecolor="white", lw=2)
    for i, (_, c) in enumerate(zip(labels, csats)):
        ax2.text(i, c + 1.5, f"{c:.0f}", ha="center", fontsize=12, fontweight="bold")
    ax2.axvline(x=n_attack - 0.5, color="#2c3e50", ls="--", lw=2)
    ax2.set_ylabel("Avg CSAT (0-100)", fontsize=13)
    ax2.set_xlabel("Round", fontsize=13)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "01_comparison.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _cumulative_failures(round_stats: list[dict], output_dir: str) -> str:
    labels = [s["round"] for s in round_stats]
    cum = [s.get("unique_failures_cumulative", 0) for s in round_stats]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(cum)), cum, "o-", color="#8e44ad", lw=2.5, ms=10)
    ax.fill_between(range(len(cum)), cum, alpha=0.1, color="#8e44ad")
    for i, c in enumerate(cum):
        ax.text(i, c + 0.5, str(c), ha="center", fontsize=12, fontweight="bold", color="#8e44ad")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Round", fontsize=13)
    ax.set_ylabel("Cumulative Unique Failures", fontsize=13)
    ax.set_title("Failure Discovery Rate", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = os.path.join(output_dir, "02_cumulative_failures.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _experiment_scatter(experiments: list[dict], output_dir: str) -> str:
    tier_colors = {"A": "#2ecc71", "B": "#3498db", "C": "#f39c12", "D": "#e74c3c"}
    seen: set[str] = set()

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, exp in enumerate(experiments):
        d = exp.get("difficulty", "B")
        label = f"Tier {d}" if d not in seen else None
        seen.add(d)
        ax.scatter(
            i + 1, exp["score"],
            color=tier_colors.get(d, "#999"),
            s=120, edgecolors="white", lw=1.5, label=label, zorder=5,
        )
        if not exp.get("passed", True):
            ax.scatter(
                i + 1, exp["score"],
                color="none", s=200, edgecolors="#e74c3c", lw=2.5, zorder=6,
            )

    n_attack_exp = sum(1 for e in experiments if e.get("phase") == "attack")
    if n_attack_exp > 0:
        ax.axvline(x=n_attack_exp + 0.5, color="#2c3e50", ls="--", lw=2, alpha=0.7)

    ax.set_xlabel("Experiment #", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Every Experiment (Red rings = FAIL)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=5)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(output_dir, "03_scatter.png")
    plt.savefig(p, dpi=150)
    plt.close()
    return p


def _issue_breakdown(experiments: list[dict], output_dir: str) -> str | None:
    issue_counts: Counter = Counter()
    sev_counts: Counter = Counter()

    for exp in experiments:
        for iss in exp.get("issues", []):
            issue_counts[iss.get("type", "?")] += 1
            sev_counts[iss.get("severity", "?")] += 1

    if not issue_counts:
        return None

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))

    cols = plt.cm.Set2(range(len(issue_counts)))
    a1.pie(
        issue_counts.values(),
        labels=[k[:25] for k in issue_counts.keys()],
        autopct="%1.0f%%", colors=cols, textprops={"fontsize": 9},
    )
    a1.set_title("Issues by Type", fontsize=13, fontweight="bold")

    sev_colors = {
        "critical": "#8e44ad", "high": "#e74c3c",
        "medium": "#f39c12", "low": "#2ecc71",
    }
    sev_labels = [s for s in ["critical", "high", "medium", "low"] if s in sev_counts]
    a2.pie(
        [sev_counts[s] for s in sev_labels],
        labels=[s.upper() for s in sev_labels],
        autopct="%1.0f%%",
        colors=[sev_colors.get(s, "#999") for s in sev_labels],
        textprops={"fontsize": 10},
    )
    a2.set_title("Issues by Severity", fontsize=13, fontweight="bold")

    plt.suptitle("Issue Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p = os.path.join(output_dir, "04_issues.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p
