#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("results") / ".matplotlib_cache"))

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SCENARIOS = [
    "biased_outliers",
    "persistent_bias",
    "heterogeneous_noise",
    "reliability_drift",
    "stale_reports",
    "outlier_bursts",
    "mixed_unreliable",
]

AGENT_ORDER = [
    "oracle",
    "bayesian_uncertainty",
    "bayesian_imputation",
    "mean_reports",
    "median_reports",
]

AGENT_LABELS = {
    "oracle": "Oracle",
    "bayesian_uncertainty": "Bayes mean+std",
    "bayesian_imputation": "Bayes imputation",
    "mean_reports": "Mean reports",
    "median_reports": "Median reports",
}

COLORS = {
    "oracle": "#d62728",
    "bayesian_uncertainty": "#1f77b4",
    "bayesian_imputation": "#9467bd",
    "mean_reports": "#ff7f0e",
    "median_reports": "#2ca02c",
}


def scenario_title(name: str) -> str:
    return name.replace("_", " ").title()


def load_results(results_root: Path, run_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_frames = []
    episode_frames = []
    missing = []
    for scenario in SCENARIOS:
        run_dir = results_root / f"{run_prefix}_{scenario}_200"
        summary_path = run_dir / "summary.csv"
        episode_path = run_dir / "episode_results.csv"
        if not summary_path.exists() or not episode_path.exists():
            missing.append(str(run_dir))
            continue
        summary = pd.read_csv(summary_path)
        summary["scenario"] = scenario
        summary_frames.append(summary)
        episode = pd.read_csv(episode_path)
        episode["scenario"] = scenario
        episode_frames.append(episode)
    if missing:
        raise FileNotFoundError(
            "Missing experiment outputs. Run these first:\n"
            + "\n".join(f"  {path}" for path in missing)
        )
    return pd.concat(summary_frames, ignore_index=True), pd.concat(episode_frames, ignore_index=True)


def add_text_page(
    pdf: PdfPages,
    title: str,
    body: str,
    footer: str | None = None,
    fontsize: float = 11,
    wrap_width: int = 105,
) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.06, 0.06, 0.88, 0.88])
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=22, fontweight="bold", va="top")
    wrapped = []
    for paragraph in body.strip().split("\n\n"):
        lines = textwrap.wrap(paragraph.strip(), width=wrap_width)
        wrapped.extend(lines)
        wrapped.append("")
    ax.text(0.0, 0.87, "\n".join(wrapped), fontsize=fontsize, va="top", linespacing=1.3)
    if footer:
        ax.text(0.0, 0.01, footer, fontsize=9, color="#555555", va="bottom")
    pdf.savefig(fig)
    plt.close(fig)


def add_table_page(pdf: PdfPages, title: str, df: pd.DataFrame, columns: list[str]) -> None:
    display = df[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:,.1f}")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.set_title(title, fontsize=18, fontweight="bold", loc="left", pad=18)
    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.45)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#30343b")
        elif row % 2 == 0:
            cell.set_facecolor("#f1f3f5")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_metric_bars(
    pdf: PdfPages,
    summary: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    lower_is_better: bool = False,
) -> None:
    pivot = summary.pivot(index="scenario", columns="agent", values=metric).reindex(SCENARIOS)
    pivot = pivot[[agent for agent in AGENT_ORDER if agent in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12, 7))
    pivot.rename(index=scenario_title, columns=AGENT_LABELS).plot(
        kind="bar",
        ax=ax,
        color=[COLORS[agent] for agent in pivot.columns],
        width=0.78,
    )
    ax.set_title(title, fontsize=17, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=28)
    ax.legend(loc="best", fontsize=9)
    annotation = "Lower is better" if lower_is_better else "Higher is better"
    ax.text(0.01, 0.97, annotation, transform=ax.transAxes, va="top", fontsize=10, color="#555555")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def add_best_agent_page(pdf: PdfPages, summary: pd.DataFrame) -> None:
    non_oracle = summary[summary["agent"] != "oracle"].copy()
    idx = non_oracle.groupby("scenario")["total_regret"].idxmin()
    best = non_oracle.loc[idx].copy()
    best["scenario"] = best["scenario"].map(scenario_title)
    best["agent"] = best["agent"].map(AGENT_LABELS)
    best = best.sort_values("scenario")
    add_table_page(
        pdf,
        "Best Non-Oracle Agent by Regret",
        best,
        [
            "scenario",
            "agent",
            "total_regret",
            "total_reward",
            "mean_context_error",
            "architecture_accuracy",
        ],
    )


def add_cumulative_pages(pdf: PdfPages, episodes: pd.DataFrame, metric: str, ylabel: str) -> None:
    for scenario in SCENARIOS:
        data = episodes[episodes["scenario"] == scenario]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for agent in AGENT_ORDER:
            agent_rows = data[data["agent"] == agent]
            if agent_rows.empty:
                continue
            y_col = metric
            if y_col not in agent_rows.columns:
                y = agent_rows[y_col.replace("cumulative_", "")].cumsum()
            else:
                y = agent_rows[y_col]
            ax.plot(
                agent_rows["episode"],
                y,
                label=AGENT_LABELS.get(agent, agent),
                color=COLORS.get(agent),
                linewidth=2,
            )
        ax.set_title(f"{scenario_title(scenario)}: {ylabel}", fontsize=16, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def add_context_vs_regret(pdf: PdfPages, summary: pd.DataFrame) -> None:
    non_oracle = summary[summary["agent"] != "oracle"]
    fig, ax = plt.subplots(figsize=(11, 7))
    for agent in [agent for agent in AGENT_ORDER if agent != "oracle"]:
        data = non_oracle[non_oracle["agent"] == agent]
        ax.scatter(
            data["mean_context_error"],
            data["total_regret"],
            label=AGENT_LABELS.get(agent, agent),
            color=COLORS.get(agent),
            s=95,
            alpha=0.82,
        )
        for _, row in data.iterrows():
            ax.annotate(
                row["scenario"].replace("_", "\n"),
                (row["mean_context_error"], row["total_regret"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_title("Context Accuracy Does Not Always Translate to Reward", fontsize=16, fontweight="bold")
    ax.set_xlabel("Mean context error")
    ax.set_ylabel("Total regret")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def make_report(summary: pd.DataFrame, episodes: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_summary = summary.copy()
    report_summary["agent_label"] = report_summary["agent"].map(AGENT_LABELS)
    report_summary["scenario_label"] = report_summary["scenario"].map(scenario_title)
    report_summary = report_summary.sort_values(["scenario", "total_regret"])

    with PdfPages(output_path) as pdf:
        add_text_page(
            pdf,
            "Bayesian AdaChain Under Unreliable Node Reports",
            """
            This report evaluates an AdaChain extension for settings where the true workload is not directly observed. Each episode has a latent workload state, multiple node reports, a robust Bayesian filter over the workload, and a learned Random Forest reward model for architecture selection.

            The main result is nuanced. Multiple imputation is not consistently strong. A simpler uncertainty-aware variant, trained on posterior mean and posterior standard deviation features, is stronger in reliability-drift and biased-outlier scenarios, but simple averaging still performs well in some noise-only cases.

            The correct interpretation is not that this is a classical Byzantine-defense mechanism. The model addresses unreliable, noisy, stale, biased, or drifting node reports that remain statistically related to the true workload.
            """,
            footer=f"Generated from {len(SCENARIOS)} scenarios, {episodes['episode'].max()} episodes each.",
        )
        add_text_page(
            pdf,
            "Updated Algorithm",
            """
            1. At episode t, the true workload x_t is hidden. Nodes report noisy workload estimates o_{i,t}.

            2. A robust Bayesian filter consumes all node reports and estimates a belief over the latent workload. The implemented belief is approximated by posterior mean E[x_t | o_t] and posterior standard deviation Std[x_t | o_t].

            3. The uncertainty-aware reward model trains a RandomForestRegressor on feature vectors [posterior_mean, posterior_std, architecture_features] -> observed throughput.

            4. At decision time, the filter draws posterior workload samples. For each candidate architecture, the Random Forest predicts throughput for sampled workload contexts while retaining the posterior uncertainty features. Predictions are averaged to approximate expected value under the belief state.

            5. The selected architecture is executed in the simulator. The observed throughput is stored with the posterior mean, posterior standard deviation, and selected architecture. The Random Forest is retrained on a sliding window.

            Compared with the earlier multiple-imputation approach, this avoids attaching the same reward to many sampled contexts. It keeps training cleaner while still exposing the policy to uncertainty.
            """,
        )
        add_text_page(
            pdf,
            "Exact Algorithm Mechanics",
            """
            Agents compared in the report. Oracle sees the true latent workload and is an upper-reference policy for expected reward. Mean reports averages all node reports and feeds that context into the AdaChain-style Random Forest bandit. Median reports uses the coordinate-wise median. Bayes imputation uses the robust posterior, stores multiple sampled latent histories, and trains one Random Forest per imputed history. Bayes mean+std is the new uncertainty-feature model.

            Workload/context vector. Every episode has x = [write_ratio, hot_key_ratio, arrival_rate, execution_delay]. Reports are N x 4 matrices, one four-dimensional estimate per node. The architecture/action features are [blocksize, early_execution, reorder, signed_blocksize], where signed_blocksize is positive when early execution is enabled and negative otherwise.

            Robust Bayesian filter implementation. The estimator first subtracts the previous mean-reverted node bias from each report. It initializes the latent context with the coordinate-wise median. It then performs four robust reweighting iterations. For each node and feature, residuals are scored with Student-t style weights: weight = (nu + 1) / (nu + (residual / scale)^2). Large residuals therefore receive lower precision. The weighted precision average gives the posterior mean approximation.

            Reliability state. The filter keeps node-feature bias and noise-scale estimates. Bias follows mean reversion with rho_b = 0.95 and is constrained to sum to zero across nodes per feature, so the latent context remains identifiable as the consensus workload. Noise scale follows a smoothed update with rho_sigma = 0.95. Posterior standard deviation is computed from the inverse summed precision and floored to avoid pretending certainty is exact.

            Bayes mean+std training. After action execution, the stored supervised example is [posterior_mean, posterior_std, selected_action_features] -> observed_reward. This is a posterior-summary feature augmentation method: the reward model sees both the inferred workload and how uncertain that inference was.

            Bayes mean+std decision rule. For each episode, the filter draws posterior context samples. For each candidate action, the Random Forest predicts reward for each sampled context while appending the same posterior_std and that action's features. The action score is the average predicted reward over samples. The selected architecture is the argmax score.

            Why this differs from multiple imputation. Multiple imputation attaches one observed reward to several sampled contexts, which can blur the reward surface when the posterior is wide. The mean+std version stores one cleaner training point per episode while still allowing uncertainty to affect both training and decision-time scoring.
            """,
            fontsize=9.5,
            wrap_width=118,
        )
        add_text_page(
            pdf,
            "Exact Data Generation Scenarios",
            """
            Shared simulator setup. Each scenario starts from the same hidden workload generator and the same report generator structure. There are 9 reporting nodes by default. The unreliable fraction is 0.25, so ceil(9 * 0.25) = 3 nodes are marked unreliable. All reports are clipped to legal workload bounds. Reliable nodes report true_context plus small Student-t noise and a small initialized node bias. The base feature noise scale is [0.035, 0.035, 250, 350] for [write_ratio, hot_key_ratio, arrival_rate, execution_delay].

            biased_outliers. This is the default heavy-bias scenario. Each unreliable node receives a fixed random sign per feature. Its report is true_context + 1.8 * sign * unreliable_bias_scale + heavy-tailed Student-t noise. The unreliable bias scale is [0.18, 0.18, 900, 2200]. This creates consistently wrong, heavy-tailed reports, but not strategic adversarial behavior.

            persistent_bias. At initialization, unreliable nodes receive persistent feature-wise offsets equal to sign * unreliable_bias_scale. During sampling they still follow the true context plus noise, but from a shifted reporting baseline. This tests whether a method can handle nodes that are systematically calibrated wrong.

            heterogeneous_noise. Unreliable nodes do not necessarily have large fixed bias. Instead, their noise multiplier is [3, 3, 4, 4], so their write/hot-key reports are three times noisier and their rate/delay reports are four times noisier. This tests unequal reliability across nodes.

            reliability_drift. Unreliable nodes change behavior by phase. Every 50 episodes, the unreliable nodes move between phases. In phase 1, their noise multiplier becomes [4, 4, 5, 5] and their reported context is shifted by 0.7 * sign * unreliable_bias_scale. In phase 2, their noise multiplier becomes [1.5, 1.5, 2, 2]. In phase 0, they reset closer to normal. This is the scenario most aligned with the Bayesian filter's mean-reverting reliability state.

            stale_reports. Once enough history exists, unreliable nodes report an old true context instead of the current true context. The default lag is 4, so they report the workload from five stored positions back. This simulates delayed/stale monitoring rather than random corruption.

            outlier_bursts. Unreliable nodes usually behave like normal noisy nodes, but with probability 0.10 per episode they add a burst equal to sign * unreliable_bias_scale * U(1.0, 2.5). This tests rare transient failures.

            mixed_unreliable. This combines persistent bias, heterogeneous noise, reliability drift, stale reports, and burst outliers. It is a stress scenario for structured but non-adversarial unreliability.
            """,
            fontsize=9.2,
            wrap_width=118,
        )
        add_text_page(
            pdf,
            "What The Plots And Tables Mean",
            """
            Total regret is the most important comparison metric. In each episode, the simulator computes the expected throughput of the best action under the true hidden workload. Regret is best_expected_throughput - selected_action_expected_throughput, clipped at zero. Lower total regret means the policy chose architectures closer to the true best architecture. This avoids being fooled by random reward noise.

            Total reward is the sum of observed sampled rewards. It is useful, but noisy. Because reward includes stochastic noise, an agent can sometimes have higher realized reward than the oracle even when its expected decisions are worse. This is why regret is treated as the cleaner performance metric.

            Mean context error is the average L2 distance between the agent's estimated workload and the hidden true workload. It measures state-estimation quality, not action quality. The report shows that better context estimates do not always imply better reward, because the downstream Random Forest policy also has to learn the reward surface correctly.

            Architecture accuracy is the fraction of episodes where the agent chose the same action as the expected-throughput oracle. This is strict: two architectures can have similar expected reward, but only the exact oracle action counts as correct.

            Cumulative reward/regret plots show how performance evolves over time. A flatter cumulative regret curve means the agent is making fewer costly action mistakes. Separation late in training indicates that the learned policy is exploiting accumulated experience differently across report scenarios.

            The key story shown by the report is that the uncertainty-feature model is not merely estimating context; it exposes posterior uncertainty to the reward predictor. This helps especially under reliability drift, persistent bias, stale reports, and mixed unreliability. Mean or median baselines remain competitive in simpler noise patterns, which is important for honest evaluation.
            """,
            fontsize=9.8,
            wrap_width=116,
        )
        add_best_agent_page(pdf, report_summary)
        add_metric_bars(pdf, report_summary, "total_regret", "Total Regret by Scenario", "Total regret", True)
        add_metric_bars(pdf, report_summary, "total_reward", "Total Reward by Scenario", "Total reward")
        add_metric_bars(
            pdf,
            report_summary,
            "mean_context_error",
            "Mean Context Error by Scenario",
            "L2 context error",
            True,
        )
        add_metric_bars(
            pdf,
            report_summary,
            "architecture_accuracy",
            "Architecture Accuracy by Scenario",
            "Fraction matching oracle action",
        )
        add_context_vs_regret(pdf, report_summary)
        add_cumulative_pages(pdf, episodes, "cumulative_regret", "Cumulative Regret")
        add_cumulative_pages(pdf, episodes, "cumulative_reward", "Cumulative Reward")
        add_text_page(
            pdf,
            "Takeaways",
            """
            The uncertainty-feature model is the strongest Bayesian variant in the current implementation. It is especially compelling when node reliability changes over time, because posterior uncertainty carries useful information for the action-value model.

            Multiple imputation is weaker here. It can inject label noise by assigning one observed reward to several sampled latent contexts. The posterior can be useful, but the way it is fed into the learner matters.

            Simple baselines remain important. Mean and median aggregation are competitive in several scenarios, which makes the evaluation more credible. The proposed contribution should be framed as an uncertainty-aware latent-context AdaChain extension, with strongest evidence under reliability drift and structured unreliable reporting.
            """,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a PDF report from Bayesian AdaChain experiment outputs.")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--run-prefix", default="uncertainty")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "report" / "bayesian_adachain_report.pdf")
    args = parser.parse_args()

    summary, episodes = load_results(args.results_root, args.run_prefix)
    make_report(summary, episodes, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
