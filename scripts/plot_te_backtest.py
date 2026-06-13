import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "dynamic": "#0072B2",
    "static": "#D55E00",
    "benchmark": "#666666",
    "accent": "#009E73",
}


def _read_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_report(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[skip] missing report: {path}")
        return None
    report = _read_pickle(path)
    if not isinstance(report, pd.DataFrame):
        raise TypeError(f"{path} does not contain a pandas DataFrame")
    report = report.copy().sort_index()
    report.index = pd.to_datetime(report.index)
    return report


def _load_metrics(output_prefix: Path) -> Tuple[Optional[Dict], Optional[Dict]]:
    summary = _read_json(Path(str(output_prefix) + "-summary.json"))
    if summary is not None:
        return summary.get("dynamic_result"), summary.get("static_w0_result")

    dynamic = _read_json(Path(str(output_prefix) + "-result.json"))
    static = _read_json(Path(str(output_prefix) + "_static_w0-result.json"))
    return dynamic, static


def _net_daily_return(report: pd.DataFrame) -> pd.Series:
    cost = report["cost"] if "cost" in report else 0.0
    return report["return"] - cost


def _cum_return(daily_return: pd.Series) -> pd.Series:
    return (1.0 + daily_return.fillna(0.0)).cumprod() - 1.0


def _setup_style(dpi: int) -> None:
    plt.rcParams.update({
        "figure.dpi": dpi,
        "savefig.dpi": dpi,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.8,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path}")


def _plot_cumulative_returns(
    dynamic_report: Optional[pd.DataFrame],
    static_report: Optional[pd.DataFrame],
    fig_dir: Path,
    dynamic_label: str,
    static_label: str,
) -> None:
    if dynamic_report is None and static_report is None:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    benchmark_drawn = False
    if dynamic_report is not None:
        _cum_return(_net_daily_return(dynamic_report)).plot(
            ax=ax, color=COLORS["dynamic"], linewidth=2.0, label=dynamic_label
        )
        if "bench" in dynamic_report:
            _cum_return(dynamic_report["bench"]).plot(
                ax=ax,
                color=COLORS["benchmark"],
                linewidth=1.6,
                linestyle="--",
                label="Benchmark",
            )
            benchmark_drawn = True

    if static_report is not None:
        _cum_return(_net_daily_return(static_report)).plot(
            ax=ax, color=COLORS["static"], linewidth=2.0, label=static_label
        )
        if "bench" in static_report and not benchmark_drawn:
            _cum_return(static_report["bench"]).plot(
                ax=ax,
                color=COLORS["benchmark"],
                linewidth=1.6,
                linestyle="--",
                label="Benchmark",
            )

    ax.set_title("Cumulative Return")
    ax.set_xlabel("")
    ax.set_ylabel("Return")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False)
    _save(fig, fig_dir / "cumulative_return.png")


def _plot_excess_returns(
    dynamic_report: Optional[pd.DataFrame],
    static_report: Optional[pd.DataFrame],
    fig_dir: Path,
    dynamic_label: str,
    static_label: str,
) -> None:
    if dynamic_report is None and static_report is None:
        return

    fig, ax = plt.subplots(figsize=(8.8, 4.8))

    if dynamic_report is not None and "bench" in dynamic_report:
        dynamic_net = 1.0 + _cum_return(_net_daily_return(dynamic_report))
        dynamic_bench = 1.0 + _cum_return(dynamic_report["bench"])
        (dynamic_net / dynamic_bench - 1.0).plot(
            ax=ax, color=COLORS["dynamic"], linewidth=2.0, label=dynamic_label
        )

    if static_report is not None and "bench" in static_report:
        static_net = 1.0 + _cum_return(_net_daily_return(static_report))
        static_bench = 1.0 + _cum_return(static_report["bench"])
        (static_net / static_bench - 1.0).plot(
            ax=ax, color=COLORS["static"], linewidth=2.0, label=static_label
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_title("Cumulative Excess Return")
    ax.set_xlabel("")
    ax.set_ylabel("Excess return")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(frameon=False)
    _save(fig, fig_dir / "excess_return.png")


def _plot_metrics(
    dynamic_metrics: Optional[Dict],
    static_metrics: Optional[Dict],
    fig_dir: Path,
    dynamic_label: str,
    static_label: str,
) -> None:
    if dynamic_metrics is None and static_metrics is None:
        print("[skip] missing metrics json")
        return

    metric_names = [
        "sharpe",
        "information_ratio",
        "annual_return",
        "annual_excess_return",
        "max_drawdown",
        "excess_max_drawdown",
    ]
    rows = []
    for label, metrics in [(dynamic_label, dynamic_metrics), (static_label, static_metrics)]:
        if metrics is None:
            continue
        for name in metric_names:
            if name in metrics and metrics[name] is not None:
                rows.append({"method": label, "metric": name, "value": float(metrics[name])})
    if not rows:
        print("[skip] no metrics available")
        return

    df = pd.DataFrame(rows)
    metrics = [name for name in metric_names if name in set(df["metric"])]
    x = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for offset, (label, color) in enumerate(
        [(dynamic_label, COLORS["dynamic"]), (static_label, COLORS["static"])]
    ):
        values = []
        for metric in metrics:
            subset = df[(df["method"] == label) & (df["metric"] == metric)]
            values.append(np.nan if subset.empty else subset["value"].iloc[0])
        ax.bar(x + (offset - 0.5) * width, values, width=width, color=color, label=label)

    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_title("Backtest Metrics")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.legend(frameon=False)
    _save(fig, fig_dir / "metrics_comparison.png")


def _ordered_alpha_columns(columns: Iterable[str], prefix: str) -> List[str]:
    def key(name: str) -> int:
        return int(name.rsplit("_", 1)[-1])

    return sorted([col for col in columns if col.startswith(prefix)], key=key)


def _plot_gate_heatmap(daily_weights_path: Path, fig_dir: Path) -> None:
    if not daily_weights_path.exists():
        print(f"[skip] missing daily weights: {daily_weights_path}")
        return

    daily = pd.read_csv(daily_weights_path)
    if "date" not in daily:
        raise ValueError(f"{daily_weights_path} must contain a date column")
    daily["date"] = pd.to_datetime(daily["date"])
    gate_cols = _ordered_alpha_columns(daily.columns, "gate_alpha_")
    if not gate_cols:
        print("[skip] no gate_alpha columns")
        return

    matrix = daily[gate_cols].to_numpy(dtype=float).T
    dates = daily["date"]
    alpha_labels = [col.replace("gate_alpha_", "alpha ") for col in gate_cols]

    fig, ax = plt.subplots(figsize=(9.4, max(3.8, 0.35 * len(gate_cols) + 1.8)))
    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_title("Market Gate Values Over Time")
    ax.set_ylabel("Alpha")
    ax.set_yticks(np.arange(len(alpha_labels)))
    ax.set_yticklabels(alpha_labels)

    tick_count = min(8, len(dates))
    tick_positions = np.linspace(0, len(dates) - 1, tick_count, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates.iloc[i].strftime("%Y-%m-%d") for i in tick_positions], rotation=30, ha="right")
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("gate")
    _save(fig, fig_dir / "gate_heatmap.png")


def _plot_top_dynamic_weights(
    timing_summary_path: Path,
    daily_weights_path: Path,
    fig_dir: Path,
    max_alpha_lines: int,
) -> None:
    if not timing_summary_path.exists() or not daily_weights_path.exists():
        print("[skip] missing timing summary or daily weights")
        return

    summary = pd.read_csv(timing_summary_path)
    daily = pd.read_csv(daily_weights_path)
    daily["date"] = pd.to_datetime(daily["date"])
    if "mean_abs_dynamic_weight" not in summary or "alpha_idx" not in summary:
        print("[skip] timing summary lacks alpha importance columns")
        return

    top = (
        summary.sort_values("mean_abs_dynamic_weight", ascending=False)
        .head(max_alpha_lines)
        ["alpha_idx"]
        .astype(int)
        .tolist()
    )
    if not top:
        return

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    for idx in top:
        col = f"weight_alpha_{idx}"
        if col in daily:
            ax.plot(daily["date"], daily[col], linewidth=1.6, label=f"alpha {idx}")

    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_title("Top Dynamic Alpha Weights")
    ax.set_xlabel("")
    ax.set_ylabel("dynamic weight")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(frameon=False, ncols=min(3, len(top)))
    _save(fig, fig_dir / "top_dynamic_weights.png")


def _plot_alpha_importance(timing_summary_path: Path, fig_dir: Path) -> None:
    if not timing_summary_path.exists():
        print(f"[skip] missing timing summary: {timing_summary_path}")
        return

    summary = pd.read_csv(timing_summary_path)
    required = {"alpha_idx", "base_weight_w0", "mean_abs_dynamic_weight"}
    if not required.issubset(summary.columns):
        print("[skip] timing summary lacks required columns")
        return

    summary = summary.sort_values("mean_abs_dynamic_weight", ascending=False)
    labels = [f"alpha {int(idx)}" for idx in summary["alpha_idx"]]
    x = np.arange(len(summary))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    ax.bar(
        x - width / 2,
        summary["base_weight_w0"].astype(float),
        width=width,
        color=COLORS["static"],
        label="base w0",
    )
    ax.bar(
        x + width / 2,
        summary["mean_abs_dynamic_weight"].astype(float),
        width=width,
        color=COLORS["dynamic"],
        label="mean abs dynamic weight",
    )
    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_title("Alpha Weight Importance")
    ax.set_ylabel("weight")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.legend(frameon=False)
    _save(fig, fig_dir / "alpha_importance.png")


def run(
    output_prefix: str,
    fig_dir: Optional[str] = None,
    dynamic_label: str = "TE-AlphaGen v1.5",
    static_label: str = "Static w0",
    max_alpha_lines: int = 5,
    dpi: int = 180,
) -> None:
    output_prefix_path = Path(output_prefix)
    figure_dir = Path(fig_dir) if fig_dir is not None else Path(str(output_prefix_path) + "_figures")
    _setup_style(dpi)

    dynamic_report = _load_report(Path(str(output_prefix_path) + "-report.pkl"))
    static_report = _load_report(Path(str(output_prefix_path) + "_static_w0-report.pkl"))
    dynamic_metrics, static_metrics = _load_metrics(output_prefix_path)
    timing_summary_path = Path(str(output_prefix_path) + "-timing-summary.csv")
    daily_weights_path = Path(str(output_prefix_path) + "-timing-daily-weights.csv")

    _plot_cumulative_returns(dynamic_report, static_report, figure_dir, dynamic_label, static_label)
    _plot_excess_returns(dynamic_report, static_report, figure_dir, dynamic_label, static_label)
    _plot_metrics(dynamic_metrics, static_metrics, figure_dir, dynamic_label, static_label)
    _plot_gate_heatmap(daily_weights_path, figure_dir)
    _plot_top_dynamic_weights(timing_summary_path, daily_weights_path, figure_dir, max_alpha_lines)
    _plot_alpha_importance(timing_summary_path, figure_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TE-AlphaGen v1.5 backtest diagnostics.")
    parser.add_argument("--output_prefix", required=True, help="Prefix used by scripts/te_backtest.py.")
    parser.add_argument("--fig_dir", default=None, help="Directory for generated PNG figures.")
    parser.add_argument("--dynamic_label", default="TE-AlphaGen v1.5")
    parser.add_argument("--static_label", default="Static w0")
    parser.add_argument("--max_alpha_lines", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
