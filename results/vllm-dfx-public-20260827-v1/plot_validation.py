from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(path: Path) -> float:
    return float(path.read_text(encoding="utf-8").strip().rsplit(" ", 1)[-1])


def plot_oom(root: Path, output: Path) -> None:
    utils = [0.35, 0.40, 0.41]
    kv_gib = [0.0, 1.01, 1.25]
    outcomes = ["no cache blocks", "8K rejected", "healthy"]
    colors = ["#fa5252", "#fd7e14", "#40c057"]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    bars = ax.bar([str(v) for v in utils], kv_gib, color=colors, width=0.58)
    ax.axhline(1.12, color="#1c7ed6", linestyle="--", label="8K minimum: 1.12 GiB")
    for bar, value, outcome in zip(bars, kv_gib, outcomes):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.2f} GiB\n{outcome}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xlabel("gpu_memory_utilization")
    ax.set_ylabel("Available KV cache (GiB)")
    ax.set_title("Startup capacity boundary: Qwen3-4B BF16, max_model_len=8192")
    ax.set_ylim(0, 1.55)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_preemption(root: Path, output: Path) -> None:
    base = root / "preemption"
    labels = ["11.59 GiB KV", "6.00 GiB KV"]
    high = load(base / "high-kv-c8/bench.json")
    low = load(base / "low-kv-c8/bench.json")
    benches = [high, low]
    preemptions = [
        metric_value(base / "high-kv-c8/metrics-after.txt")
        - metric_value(base / "high-kv-c8/metrics-before.txt"),
        metric_value(base / "low-kv-c8/metrics-after.txt")
        - metric_value(base / "low-kv-c8/metrics-before.txt"),
    ]
    colors = ["#339af0", "#ff922b"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.3))
    panels = [
        ([b["output_throughput"] for b in benches], "Output throughput", "tok/s"),
        ([b["p99_ttft_ms"] / 1000 for b in benches], "TTFT P99", "seconds"),
        (preemptions, "Preemptions", "count"),
    ]
    for ax, (values, title, ylabel) in zip(axes, panels):
        bars = ax.bar(labels, values, color=colors, width=0.58)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values + [1]) * 0.025,
                f"{value:.2f}",
                ha="center",
                fontsize=9,
            )
    fig.suptitle("Matched workload: 8 requests × (7000 input + 512 output tokens)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def regression_slope(xs: list[float], ys: list[float]) -> float:
    x_mean, y_mean = mean(xs), mean(ys)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum(
        (x - x_mean) ** 2 for x in xs
    )


def plot_soak(root: Path, output: Path) -> dict[str, float]:
    base = root / "soak"
    rows = [json.loads(line) for line in (base / "dfx/timeline.jsonl").read_text().splitlines()]
    bench = load(base / "bench-r2-c8-i512-o128.json")
    start = min(bench["start_times"])
    end = max(bench["start_times"])
    active = [row for row in rows if start <= row["monotonic_seconds"] <= end]
    minutes = [(row["monotonic_seconds"] - start) / 60 for row in active]
    rss = [row["process"]["rss_kib"] / 1024 for row in active]
    gpu = [row["gpus"][0]["memory_used_mib"] for row in active]
    kv = [row["metrics"]["vllm:kv_cache_usage_perc"] * 100 for row in active]

    fit_rows = [row for row in active if row["monotonic_seconds"] >= start + 60]
    fit_x = [row["monotonic_seconds"] for row in fit_rows]
    fit_y = [row["process"]["rss_kib"] / 1024 for row in fit_rows]
    slope_per_hour = regression_slope(fit_x, fit_y) * 3600

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
    axes[0].plot(minutes, rss, color="#1971c2", label="API RSS")
    axes[0].set_ylabel("RSS (MiB)")
    axes[0].set_title(f"10-minute soak: RSS slope after first 60s = {slope_per_hour:.2f} MiB/hour")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(minutes, gpu, color="#2f9e44", label="GPU memory (MiB)")
    axes[1].set_ylabel("GPU memory (MiB)")
    axes[1].set_xlabel("Minutes from workload start")
    axes[1].grid(alpha=0.25)
    kv_axis = axes[1].twinx()
    kv_axis.plot(minutes, kv, color="#f08c00", alpha=0.65, label="KV usage")
    kv_axis.set_ylabel("KV usage (%)")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = kv_axis.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return {
        "rss_slope_mib_per_hour_after_60s": slope_per_hour,
        "rss_mib_first_active": rss[0],
        "rss_mib_last_active": rss[-1],
        "gpu_mib_min": min(gpu),
        "gpu_mib_max": max(gpu),
        "kv_usage_percent_max": max(kv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_oom(args.results, args.output / "oom-boundary.png")
    plot_preemption(args.results, args.output / "preemption-cost.png")
    soak = plot_soak(args.results, args.output / "soak-curve.png")
    (args.output / "plot-summary.json").write_text(
        json.dumps(soak, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

