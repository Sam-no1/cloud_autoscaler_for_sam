"""Run the learned policy and the baselines on the same traces and save plots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from baselines import StaticPeak, ThresholdAutoscaler
from cloud_env import CloudAutoscaleEnv, EPISODE_LEN, PEAK_SERVERS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

WORKLOADS = ("daily", "spiky")
N_SEEDS = 10
EVAL_SEED0 = 1000


def run_episode(workload: str, seed: int, policy, initial_servers: int) -> dict:
    env = CloudAutoscaleEnv(workload=workload)
    obs, info = env.reset(seed=seed, options={"workload": workload, "initial_servers": initial_servers})
    policy.reset()
    rows = []
    done = False
    while not done:
        action = policy.act(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append(info)
        done = terminated or truncated
    rows = rows[:EPISODE_LEN]
    return {
        "workload": workload,
        "seed": seed,
        "server_minutes": float(sum(r["n_total"] for r in rows)),
        "violation_rate": float(np.mean([r["violation"] for r in rows])),
        "mean_response_time": float(np.mean([r["response_time"] for r in rows])),
        "scale_actions": int(sum(r["scaled"] for r in rows)),
        "trace": rows,
    }


class LearnedPolicy:
    def __init__(self, model):
        self.model = model

    def reset(self) -> None:
        return None

    def act(self, obs, info) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def summarize(episodes: list[dict]) -> dict:
    keys = ("server_minutes", "violation_rate", "mean_response_time", "scale_actions")
    out = {}
    for key in keys:
        vals = np.array([ep[key] for ep in episodes], dtype=np.float64)
        out[key] = float(vals.mean())
        out[key + "_std"] = float(vals.std())
    return out


def evaluate_all(model=None, n_seeds: int = N_SEEDS) -> dict:
    policies = {
        "Threshold": (ThresholdAutoscaler, 4),
        "Static peak": None,  # built per workload
    }
    if model is not None:
        policies = {"DQN": (lambda: LearnedPolicy(model), 4), **policies}

    results = {}
    for workload in WORKLOADS:
        results[workload] = {}
        for name, spec in policies.items():
            episodes = []
            for seed in range(EVAL_SEED0, EVAL_SEED0 + n_seeds):
                if name == "Static peak":
                    policy = StaticPeak(workload)
                    initial = PEAK_SERVERS[workload]
                else:
                    factory, initial = spec
                    policy = factory()
                episodes.append(run_episode(workload, seed, policy, initial))
            results[workload][name] = {
                "summary": summarize(episodes),
                "episodes": episodes,
            }
    return results


def print_table(results: dict) -> None:
    print(f"{'workload':<8} {'policy':<14} {'server-min':>12} {'violations':>12} {'resp. time':>12} {'scales':>8}")
    for workload, policies in results.items():
        for name, payload in policies.items():
            s = payload["summary"]
            print(
                f"{workload:<8} {name:<14} "
                f"{s['server_minutes']:>12.1f} "
                f"{s['violation_rate']*100:>11.1f}% "
                f"{s['mean_response_time']:>12.3f} "
                f"{s['scale_actions']:>8.1f}"
            )


def plot_comparison(results: dict, path: Path) -> None:
    policies = list(next(iter(results.values())).keys())
    # Put DQN last if present so the baseline pair reads left to right.
    preferred = [p for p in ("Static peak", "Threshold", "DQN") if p in policies]
    policies = preferred
    x = np.arange(len(WORKLOADS))
    width = 0.8 / len(policies)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    metrics = (
        ("server_minutes", "Server-minutes (lower is cheaper)"),
        ("violation_rate", "Steps over the response-time target"),
    )
    for ax, (key, title) in zip(axes, metrics):
        tops = []
        for i, name in enumerate(policies):
            means = [results[w][name]["summary"][key] for w in WORKLOADS]
            stds = [results[w][name]["summary"][key + "_std"] for w in WORKLOADS]
            tops.extend(m + s for m, s in zip(means, stds))
            offset = (i - (len(policies) - 1) / 2) * width
            ax.bar(x + offset, means, width=width * 0.92, yerr=stds, capsize=3, label=name)
        ax.set_xticks(x, ["Daily curve", "Daily curve + spikes"])
        ax.set_title(title)
        if key == "violation_rate":
            ax.set_ylim(0, max(0.05, max(tops) * 1.45))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_trace(results: dict, path: Path, seed: int | None = None) -> None:
    """One spiky episode: arrivals against ready servers for each policy."""
    fig, ax = plt.subplots(figsize=(9.2, 3.8))
    shown_arrivals = False
    for name, payload in results["spiky"].items():
        if seed is None:
            episode = payload["episodes"][0]
            seed = episode["seed"]
        else:
            episode = next(ep for ep in payload["episodes"] if ep["seed"] == seed)
        t = np.arange(len(episode["trace"]))
        ready = [row["n_ready"] for row in episode["trace"]]
        ax.plot(t, ready, label=f"{name} ready servers", linewidth=1.6)
        if not shown_arrivals:
            arrivals = np.array([row["arrivals"] for row in episode["trace"]]) / 10.0
            ax.plot(t, arrivals, color="0.45", linewidth=1.0, label="Arrivals / 10 (servers of load)")
            shown_arrivals = True
    ax.set_xlabel("Minute")
    ax.set_ylabel("Servers")
    ax.set_title(f"One spiky episode (seed {seed})")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning_curve(history: list[dict], path: Path) -> None:
    if not history:
        return
    steps = [row["timesteps"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    for ax, workload, title in (
        (axes[0], "daily", "Daily curve"),
        (axes[1], "spiky", "Daily curve + spikes"),
    ):
        viol = [row[workload]["violation_rate"] for row in history]
        cost = [row[workload]["server_minutes"] for row in history]
        ax.plot(steps, viol, color="C0", label="Violation rate")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Training steps")
        ax.set_ylabel("Violation rate", color="C0")
        ax.tick_params(axis="y", colors="C0")
        ax.set_title(title)
        twin = ax.twinx()
        twin.plot(steps, cost, color="C1", label="Server-minutes")
        twin.set_ylabel("Server-minutes", color="C1")
        twin.tick_params(axis="y", colors="C1")
        lines = ax.get_lines() + twin.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    args = parser.parse_args()
    model = None
    if args.model is not None:
        from stable_baselines3 import DQN

        model = DQN.load(args.model)
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = evaluate_all(model=model, n_seeds=args.seeds)
    print_table(results)
    plot_comparison(results, RESULTS / "comparison.png")
    plot_trace(results, RESULTS / "episode_trace.png")
    print(f"wrote {RESULTS / 'comparison.png'}")
    print(f"wrote {RESULTS / 'episode_trace.png'}")


if __name__ == "__main__":
    main()
