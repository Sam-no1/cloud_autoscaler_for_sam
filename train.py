"""Train a DQN autoscaler and write the mid-project plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from cloud_env import CloudAutoscaleEnv
from evaluate import RESULTS, evaluate_all, plot_comparison, plot_learning_curve, plot_trace, print_table, summarize, run_episode

ROOT = Path(__file__).resolve().parent


class EvalHistoryCallback(BaseCallback):
    """Record cost and violation rate on fixed traces during training."""

    def __init__(self, freq: int, seeds: int = 5):
        super().__init__()
        self.freq = freq
        self.seeds = seeds
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.freq != 0:
            return True
        row = {"timesteps": int(self.num_timesteps)}
        for workload in ("daily", "spiky"):
            episodes = [
                run_episode(workload, 1000 + seed, _ModelPolicy(self.model), initial_servers=4)
                for seed in range(self.seeds)
            ]
            row[workload] = summarize(episodes)
        self.history.append(row)
        spiky = row["spiky"]
        print(
            f"step {row['timesteps']:>7}  "
            f"daily viol {row['daily']['violation_rate']*100:5.1f}%  "
            f"cost {row['daily']['server_minutes']:.0f}  "
            f"spiky viol {spiky['violation_rate']*100:5.1f}%  "
            f"cost {spiky['server_minutes']:.0f}"
        )
        return True


class _ModelPolicy:
    def __init__(self, model):
        self.model = model

    def reset(self) -> None:
        return None

    def act(self, obs, info) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def train(timesteps: int, seed: int, eval_freq: int) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = Monitor(CloudAutoscaleEnv(workload="mix"))
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.4,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        seed=seed,
        verbose=0,
        device="cpu",
    )
    callback = EvalHistoryCallback(freq=eval_freq, seeds=5)
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
    model_path = RESULTS / "dqn_cloud"
    model.save(model_path)
    history_path = RESULTS / "learning_history.json"
    history_path.write_text(json.dumps(callback.history, indent=2))
    plot_learning_curve(callback.history, RESULTS / "learning_curve.png")

    results = evaluate_all(model=model, n_seeds=10)
    print_table(results)
    plot_comparison(results, RESULTS / "comparison.png")
    plot_trace(results, RESULTS / "episode_trace.png")
    print(f"saved model to {model_path}.zip")
    print(f"wrote plots in {RESULTS}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    args = parser.parse_args()
    train(args.timesteps, args.seed, args.eval_freq)


if __name__ == "__main__":
    main()
