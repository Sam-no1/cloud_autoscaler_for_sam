"""Gymnasium environment for cost-aware cloud autoscaling.

Each step is one minute. A newly requested server starts serving only after
``STARTUP_DELAY`` steps. The learning reward is the proposal reward divided
by ``REWARD_SCALE`` so Q-values stay in a range DQN can fit. The unscaled
cost, violation, and scaling terms are reported in ``info``.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

ADD = 2
HOLD = 1
REMOVE = 0

EPISODE_LEN = 240
HISTORY = 12
CAP_PER_SERVER = 10.0
MIN_SERVERS = 1
MAX_SERVERS = 12
STARTUP_DELAY = 5
TARGET_BACKLOG = 1.0  # minutes of queued work
ARRIVAL_NORM = 130.0
NOISE_STD = 2.0

# r = -(c * n + λ * v + μ * |Δn|)  / REWARD_SCALE
COST_PER_SERVER = 1.0
VIOLATION_PENALTY = 20.0
SCALE_PENALTY = 1.0
REWARD_SCALE = 20.0

# Design peaks of the two workload envelopes, in servers.
PEAK_SERVERS = {"daily": 8, "spiky": 12}


def daily_mean(t: int, length: int = EPISODE_LEN) -> float:
    """One smooth cycle: about 20 requests at the edges, about 70 in the middle."""
    phase = 2.0 * np.pi * (t / length) - np.pi / 2.0
    return float(45.0 + 25.0 * np.sin(phase))


class CloudAutoscaleEnv(gym.Env):
    """One service, a queue, and servers that take time to boot."""

    metadata = {"render_modes": []}

    def __init__(self, workload: str = "mix"):
        super().__init__()
        if workload not in {"daily", "spiky", "mix"}:
            raise ValueError(f"unknown workload {workload!r}")
        self.workload_mode = workload
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(HISTORY + 5,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._rng = np.random.default_rng()
        self.trace = np.zeros(EPISODE_LEN, dtype=np.float64)
        self.workload = "daily"
        self.t = 0
        self.n_ready = MIN_SERVERS
        self.boot_timers: list[int] = []
        self.queue = 0.0
        self.history = np.zeros(HISTORY, dtype=np.float64)
        self.last_response = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        workload = options.get("workload", self.workload_mode)
        if workload == "mix":
            workload = "spiky" if self._rng.random() < 0.5 else "daily"
        self.workload = workload
        self.trace = self._build_trace(workload)
        self.t = 0
        self.n_ready = int(options.get("initial_servers", 4))
        self.n_ready = int(np.clip(self.n_ready, MIN_SERVERS, MAX_SERVERS))
        self.boot_timers = []
        self.queue = 0.0
        self.last_response = 0.0
        start = daily_mean(0) / ARRIVAL_NORM
        self.history = np.full(HISTORY, start, dtype=np.float64)
        return self._obs(), self._info(arrivals=0.0, violation=0, scaled=0, response=0.0, delta_n=0)

    def step(self, action):
        action = int(action)
        delta_n = self._apply_action(action)
        self._tick_boots()

        arrivals = float(self.trace[self.t]) if self.t < EPISODE_LEN else 0.0
        service = self.n_ready * CAP_PER_SERVER
        demand = self.queue + arrivals
        served = min(demand, service) if service > 0 else 0.0
        self.queue = demand - served
        if service <= 0:
            response = float(EPISODE_LEN)
        else:
            response = self.queue / service
        violation = int(response > TARGET_BACKLOG)
        self.last_response = response

        n_total = self.n_ready + len(self.boot_timers)
        raw = -(
            COST_PER_SERVER * n_total
            + VIOLATION_PENALTY * violation
            + SCALE_PENALTY * abs(delta_n)
        )
        reward = raw / REWARD_SCALE

        self.history = np.roll(self.history, -1)
        self.history[-1] = arrivals / ARRIVAL_NORM
        self.t += 1
        truncated = self.t >= EPISODE_LEN
        info = self._info(arrivals, violation, abs(delta_n), response, delta_n)
        info["raw_reward"] = raw
        return self._obs(), float(reward), False, truncated, info

    def _apply_action(self, action: int) -> int:
        n_total = self.n_ready + len(self.boot_timers)
        if action == ADD and n_total < MAX_SERVERS:
            self.boot_timers.append(STARTUP_DELAY)
            return 1
        if action == REMOVE:
            if self.n_ready > MIN_SERVERS:
                self.n_ready -= 1
                return -1
            if self.boot_timers:
                self.boot_timers.pop()
                return -1
        return 0

    def _tick_boots(self) -> None:
        still = []
        for remaining in self.boot_timers:
            remaining -= 1
            if remaining <= 0:
                self.n_ready += 1
            else:
                still.append(remaining)
        self.boot_timers = still

    def _build_trace(self, workload: str) -> np.ndarray:
        trace = np.array([daily_mean(t) for t in range(EPISODE_LEN)], dtype=np.float64)
        trace += self._rng.normal(0.0, NOISE_STD, size=EPISODE_LEN)
        if workload == "spiky":
            cursor = 35
            for _ in range(3):
                if cursor >= EPISODE_LEN - 30:
                    break
                start = cursor + int(self._rng.integers(0, 20))
                # A short ramp, a hold, then a drop. The ramp is visible in the
                # arrival history before the queue has built up.
                extras = [8, 22, 40, 48, 48, 48, 48, 36, 18, 8]
                for i, extra in enumerate(extras):
                    idx = start + i
                    if idx < EPISODE_LEN:
                        trace[idx] += extra
                cursor += 70
        return np.clip(trace, 0.0, None)

    def _obs(self) -> np.ndarray:
        service = max(self.n_ready * CAP_PER_SERVER, 1.0)
        util = min((self.history[-1] * ARRIVAL_NORM) / service, 1.5) / 1.5
        if self.boot_timers:
            soonest = min(self.boot_timers) / STARTUP_DELAY
        else:
            soonest = 0.0
        queue_norm = min(self.queue / (MAX_SERVERS * CAP_PER_SERVER), 1.0)
        response_norm = min(self.last_response / 5.0, 1.0)
        tail = np.array(
            [
                util,
                self.n_ready / MAX_SERVERS,
                len(self.boot_timers) / MAX_SERVERS,
                soonest,
                max(queue_norm, response_norm),
            ],
            dtype=np.float64,
        )
        obs = np.concatenate([np.clip(self.history, 0.0, 1.0), tail])
        return obs.astype(np.float32)

    def _info(self, arrivals, violation, scaled, response, delta_n) -> dict:
        service = self.n_ready * CAP_PER_SERVER
        if service <= 0:
            utilization = 1.0
        else:
            utilization = float(arrivals / service) if self.t > 0 or arrivals else float(
                (self.history[-1] * ARRIVAL_NORM) / service
            )
        return {
            "workload": self.workload,
            "arrivals": float(arrivals),
            "n_ready": int(self.n_ready),
            "n_booting": int(len(self.boot_timers)),
            "n_total": int(self.n_ready + len(self.boot_timers)),
            "queue": float(self.queue),
            "utilization": float(utilization),
            "response_time": float(response),
            "violation": int(violation),
            "scaled": int(scaled),
            "delta_n": int(delta_n),
            "t": int(self.t),
        }
