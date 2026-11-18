"""Threshold autoscaler and static peak allocation.

Both policies act through the same environment as the learned agent, so they
face the same startup delay and the same capacity limits.
"""

from __future__ import annotations

from cloud_env import ADD, HOLD, MAX_SERVERS, MIN_SERVERS, PEAK_SERVERS, REMOVE

HIGH_UTIL = 0.70
LOW_UTIL = 0.35
SUSTAIN_STEPS = 2
COOLDOWN_STEPS = 2


class ThresholdAutoscaler:
    """Add one server after utilization stays high, remove one after it stays low."""

    def __init__(self):
        self.high_run = 0
        self.low_run = 0
        self.cooldown = 0

    def reset(self) -> None:
        self.high_run = 0
        self.low_run = 0
        self.cooldown = 0

    def act(self, obs, info) -> int:
        util = info["utilization"]
        if util > HIGH_UTIL:
            self.high_run += 1
            self.low_run = 0
        elif util < LOW_UTIL:
            self.low_run += 1
            self.high_run = 0
        else:
            self.high_run = 0
            self.low_run = 0

        action = HOLD
        if self.cooldown <= 0 and self.high_run >= SUSTAIN_STEPS:
            action = ADD
        elif self.cooldown <= 0 and self.low_run >= SUSTAIN_STEPS:
            action = REMOVE

        if action != HOLD:
            self.high_run = 0
            self.low_run = 0
            self.cooldown = COOLDOWN_STEPS
        elif self.cooldown > 0:
            self.cooldown -= 1
        return action


class StaticPeak:
    """Hold a fixed capacity sized to the workload envelope."""

    def __init__(self, workload: str):
        self.peak = PEAK_SERVERS[workload]
        self.workload = workload

    def reset(self) -> None:
        return None

    def act(self, obs, info) -> int:
        n_total = info["n_total"]
        if n_total < self.peak and n_total < MAX_SERVERS:
            return ADD
        if n_total > self.peak and n_total > MIN_SERVERS:
            return REMOVE
        return HOLD
