# Cost-aware cloud autoscaling

CSC 496/596 project. A Gymnasium simulator of one cloud service, a DQN policy trained with Stable-Baselines3, and two baselines: a threshold autoscaler and a static peak allocation.

## Run

```bash
python evaluate.py
python train.py --timesteps 120000
python evaluate.py --model results/dqn_cloud.zip
```

`evaluate.py` with no model runs only the baselines. `train.py` trains DQN, then writes:

- `results/learning_curve.png` — violation rate and server-minutes during training
- `results/comparison.png` — DQN against both baselines on the daily curve and the spiky curve
- `results/episode_trace.png` — one spiky episode, ready servers against load
- `results/dqn_cloud.zip` — the trained policy

## Simulator

Each step is one minute. A requested server serves traffic only after 5 minutes. The agent can add one server, remove one server, or hold. The reward is

```
r = -(servers + 20 * target_miss + scaling_action) / 20
```

Server-minutes, the fraction of steps over the response-time target, mean response time, and scaling actions are logged unscaled. The division by 20 only keeps the learning signal in a range DQN can fit. It does not change which action is best.
