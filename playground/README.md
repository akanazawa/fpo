# FPO + MuJoCo Playground

This directory contains implementations of FPO and PPO in MuJoCo Playground.
It's adapted from and inherits implementation details from
[Brax](https://github.com/google/brax)'s PPO implementation.

The directory is structured as follows:

```
.
├── train_fpo.ipynb          - Interactive notebook for FPO training.
├── train_ppo.ipynb          - Interactive notebook for PPO training.
│
├── src/flow_policy/
│   ├── fpo.py               - FPO algorithm implementation.
│   ├── ppo.py               - PPO algorithm implementation.
│   ├── networks.py          - Neural network architectures.
│   ├── rollouts.py          - Trajectory collection utilities.
│   └── math_utils.py        - Mathematical utilities.
│
├── renders/                 - Example rollout videos.
│   ├── fpo_rollout.mp4
│   └── ppo_rollout.mp4
│
├── scripts/
│   ├── train_fpo.py         - Training script for Flow Policy Optimization.
│   └── train_ppo.py         - Training script for Proximal Policy Optimization.
│
└── pyproject.toml           - Python dependencies/package metadata.
```

## Setup

This package has been tested in Python 3.12 on an RTX 4090. We generally use `conda` for environment management.

1. Install JAX into your environment with CUDA support:

   ```bash
   pip install "jax[cuda12]==0.7.0"
   ```

2. Install the package:

   ```bash
   pip install -e .  # From the `fpo/playground` directory.
   ```

3. Start with the notebooks for interactive training:

   - `train_fpo.ipynb` - Flow Policy Optimization
   - `train_ppo.ipynb` - Proximal Policy Optimization

4. For training with wandb logging, run the scripts with `--help` to see available options:
   ```bash
   python scripts/train_fpo.py --help
   python scripts/train_ppo.py --help
   ```

## Results

The final cell of each training notebook renders a result from a trained policy.

**Examples on `CheetahRun`, from `train_fpo.ipynb`:**

https://github.com/user-attachments/assets/77d61984-c9a0-44e2-b9c0-9fa9ff934287

https://github.com/user-attachments/assets/f305430a-35d2-46c0-870e-99e63dcb6eda

**Examples on `CheetahRun`, from `train_ppo.ipynb`:**

https://github.com/user-attachments/assets/99a4eb94-6edb-4391-90fb-d986d3c06211

https://github.com/user-attachments/assets/bacda075-d58b-41f6-a731-e03ba68b3bec
