# Flow Matching Policy Gradients

This repo implements **Flow Policy Optimization (FPO)** for online reinforcement learning in continuous action spaces.

Please see the [blog](https://flowreinforce.github.io/) and [paper](todo) for more details.

Different tasks in our paper are currently organized into folders:

### Gridworld

`gridworld/` contains code for our toy gridworld experiments, which are based on the
[PPO grid-world repo by Eric Yu](https://github.com/EricYu97/grid-world-ppo).

### MuJoCo Playground

`playground/` contains code for our DeepMind Control Suite experiments, which are based on the
[MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and [Brax](https://github.com/google/brax).
