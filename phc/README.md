# FPO for Humanoid Control

This repository implements **FPO** for humanoid control tasks, as described in the paper ["Flow Matching Policy Gradients"](https://arxiv.org/pdf/2507.21053) and our [blog post](https://flowreinforce.github.io/). It's built upon the great work of [Puffer PHC](https://github.com/kywch/puffer-phc).

## Installation

### Prerequisites

Most of the dependencies are the same with [Puffer PHC](https://github.com/kywch/puffer-phc), except that we use Viser for visualization.

1. **Install pixi** (if not already installed):
    ```bash
    curl -fsSL https://pixi.sh/install.sh | bash
    ```

2. **Setup virtual environment and install dependencies**:
    ```bash
    pixi shell
    pip install viser open3d torchdiffeq
    ```

3. **Install Isaac Gym**:
    - Download Isaac Gym from [NVIDIA Developer](https://developer.nvidia.com/isaac-gym)
    - Install inside the virtual environment:
    ```bash
    cd <isaac_gym_directory>/python
    pip install -e .
    ```

4. **Install gymtorch** (for debugging):
    ```bash
    pixi run build_gymtorch
    ```

5. **Test installation**:
    ```bash
    pixi run test_deps
    ```

6. **Download SMPL parameters**:
    - Download from [SMPL](https://smpl.is.tue.mpg.de/)
    - Extract to `smpl/` folder
    - Rename files:
      - `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl` → `SMPL_NEUTRAL.pkl`
      - `basicmodel_m_lbs_10_207_0_v1.1.0.pkl` → `SMPL_MALE.pkl`
      - `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` → `SMPL_FEMALE.pkl`

## Usage

### Preparing Motion Data

Convert and inspect motion data from the [AMASS](https://amass.is.tue.mpg.de/) dataset. Please refer to [PHC](https://github.com/ZhengyiLuo/PHC) for more details. We also provide a copy of the processed motion data [here](https://drive.google.com/file/d/1Yc1UCT63BHVohYHwOeTg8mqJMTN3Y124/view?usp=sharing).

```bash
# Convert AMASS data
python scripts/convert_amass_data.py

# Visualize motion data
python scripts/vis_motion_mj.py
```

### Training FPO Policies

Train a flow-based policy using the FPO algorithm:

```bash
python scripts/train.py --config configs/fpo_hand.ini -m ./amass/amass_train_11313_upright.pkl --track
```

**Key Configuration Parameters** (in `configs/fpo.ini`):
- `policy_name = FlowMatchingPolicy`: Uses flow matching policy
- `flow_matching = True`: Enables FPO training
- `solver_step_size = 0.1`: Flow integration step size
- `perturb_action_std = 0.05`: Action perturbation for exploration
- `hand_track = True`: Use both hand and body positions as the goal condition.
- `root_track = True`: Use root position as the goal condition.
- `condition_drop_ratio = 1.0`: Drop the goal condition that are not trackedby 100%.

### Comparing with Gaussian Baselines

We use [Viser](https://github.com/nerfstudio-project/viser) to visualize motions predicted by FPO and Gaussian policies:

```bash
python scripts/visualize_from_two_checkpoints.py \
    --config1 configs/fpo_hand.ini \
    --config2 configs/gaussian_baseline_hand.ini \
    --checkpoint-path1 experiments/fpo_hand-0923_0622-849d810a/model_009500.pt \
    --checkpoint-path2 experiments/gaussian_baseline_hand-0923_0622-c7575bb4/model_009500.pt \
    -m ./amass/amass_train_11313_upright.pkl
```


https://github.com/user-attachments/assets/e7127e9a-ec2f-4c2c-8d7b-0f0d549be975



## Configuration Details

We adopt most configurations from [Puffer PHC config](https://github.com/kywch/puffer-phc/blob/main/config.ini) to demonstrate a drop-in replacement for FPO in PHC. The main differences are:

- **Flow Matching**: `flow_matching = True` enables FPO training
- **Parameterization**: `parameterization = data` makes the flow network to output data. The alternative is `parameterization = velocity` which outputs velocity. We found that velocity parameterization is more stable but normally leads to lower performance, probably due to under-tuned hyperparameters.
- **timestep sampling strategy**: `sample_t_strategy = lognormal` samples timesteps from a lognormal distribution. The alternative is `sample_t_strategy = uniform` which samples timesteps from a uniform distribution.
- **Sampling**: 0.1 for flow integration
- **Perturbation**: Action perturbation with std=0.05. This is important for data parameterization since the network is zero-initialized.
- **Goal Condition**: `hand_track = True` uses both hand and body positions as the goal condition. `root_track = True` uses root position as the goal condition. `condition_drop_ratio = 1.0` drops the goal condition that are not trackedby 100%.
- **Stability**: `max_grad_norm = 1` is used for full conditioning, which we found necessary for stabliing the FPO training, but not necessary for under-conditioning settings.


## References

This repository builds upon the following excellent work:

- **Puffer PHC**: [Puffer PHC](https://github.com/kywch/puffer-phc) - A clean and simplified version of PHC
- **Perpetual Humanoid Control**: [PHC](https://github.com/ZhengyiLuo/PHC) - Original humanoid control framework
- **Isaac Gym**: [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) - Physics simulation
- **PufferLib**: [PufferLib](https://github.com/PufferAI/PufferLib) - RL training framework

Please follow the respective licenses for usage.
