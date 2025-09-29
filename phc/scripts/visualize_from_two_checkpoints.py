import os
import gc
import ast
import sys
import uuid
import time
import math
import json
import argparse
import configparser
from datetime import datetime

import joblib
from tqdm import tqdm
from rich_argparse import RichHelpFormatter

import isaacgym  # noqa

import torch
import numpy as np

from smpl_sim.smpllib.smpl_eval import compute_metrics_lite

import pufferlib
import pufferlib.cleanrl
import pufferlib.vector

from puffer_phc import clean_pufferl
from puffer_phc.environment import make as env_creator
import puffer_phc.policy as policy_module
from puffer_phc.flow_matching_utils.policy import FlowMatchingPolicy


def load_structured_config(config_filepath):
    p = configparser.ConfigParser()
    p.read(config_filepath)
    
    cfg = {} 
    known_sections = {"env", "policy", "rnn", "train"}

    for section_name in p.sections():
        current_section_dict = {}
        for key, value_str in p[section_name].items():
            try:
                val = ast.literal_eval(value_str)
            except (ValueError, SyntaxError):
                val = value_str
            current_section_dict[key] = val
        
        if section_name in known_sections:
            cfg[section_name] = current_section_dict
        elif section_name == "base": 
            cfg.update(current_section_dict)
        else: 
            cfg[section_name] = current_section_dict
            
    for sec in known_sections:
        if sec not in cfg:
            cfg[sec] = {}
            
    return cfg


def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args["policy"])
    if rnn_cls is not None:
        policy = rnn_cls(env, policy, **args["rnn"])
        policy = pufferlib.cleanrl.RecurrentPolicy(policy)
    elif "flow_matching" in args and args["flow_matching"]:
        policy = FlowMatchingPolicy(policy)
    else:
        policy = pufferlib.cleanrl.Policy(policy)

    return policy.to(args["train"]["device"])


def init_wandb(args, name, resume=True):
    import wandb

    exp_id = args["wandb_name"] + "-" + datetime.now().strftime("%m%d_%H%M") + "-" + str(uuid.uuid4())[:8]
    wandb.init(
        id=exp_id,
        project=args["wandb_project"],
        allow_val_change=True,
        save_code=True,
        resume=resume,
        config=args,
        name=name,
    )
    return wandb, exp_id


def rollout(vec_env, policy, eval_stats=None):
    # NOTE (Important): Using deterministic action for evaluation
    policy.policy.set_deterministic_action(True)  # Ugly... but...

    obs, _ = vec_env.reset()
    state = None

    ep_cnt = 0
    while True:
        with torch.no_grad():
            obs = torch.as_tensor(obs).to(device)
            if hasattr(policy, "lstm"):
                action, _, _, _, state = policy(obs, state)
            else:
                action, _, _, _ = policy(obs)

            action = action.cpu().numpy().reshape(vec_env.action_space.shape)

        obs, _, done, trunc, info = vec_env.step(action)

        # Check finished envs using dones and truncs
        reset_envs_mask = np.logical_or(done, trunc)

        if hasattr(policy, "lstm"):
            # Reset lstm states for the reset
            reset_envs = torch.logical_or(done, trunc)
            if reset_envs.any():
                state[0][:, reset_envs] = 0
                state[1][:, reset_envs] = 0
        
        # Handle finished environments specifically for 'play' mode (no eval_stats)
        if eval_stats is None and reset_envs_mask.any():
            finished_indices = np.where(reset_envs_mask)[0].tolist()
            # Explicitly call a resampling method if the base env has it.
            # Assumes the method is named 'resample_motions'.
            try:
                # Using keyword arguments for env_method
                vec_env.env.resample_viser_motions()
                # print(f"Explicitly resampled motions for environments: {finished_indices}") # Optional: uncomment for debugging
            except AttributeError:
                 # Method doesn't exist, reset called by step() is likely sufficient.
                 pass
            except Exception as e:
                 # Catch other potential errors during the call
                 print(f"Error calling resample_motions for indices {finished_indices}: {e}")

        # Get episode-related info here
        if len(info) > 0:
            ep_ret = info[0]["episode_return"]
            ep_len = info[0]["episode_length"]
            print(f"Episode cnt: {vec_env.episode_count - ep_cnt}, Reward: {ep_ret:.3f}, Length: {ep_len:.3f}")
            ep_cnt = vec_env.episode_count

        if eval_stats:
            is_done, next_batch = eval_stats.post_step_eval()
            if is_done:
                policy.policy.set_deterministic_action(False)
                break

            if next_batch and state is not None:
                # Reset the states
                state[0][:] = 0
                state[1][:] = 0


def rollout_dual(vec_env, policy1, policy2):
    # Set both policies to deterministic mode
    policy1.policy.set_deterministic_action(True)
    policy2.policy.set_deterministic_action(True)

    obs, _ = vec_env.reset()
    state1 = None
    state2 = None

    ep_cnt = 0
    num_envs_per_policy = vec_env.num_envs // 2
    
    while True:
        with torch.no_grad():
            # Split observations for each policy
            obs1 = obs[:num_envs_per_policy]
            obs2 = obs[num_envs_per_policy:]
            
            # Process policy 1
            obs1_tensor = torch.as_tensor(obs1)
            if hasattr(policy1, "lstm"):
                action1, _, _, _, state1 = policy1(obs1_tensor, state1)
            else:
                action1, _, _, _ = policy1(obs1_tensor)
            
            # Process policy 2
            obs2_tensor = torch.as_tensor(obs2)
            if hasattr(policy2, "lstm"):
                action2, _, _, _, state2 = policy2(obs2_tensor, state2)
            else:
                action2, _, _, _ = policy2(obs2_tensor)

            # Combine actions
            action1 = action1.cpu().numpy().reshape((num_envs_per_policy,) + vec_env.action_space.shape[1:])
            action2 = action2.cpu().numpy().reshape((num_envs_per_policy,) + vec_env.action_space.shape[1:])
            action = np.concatenate([action1, action2])

        # Step environment
        obs, _, done, trunc, info = vec_env.step(action)

        # Handle LSTM states
        if hasattr(policy1, "lstm") or hasattr(policy2, "lstm"):
            reset_envs = torch.logical_or(done, trunc)
            if hasattr(policy1, "lstm") and reset_envs[:num_envs_per_policy].any():
                state1[0][:, reset_envs[:num_envs_per_policy]] = 0
                state1[1][:, reset_envs[:num_envs_per_policy]] = 0
            if hasattr(policy2, "lstm") and reset_envs[num_envs_per_policy:].any():
                state2[0][:, reset_envs[num_envs_per_policy:]] = 0
                state2[1][:, reset_envs[num_envs_per_policy:]] = 0

        # Handle resets
        reset_envs_mask = np.logical_or(done, trunc)
        if reset_envs_mask.any():
            try:
                vec_env.env.resample_viser_motions()
            except (AttributeError, Exception) as e:
                if not isinstance(e, AttributeError):
                    print(f"Error resampling motions: {e}")

        # Print episode info
        if info:  # Check if info list is not empty
            # Process info for each environment
            for env_idx, env_info in enumerate(info):
                if env_info is None or "episode_return" not in env_info:
                    continue
                    
                ep_ret = env_info["episode_return"]
                ep_len = env_info["episode_length"]
                
                # Determine which policy this environment belongs to
                if env_idx < num_envs_per_policy:
                    print(f"Policy 1 - Episode cnt: {ep_cnt}, Reward: {ep_ret:.3f}, Length: {ep_len:.3f}")
                else:
                    print(f"Policy 2 - Episode cnt: {ep_cnt}, Reward: {ep_ret:.3f}, Length: {ep_len:.3f}")
            
            # Only increment episode count if any environment completed an episode
            if any(env_info is not None and "episode_return" in env_info for env_info in info):
                ep_cnt += 1


def compute_weight_stats(state_dict):
    stats = {}
    for name, param in state_dict.items():
        if param.dim() > 0:  # Skip scalar parameters
            stats[name] = {
                'mean': param.mean().item(),
                'std': param.std().item(),
                'min': param.min().item(),
                'max': param.max().item(),
                'shape': list(param.shape),
                'abs_mean': param.abs().mean().item()
            }
    return stats

def print_comparative_stats(stats1, stats2):
    print("\nWeight Statistics Comparison:")
    print("=" * 120)
    print(f"{'Layer Name':<50} {'Stat':<10} {'Checkpoint 1':>20} {'Checkpoint 2':>20} {'Abs Diff':>15}")
    print("-" * 120)
    
    all_layers = sorted(set(stats1.keys()) | set(stats2.keys()))
    
    for layer in all_layers:
        if layer in stats1 and layer in stats2:
            s1, s2 = stats1[layer], stats2[layer]
            
            # Calculate absolute differences
            mean_diff = abs(s1['mean'] - s2['mean'])
            std_diff = abs(s1['std'] - s2['std'])
            abs_mean_diff = abs(s1['abs_mean'] - s2['abs_mean'])
            
            # Print with truncated layer name if too long
            layer_name = layer if len(layer) < 50 else layer[:47] + "..."
            
            print(f"\n{layer_name:<50}")
            print(f"{'':50} {'mean':.<10} {s1['mean']:>20.6f} {s2['mean']:>20.6f} {mean_diff:>15.6f}")
            print(f"{'':50} {'std':.<10} {s1['std']:>20.6f} {s2['std']:>20.6f} {std_diff:>15.6f}")
            print(f"{'':50} {'abs_mean':.<10} {s1['abs_mean']:>20.6f} {s2['abs_mean']:>20.6f} {abs_mean_diff:>15.6f}")
            print(f"{'':50} {'shape':.<10} {str(s1['shape']):>20} {str(s2['shape']):>20}")
        else:
            print(f"\n{layer} present in only one checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RichHelpFormatter, add_help=False)
    parser.add_argument("--config1", default="config.ini", help="Path to config file for the first checkpoint")
    parser.add_argument("--config2", type=str, default=None, help="Path to config file for the second checkpoint. If None, uses config1.")
    parser.add_argument(
        "-m", "--motion-file", type=str, default="sample_data/cmu_mocap_05_06.pkl", help="Path to motion file"
    )
    parser.add_argument("-c1", "--checkpoint-path1", type=str, default=None, help="Path to first pretrained checkpoint")
    parser.add_argument("-c2", "--checkpoint-path2", type=str, default=None, help="Path to second pretrained checkpoint")
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument("--wandb-project", type=str, default="pufferlib")
    parser.add_argument("--full-track", action="store_true", help="Full conditional tracking")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (e.g., cuda:0, cpu)")

    # Add help argument after all other arguments are defined
    parser.add_argument(
        "-h", "--help", default=argparse.SUPPRESS, action="help", help="Show this help message and exit"
    )

    cli_args = parser.parse_args()

    config_path1 = cli_args.config1
    config_path2 = cli_args.config2 if cli_args.config2 else cli_args.config1

    args1 = load_structured_config(config_path1)
    args2 = load_structured_config(config_path2)

    # Apply global CLI overrides
    device_cli_str = cli_args.device # Original string e.g. "cuda:0" or "cpu"
    
    # Parse device_cli_str for environment creation
    # IsaacGym/environment expects separate device_type ("cuda" or "cpu") and device_id (int)
    parsed_env_device_type = "cuda"  # Default
    parsed_env_device_id = 0       # Default

    if device_cli_str == "cpu":
        parsed_env_device_type = "cpu"
        # parsed_env_device_id remains 0, consistent with environment.py defaults
    elif device_cli_str.startswith("cuda"):
        parsed_env_device_type = "cuda" # Type is "cuda"
        if ":" in device_cli_str:
            try:
                parsed_env_device_id = int(device_cli_str.split(":")[1])
            except ValueError:
                print(f"Warning: Could not parse device ID from '{device_cli_str}'. Using default ID 0 for CUDA.")
                # parsed_env_device_id remains 0
        # If just "cuda" (no specific ID), parsed_env_device_id remains 0 (default for cuda)
    else:
        print(f"Warning: Unrecognized device string '{device_cli_str}'. Defaulting to cuda:0 for environment setup.")
        # parsed_env_device_type and parsed_env_device_id remain "cuda" and 0
    
    # Ensure essential nested dicts exist.
    # The 'device' for training in argsX can be the full string like "cuda:0" as torch.device() handles it.
    for arg_set in [args1, args2]:
        for section in ["env", "policy", "rnn", "train"]:
            if section not in arg_set:
                arg_set[section] = {}
    
    args1["train"]["device"] = device_cli_str 
    args2["train"]["device"] = device_cli_str

    if cli_args.motion_file:
        args1["env"]["motion_file"] = cli_args.motion_file
        args2["env"]["motion_file"] = cli_args.motion_file # Apply to both, though env is primarily from args1

    # Create the environment using args1 as the primary source, with specific overrides
    env_config_for_creation = args1["env"].copy()
    # Ensure 'env_name' is present in args1, possibly from [base] section. Fallback if missing.
    env_config_for_creation["name"] = args1.get("env_name", env_config_for_creation.get("name", "Humanoid"))
    
    # Apply parsed device settings for environment creation
    env_config_for_creation["device_type"] = parsed_env_device_type
    env_config_for_creation["device_id"] = parsed_env_device_id
    
    # Overrides for dual rollout
    env_config_for_creation["num_envs"] = 2
    env_config_for_creation["visualization"] = "viser"
    env_config_for_creation["viser_visualize_non_overlap"] = True
    
    vec_env = pufferlib.vector.make(env_creator, env_kwargs=env_config_for_creation)

    # Determine policy and RNN classes from args1 and args2 respectively
    # Fallback to default names if not specified in configs
    policy_name1 = args1.get("policy_name", "BaselinePolicy")
    rnn_name1 = args1.get("rnn_name", None)
    args1["policy"]["num_envs"] = 1
    args1["policy"]["condition_drop_ratio"] = 0.0 if cli_args.full_track else 1.0
    policy_cls1 = getattr(policy_module, policy_name1)
    rnn_cls1 = getattr(policy_module, rnn_name1) if rnn_name1 else None
    
    policy_name2 = args2.get("policy_name", "BaselinePolicy")
    rnn_name2 = args2.get("rnn_name", None)
    args2["policy"]["num_envs"] = 1
    args2["policy"]["condition_drop_ratio"] = 0.0 if cli_args.full_track else 1.0
    policy_cls2 = getattr(policy_module, policy_name2)
    rnn_cls2 = getattr(policy_module, rnn_name2) if rnn_name2 else None
    
    # Pass the original device_cli_str for policy.to(device) and checkpoint loading map_location
    policy1 = make_policy(vec_env.driver_env, policy_cls1, rnn_cls1, args1) # args1 already has train.device = device_cli_str
    policy2 = make_policy(vec_env.driver_env, policy_cls2, rnn_cls2, args2) # args2 already has train.device = device_cli_str

    # Load checkpoints
    checkpoint1_data = None
    if cli_args.checkpoint_path1:
        checkpoint1_data = torch.load(cli_args.checkpoint_path1, map_location=device_cli_str)
        policy1.load_state_dict(checkpoint1_data["state_dict"])
        print(f"Loaded first checkpoint from {cli_args.checkpoint_path1}")

    if cli_args.checkpoint_path2:
        checkpoint2_data = torch.load(cli_args.checkpoint_path2, map_location=device_cli_str)
        policy2.load_state_dict(checkpoint2_data["state_dict"])
        print(f"Loaded second checkpoint from {cli_args.checkpoint_path2}")

        # Compare weight statistics if both checkpoints are loaded
        if checkpoint1_data:
            stats1 = compute_weight_stats(checkpoint1_data["state_dict"])
            stats2 = compute_weight_stats(checkpoint2_data["state_dict"])
            print_comparative_stats(stats1, stats2)
        else:
            print("First checkpoint not loaded, cannot compare statistics.")
            
    # WandB Initialization (using args1 for config logging)
    if cli_args.track:
        # Pass a dictionary that combines CLI args and structured config from args1
        config_to_log = vars(cli_args).copy()
        config_to_log.update({"config1_params": args1, "config2_params": args2 if cli_args.config2 else "Same as config1"})
        
        import wandb # Import wandb only if tracking
        exp_id = args1.get("wandb_name", "compare_ckpt") + "-" + datetime.now().strftime("%m%d_%H%M") + "-" + str(uuid.uuid4())[:8]
        wandb.init(
            id=exp_id,
            project=cli_args.wandb_project,
            allow_val_change=True,
            save_code=True,
            resume="allow", # Changed from True to allow for new runs
            config=config_to_log,
            name=f"compare_{policy_name1}_vs_{policy_name2}"
        )

    # Set termination distances and run dual rollout
    vec_env.env.set_termination_distances(10) # Example, consider making this configurable
    rollout_dual(vec_env, policy1, policy2)