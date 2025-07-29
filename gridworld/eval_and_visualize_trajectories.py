#!/usr/bin/env python3
"""
Combined script for evaluating policies from fixed start points and visualizing trajectories.

This script combines the functionality of eval_fixed_starts.py and visualize_fixed_traj.py
into a single workflow. It can either:
1. Evaluate a policy and then visualize the results (default)
2. Only evaluate (--no-visualize)
3. Only visualize existing trajectory data (--visualize-only)
"""
import sys
import argparse
import time
import os
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from utils.gridworld import GridWorldEnv
from utils.eval_policy import _log_summary


def evaluate_trajectories(args):
    """Evaluate policy from fixed start points and save trajectory data."""
    print(f"Testing {args.actor_model}", flush=True)
    if not args.actor_model:
        print("Didn't specify model file. Exiting.", flush=True)
        sys.exit(0)

    # build env & policy
    env = GridWorldEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
        
    # Load policy
    if args.method == "ppo":
        policy = FeedForwardNN(obs_dim, act_dim).to(device)
    elif args.method == "fpo":
        policy = DiffusionPolicy(in_dim=obs_dim + act_dim + 1, out_dim=act_dim, device=device)
    else:
        raise ValueError(f"Unsupported method: {args.method}")
        
    policy.load_state_dict(
        torch.load(args.actor_model, map_location=device)
    )
    policy.eval()

    # Fixed start points used in the paper figure
    start_points = [
        (17, 12),
        (11, 13),
        (6, 9),
        (20, 17),
    ]

    # derive default output filename based on model name
    base = os.path.splitext(os.path.basename(args.actor_model))[0]
    out_path = args.output or f"{base}_fixed_traj.pkl"

    all_data = []  # will store dicts of {start_idx, start, traj, ep_len, ep_ret}
    ep_counter = 0

    for sid, start in enumerate(start_points):
        for epi in range(args.episodes):
            obs, _ = env.reset()
            # set starting position
            env.pos = np.array(start, dtype=int)
            
            obs = env._get_obs()  # update obs based on new position
            
            print(f"starting at {obs}")
            done = False
            t = 0
            ep_ret = 0.0
            traj = [tuple(env.pos)]
            actions = []
            
            while not done:
                t += 1
                if args.render:
                    env.render()
                    time.sleep(args.sleep)

                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                    
                # deterministic action from policy
                if args.method == "ppo":
                    a = policy(obs_tensor).detach().cpu().numpy()
                elif args.method == "fpo":
                    a = policy.sample_action(obs_tensor).cpu().numpy()                
                
                # add optional Gaussian noise
                if args.noise > 0.0:
                    print('adding noise')
                    a = a + np.random.normal(scale=args.noise, size=a.shape)
                    # clip to valid action range
                    a = np.clip(a, env.action_space.low, env.action_space.high)

                obs, rew, term, trunc, _ = env.step(a)
                done = term or trunc
                ep_ret += rew
                # record position
                traj.append(tuple(env.pos))
                actions.append(tuple(a))

            # log summary
            _log_summary(ep_len=t,
                         ep_ret=ep_ret,
                         ep_num=ep_counter)
            ep_counter += 1

            # save episode data
            all_data.append({
                "start_idx": sid,
                "start": start,
                "traj": traj,
                "ep_len": t,
                "ep_ret": ep_ret
            })

    # dump to pickle
    with open(out_path, "wb") as f:
        pickle.dump({
            "start_points": start_points,
            "episodes": args.episodes,
            "noise": args.noise,
            "trajectories": all_data
        }, f)

    print(f"Saved {len(all_data)} trajectories → {out_path}", flush=True)
    return out_path


def visualize_trajectories(trajectory_file, args):
    """Visualize saved trajectory data over reward map with time-colored lines."""
    # load trajectories
    with open(trajectory_file, 'rb') as f:
        data = pickle.load(f)
    start_points = data['start_points']
    episodes = data['episodes']
    trajs = data['trajectories']

    model_name = os.path.splitext(os.path.basename(trajectory_file))[0]

    # create env and get reward map
    env = GridWorldEnv("two_walls")
    size = env.grid_size
    rm = env.reward_map

    fig, ax = plt.subplots(figsize=(6, 6))

    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    gc = np.array(list(env.config.goal_cells))
    for i in range(len(gc)):
        x, y = gc[i, :]
        img[y, x] = [42, 157, 143]
    ax.imshow(
        img,
        extent=(0, size, 0, size),
    )

    # highlight death cells in red
    dc = np.array(list(env.config.death_cells), dtype=float)
    if dc.size:
        ax.scatter(dc[:,0]+0.5, dc[:,1]+0.5,
                   marker='s', s=200,
                   color=np.array([229,57,70])/255.,
                   label='Death')
    # highlight goal cells in green
    gc = np.array(list(env.config.goal_cells), dtype=float)

    if gc.size:
        ax.scatter(gc[:,0]+0.5, gc[:,1]+0.5,
                   marker='s', s=200,
                   color=np.array([42,157,143])/255.,
                   label='Goal')

    ax.imshow(
        np.ones((size, size)).astype(np.float32) * 0,
        extent=(0, size, 0, size),
        cmap='Greys',
        vmin=0,
        vmax=1,
        alpha=0.4,
        zorder=2,
    )

    majors = np.arange(0, size + 1, 5)
    minors = np.arange(0, size + 1, 1)
    ax.set_xticks(majors)
    ax.set_yticks(majors)
    ax.set_xticks(minors, minor=True)
    ax.set_yticks(minors, minor=True)
    ax.grid(which="minor", color="#ddd", linestyle="-", linewidth=0.5)
    ax.grid(which="major", color="#bbb", linestyle="--", linewidth=1)

    # plot each trajectory as a time-colored line
    cmap = plt.get_cmap(args.cmap)
    for ep in trajs:
        pts = np.array(ep['traj'], dtype=float)

        xs = pts[:,0] + 0.5
        ys = pts[:,1] + 0.5
        segs = np.stack([np.column_stack([xs[:-1], ys[:-1]]),
                         np.column_stack([xs[1:], ys[1:]])], axis=1)
        norm = Normalize(0, len(xs)-1)
        lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=args.linewidth)
        lc.set_array(np.arange(len(xs)))
        ax.add_collection(lc)

    # start/end markers (start as white circle, end as cross)
    for ep in trajs:
        pts = np.array(ep['traj'], dtype=float)
        # start marker
        ax.scatter(pts[0,0]+0.5, pts[0,1]+0.5,
                   marker='o', s=150,
                   facecolors='white', edgecolors='black', zorder=3,
                   label='Start' if ep == trajs[0] else "")
        # end marker as cross
        ax.scatter(pts[-1,0]+0.5, pts[-1,1]+0.5,
                   marker='X', s=150,
                   facecolors='black', edgecolors='black', zorder=3,
                   label='End' if ep == trajs[0] else "")

    # minimal ticks
    ax.set_xticks(np.arange(0, size+1, 5))
    ax.set_yticks(np.arange(0, size+1, 5))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    # remove empty labels
    by_label = {lbl: h for h, lbl in zip(handles, labels) if lbl}
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.3,1.0))

    ax.set_title(f"{model_name}\n{episodes} runs from {len(start_points)} starts; color shows time progression", color='#333')
    plt.tight_layout()

    if args.figure_output:
        plt.savefig(args.figure_output)
        print(f"Saved figure → {args.figure_output}")
    else:
        plt.show()
        input("Press Enter to exit...")


def main():
    p = argparse.ArgumentParser(
        description="Evaluate a policy from fixed start points and visualize trajectories"
    )
    
    # Mode selection
    p.add_argument("--visualize-only", action="store_true",
                   help="Skip evaluation, only visualize existing trajectory file (requires --input)")
    p.add_argument("--no-visualize", action="store_true",
                   help="Skip visualization after evaluation")
    
    # Evaluation arguments
    p.add_argument("--actor_model",
                   help="Path to saved actor .pth file (required unless --visualize-only)")
    p.add_argument("--method", choices=["ppo", "fpo"],
                   help="Policy type: ppo (FeedForwardNN) or fpo (DiffusionPolicy) (required unless --visualize-only)")    
    p.add_argument("--episodes", "-n", type=int, default=20,
                   help="Number of episodes per start point")
    p.add_argument("--render", action="store_true",
                   help="Render env at each step during evaluation")
    p.add_argument("--sleep", type=float, default=0.001,
                   help="Delay between frames when rendering")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Std dev of Gaussian noise added to policy action (for stochasticity)")
    p.add_argument("--output", help="Where to save trajectories (.pkl). "
                   + "Defaults to <model>_fixed_traj.pkl")
    
    # Visualization arguments
    p.add_argument("--input", help="Existing trajectory pickle file (for --visualize-only mode)")
    p.add_argument("--figure-output", help="Path to save figure (PNG/PDF). If omitted, show interactively.")
    p.add_argument("--cmap", default="plasma",
                   help="Matplotlib colormap for time progression along trajectories")
    p.add_argument("--alpha", type=float, default=0.6,
                   help="Transparency for grayscale reward background")
    p.add_argument("--linewidth", type=float, default=3.0,
                   help="Line width for trajectory lines")
    
    args = p.parse_args()
    
    # Validate arguments
    if args.visualize_only:
        if not args.input:
            print("Error: --visualize-only requires --input to specify trajectory file")
            sys.exit(1)
        # Just visualize existing data
        visualize_trajectories(args.input, args)
    else:
        # Evaluation mode
        if not args.actor_model or not args.method:
            print("Error: --actor_model and --method are required for evaluation")
            sys.exit(1)
        
        # Evaluate trajectories
        trajectory_file = evaluate_trajectories(args)
        
        # Visualize unless disabled
        if not args.no_visualize:
            visualize_trajectories(trajectory_file, args)


if __name__ == "__main__":
    main()