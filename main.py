"""
        This file is the executable for running PPO. It is based on this medium article: 
        https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gymnasium as gym
import sys
import torch
import wandb
from datetime import datetime

from utils.arguments import get_args
from models.ppo import PPO
from models.fpo import FPO
from models.network import FeedForwardNN
from models.diffusion_policy import DiffusionPolicy
from utils.eval_policy import eval_policy
from utils.gridworld import GridWorldEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def train(env, hyperparameters, actor_model, critic_model, method):
        """
                Trains the model.

                Parameters:
                        env - the environment to train on
                        hyperparameters - a dict of hyperparameters to use, defined in main
                        actor_model - the actor model to load in if we want to continue training
                        critic_model - the critic model to load in if we want to continue training

                Return:
                        None
        """     
        print(f"Training using {method.upper()}", flush=True)

        if method == "ppo":
                model = PPO(policy_class=FeedForwardNN, env=env, **hyperparameters)
        elif method == "fpo":
                model = FPO(actor_class=DiffusionPolicy, critic_class=FeedForwardNN, env=env, **hyperparameters)
        else:
                print(f"Unsupported method: {method}")
                sys.exit(1)

        model.actor.to(device)
        model.critic.to(device)
        
        # Tries to load in an existing actor/critic model to continue training on
        if actor_model != '' and critic_model != '':
                print(f"Loading in {actor_model} and {critic_model}...", flush=True)
                model.actor.load_state_dict(torch.load(actor_model, map_location=device))
                model.critic.load_state_dict(torch.load(critic_model, map_location=device))
                print(f"Successfully loaded.", flush=True)
        elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
                print(f"Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
                sys.exit(0)
        else:
                print(f"Training from scratch.", flush=True)

        # Train the PPO model with a specified total timesteps
        # NOTE: You can change the total timesteps here, I put a big number just because
        # you can kill the process whenever you feel like PPO is converging
        model.learn(total_timesteps=200_000_000)

def test(env, actor_model, method):
        """
                Tests the model.

                Parameters:
                        env - the environment to test the policy on
                        actor_model - the actor model to load in

                Return:
                        None
        """
        print(f"Testing {actor_model}", flush=True)

        # If the actor model is not specified, then exit
        if actor_model == '':
                print(f"Didn't specify model file. Exiting.", flush=True)
                sys.exit(0)

        # Extract out dimensions of observation and action spaces
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        if method == 'ppo':
                policy = FeedForwardNN(obs_dim, act_dim).to(device)
        elif method == 'fpo':
                policy = DiffusionPolicy(obs_dim + act_dim + 1, act_dim).to(device)                
        else:
                print(f"Unsupported method: {method}")
                sys.exit(1)

        # Load in the actor model saved by the PPO algorithm
        policy.load_state_dict(torch.load(actor_model, map_location=device))

        # Evaluate our policy with a separate module, eval_policy, to demonstrate
        # that once we are done training the model/policy with ppo.py, we no longer need
        # ppo.py since it only contains the training algorithm. The model/policy itself exists
        # independently as a binary file that can be loaded in with torch.
        if method == 'fpo':
                eval_policy(policy=policy.sample_action, env=env, render=True)
        else:
                eval_policy(policy=policy, env=env, render=True)

def main(args):
        """
                The main function to run.

                Parameters:
                        args - the arguments parsed from command line

                Return:
                        None
        """
        # NOTE: Here's where you can set default hyperparameters. I don't include them as part of
        # ArgumentParser because it's too annoying to type them every time at command line. Instead, you can change them here.
        # To see a list of hyperparameters, look in ppo.py at function _init_hyperparameters
        hyperparameters = {
                                'timesteps_per_batch': 2048,
                                'max_timesteps_per_episode': 200, 
                                'gamma': 0.99, 
                                'n_updates_per_iteration': 10,
                                'lr': 3e-4, 
                                'clip': 0.2,
                                'render': True,
                                'render_every_i': 1000,
                                # FPO specific parameters:
                                'grid_mode': 'two_walls',
                                'num_fpo_samples': 50,
                                'positive_advantage': False,
                          }

        print(hyperparameters)
        # Creates the environment we'll be running. If you want to replace with your own
        # custom environment, note that it must inherit Gym and have both continuous
        # observation and action spaces.
        # env = gym.make('Pendulum-v1', render_mode='human' if args.mode == 'test' else 'rgb_array')
        env = GridWorldEnv(mode=hyperparameters['grid_mode'])

        # Train or test, depending on the mode specified
        if args.mode == 'train':
                # Run name for wandb
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                lr = hyperparameters['lr']
                bs = hyperparameters['timesteps_per_batch']

                run_name = f"{args.method}_lr{lr}_bs{bs}_{timestamp}"
                
                # Add extra tag for FPO
                if args.method == "fpo":
                        n = hyperparameters['num_fpo_samples']
                        run_name += f"_N{n}"
                
                print(f"running {run_name}")

                hyperparameters["run_name"] = run_name
                
                wandb.init(
                        project="fpo-diffusion-grid",
                        name=run_name,
                        config=hyperparameters,
                        tags=[args.method, "gridworld", args.mode],
                )
                
                train(env=env, hyperparameters=hyperparameters, actor_model=args.actor_model, critic_model=args.critic_model, method=args.method)
        else:
                test(env=env, actor_model=args.actor_model, method=args.method)

if __name__ == '__main__':
        args = get_args() # Parse arguments from command line
        main(args)
