"""
Visualize the policy.
Assumes that the model is trained on the `two_walls` environment.
"""

import argparse
import torch
from models.diffusion_policy import DiffusionPolicy
from utils.gridworld import GridWorldEnv
from models.network import FeedForwardNN
import numpy as np
import matplotlib

import matplotlib.pyplot as plt


def visualize_ppo(model_path, seed, num_steps):
    # Set the seed.
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)

    # Set up device and environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv(mode="two_walls")
    grid_size = float(env.grid_size)

    # Load PPO actor network.
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    actor = FeedForwardNN(obs_dim, act_dim).to(device)
    actor.load_state_dict(torch.load(model_path, map_location=device))
    actor.eval()

    # Sample a grid of normalized states.
    num_points = int(grid_size)
    xs = torch.linspace(-1, 1, num_points, device=device)
    ys = torch.linspace(-1, 1, num_points, device=device)
    XX, YY = torch.meshgrid(xs, ys)
    states = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)  # (N,2)
    N = states.shape[0]

    # Predict actions.
    with torch.no_grad():
        actions = actor(states).cpu()  # (N, 2)

    # Don't visualize the cells that are not part of the environment.
    vis_mask = torch.ones_like(XX, dtype=torch.bool)
    for i in range(num_points):
        for j in range(num_points):
            if (i, j) in env.config.wall_cells:
                vis_mask[i, j] = False
            if (i, j) in env.config.death_cells:
                vis_mask[i, j] = False
            if (i, j) in env.config.goal_cells:
                vis_mask[i, j] = False

    # Visualize.
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render_into_axes(ax)  # Draw the "environment" background.

    # Add dark background overlay (to make the arrows more visible).
    ax.imshow(
        np.ones((num_points, num_points)).astype(np.float32) * 0,
        extent=(0, env.grid_size, 0, env.grid_size),
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.4,
    )

    # Re-normalize the states to the plot range.
    # The environment plot is visualized from [0, grid_size] x [0, grid_size].
    X = XX.cpu().numpy() * ((grid_size - 1) / 2) + (grid_size / 2)
    Y = YY.cpu().numpy() * ((grid_size - 1) / 2) + (grid_size / 2)

    # Initial velocity field.
    U = actions[:, 0].reshape(num_points, num_points)
    V = actions[:, 1].reshape(num_points, num_points)
    angles = (np.arctan2(V, U) + np.pi) / (2 * np.pi)
    angles = np.where(angles > 0.5, angles - 0.5, angles + 0.5)  # Shift by 180 degrees
    C = matplotlib.colormaps["twilight"](angles)
    C = C[..., :3]

    ax.quiver(
        X[vis_mask],
        Y[vis_mask],
        U[vis_mask],
        V[vis_mask],
        color=C[vis_mask],
        pivot="mid",
        headwidth=4,
        headlength=4,
        width=0.004,
    )

    fig.canvas.draw()
    plt.show()

    breakpoint()
    # ipdb.set_trace()


def visualize_fpo(model_path, seed, n_steps):
    """
    Visualize the action field of the FPO policy, sampled at a grid of states.
    Args:
        model_path: Path to the model checkpoint.
        seed: Seed for the random number generator.
        n_steps: Number of steps to run the diffusion process.
    """
    # Set the seed.
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)

    # Different noise vectors are used for each position in the grid.
    same_noise = False

    # Set up device and environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv(mode="two_walls")
    grid_size = float(env.grid_size)

    # Load policy.
    policy = DiffusionPolicy(in_dim=5, out_dim=2).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    # Sample a grid of normalized states.
    num_points = int(grid_size)
    xs = torch.linspace(-1, 1, num_points, device=device)
    ys = torch.linspace(-1, 1, num_points, device=device)
    XX, YY = torch.meshgrid(xs, ys)
    states = torch.stack([XX.reshape(-1), YY.reshape(-1)], dim=1)  # (N,2)
    N = states.shape[0]

    # Don't visualize the cells that are not part of the environment.
    vis_mask = torch.ones_like(XX, dtype=torch.bool)
    for i in range(num_points):
        for j in range(num_points):
            if (i, j) in env.config.wall_cells:
                vis_mask[i, j] = False
            if (i, j) in env.config.death_cells:
                vis_mask[i, j] = False
            if (i, j) in env.config.goal_cells:
                vis_mask[i, j] = False

    # Initialize x_t and t.
    if same_noise:
        # Fixed noise vector.
        noise_vec = torch.randn(2, device=device)
        x_t = noise_vec.unsqueeze(0).repeat(N, 1)
    else:
        # One noise per grid instead.
        noise_vec = torch.randn(N, 2, device=device)
        x_t = noise_vec.clone()
    dt = 1.0 / n_steps

    # Run the denoising process.
    trajectories = [x_t.cpu().numpy()]
    for step in range(n_steps):
        # from linear 0 to 1
        t_val = step / float(n_steps)
        t_tensor = torch.full((N, 1), t_val, device=device)
        inp = torch.cat([states, x_t, t_tensor], dim=1)
        with torch.no_grad():
            vel = policy(inp)  # (N,2)
        # Euler update: new denoised state becomes x_t
        x_t = x_t + dt * vel
        trajectories.append(x_t.cpu().numpy())

    # Visualize.
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render_into_axes(ax)  # Draw the "environment" background.

    # Add dark background overlay (to make the arrows more visible).
    ax.imshow(
        np.ones((num_points, num_points)).astype(np.float32) * 0,
        extent=(0, env.grid_size, 0, env.grid_size),
        cmap="Greys",
        vmin=0,
        vmax=1,
        alpha=0.4,
    )

    # Re-normalize the states to the plot range.
    # The environment plot is visualized from [0, grid_size] x [0, grid_size].
    X = XX.cpu().numpy() * ((grid_size - 1) / 2) + (grid_size / 2)
    Y = YY.cpu().numpy() * ((grid_size - 1) / 2) + (grid_size / 2)

    # Initial velocity field.
    U = trajectories[0][:, 0].reshape(num_points, num_points)
    V = trajectories[0][:, 1].reshape(num_points, num_points)
    angles = (np.arctan2(V, U) + np.pi) / (2 * np.pi)
    angles = np.where(angles > 0.5, angles - 0.5, angles + 0.5)  # Shift by 180 degrees
    C = matplotlib.colormaps["twilight"](angles)
    C = C[..., :3]

    Q = ax.quiver(
        X[vis_mask],
        Y[vis_mask],
        U[vis_mask],
        V[vis_mask],
        color=C[vis_mask],
        pivot="mid",
        headwidth=4,
        headlength=4,
        width=0.004,
        cmap="twilight",
    )

    for step in range(n_steps):
        # Update quiver data.
        U = trajectories[step][:, 0].reshape(num_points, num_points)
        V = trajectories[step][:, 1].reshape(num_points, num_points)

        U, V = U.clip(-2, 2), V.clip(-2, 2)
        U, V = U[vis_mask], V[vis_mask]

        angles = (np.arctan2(V, U) + np.pi) / (2 * np.pi)
        angles = np.where(
            angles > 0.5, angles - 0.5, angles + 0.5
        )  # Shift by 180 degrees
        C = matplotlib.colormaps["twilight"](angles)
        C = C[..., :3]

        Q.set_UVC(U, V)
        Q.set_color(C)  # type: ignore

        # Update title.
        ax.set_title(
            f"Denoised vector field step {step}/{n_steps}, same noise everywhere: {same_noise}"
        )

        # Draw the plot.
        fig.canvas.draw()
        plt.pause(0.5)

    plt.show()


def visualize_fpo_specific_state(model_path, seed, num_steps, n_noise_samples):
    # Set the seed.
    print(f"Using seed: {seed}")
    torch.manual_seed(seed)

    # Set up device and environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = GridWorldEnv(mode="two_walls")
    grid_size = float(env.grid_size)

    # Load policy.
    policy = DiffusionPolicy(in_dim=5, out_dim=2).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    # Visualize.
    fig, ax = plt.subplots(figsize=(6, 6))
    env.render_into_axes(ax)  # Draw the "environment" background.

    # Pick a specific state to visualize the denoising process at.
    print("Click on the grid to pick a state...")
    coords = plt.ginput(1, timeout=0)  # waits for one mouse click
    x, y = coords[0]
    ax.plot(
        x, y, marker="*", color="red", markersize=15, markeredgecolor="black", zorder=5
    )
    plt.draw()
    print(f"Picked location: x={x:.3f}, y={y:.3f}.")

    # Convert back to normalized state.
    raw_state = torch.tensor([x, y], dtype=torch.float32, device=device)
    state_norm = (raw_state - grid_size / 2) / (grid_size / 2)

    # We create two overlaid visualizations:
    # 1. Density plot of the denoising trajectories.
    # 2. A deforming grid of points that follow the denoising trajectories.

    # Density plot, saved to `trajectories`.
    # Sample K random noise points from standard Gaussian (normalized to ~[-1,1] scale).
    noise_pts = torch.randn(n_noise_samples, 2, device=device)
    t_val = 0.0  # start at noisiest time.

    trajectories = [noise_pts.cpu().numpy().copy()]
    x_t = noise_pts.clone()
    dt = 1.0 / num_steps

    # Run Euler denoising for num_steps.
    for step in range(num_steps):
        # Noise increases linearly from 0 to 1.
        t_val = step / float(num_steps)
        t_tensor = torch.full((n_noise_samples, 1), t_val, device=device)

        # Run policy / denoising step.
        inp = torch.cat(
            [state_norm.repeat(n_noise_samples, 1), x_t, t_tensor], dim=1
        )  # (n_noise_samples, 5)
        with torch.no_grad():
            vel = policy(inp)
            x_t = x_t + dt * vel  # update all points.

        trajectories.append(x_t.cpu().numpy())

    trajectories = np.stack(trajectories, axis=0)  # (num_steps+1, n_noise_samples, 2)

    # Deforming grid of points, positions saved to `grid_trajectories`.
    xx, yy = np.meshgrid(
        np.linspace(-20, 20, 50),
        np.linspace(-20, 20, 50),
    )
    num_noise_pts = xx.size
    noise_pts = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).to(device)
    grid_trajectories = [noise_pts.cpu().numpy().copy()]
    x_t = noise_pts.clone()
    for step in range(num_steps):
        # from linear 0 to 1
        t_val = step / float(num_steps)
        t_tensor = torch.full((num_noise_pts, 1), t_val, device=device)
        inp = torch.cat(
            [state_norm.repeat(num_noise_pts, 1), x_t, t_tensor], dim=1
        )  # (K,5)
        with torch.no_grad():
            vel = policy(inp.float())
            x_t = x_t + dt * vel  # update all points
            grid_trajectories.append(x_t.cpu().numpy())

    def plot_grid_from_grid_trajectory(ax, grid_positions):
        """
        Plot a grid based on the grid corner point positions.
        """
        # Ensure xx and yy are captured from the outer scope where meshgrid is defined.
        # xx, yy are from the test_specific_state scope.
        rows, cols = xx.shape

        # Reshape grid_positions from (rows*cols, 2) to (rows, cols, 2)
        points_grid = grid_positions.reshape(rows, cols, 2)

        # Plot horizontal lines
        # Each row in points_grid[r, :, :] is a set of points (x,y) for that row.
        for r in range(rows):
            ax.plot(
                points_grid[r, :, 0],
                points_grid[r, :, 1],
                color="black",
                linewidth=0.3,
                alpha=0.6,
            )

        # Plot vertical lines
        # Each col in points_grid[:, c, :] is a set of points (x,y) for that column.
        for c in range(cols):
            ax.plot(
                points_grid[:, c, 0],
                points_grid[:, c, 1],
                color="black",
                linewidth=0.3,
                alpha=0.6,
            )

    num_steps = trajectories.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    contour = []
    for step in range(num_steps):
        pts = trajectories[step]
        ax.clear()

        if len(contour) > 0:
            for c in contour:
                c.remove()
            contour = []

        plot_grid_from_grid_trajectory(ax, grid_trajectories[step])
        contour.append(ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#FF5E00", alpha=0.01))

        # Draw a plus-sign at the middle of the plot.
        ax.plot(
            0,
            0,
            marker="+",
            color="red",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=2,
            zorder=5,
        )

        ax.set_title(f"Denoising trajectories at step {step} at grid ({x}, {y})")
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect("equal", "box")
        plt.pause(0.5)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", dest="method", type=str, default="fpo")  # or 'ppo'.
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=20)

    parser.add_argument(
        "--specific_state", action="store_true"
    )  # Only used if ``method`` is 'fpo'.
    parser.add_argument(
        "--n_noise_samples", type=int, default=50000
    )  # Only used if ``method`` is 'fpo' and ``specific_state`` is True.

    args = parser.parse_args()
    
    # Set intelligent default model path based on method
    if args.model_path is None:
        if args.method == "ppo":
            args.model_path = "ppo_actor.pth"
        elif args.method == "fpo":
            args.model_path = "fpo_actor.pth"
        else:
            raise ValueError(f"Unknown method: {args.method}")

    if args.method == "ppo":
        visualize_ppo(args.model_path, args.seed, args.num_steps)
    elif args.method == "fpo":
        if args.specific_state:
            visualize_fpo_specific_state(
                args.model_path, args.seed, args.num_steps, args.n_noise_samples
            )
        else:
            visualize_fpo(args.model_path, args.seed, args.num_steps)
    else:
        raise ValueError(f"Invalid method: {args.method}")
