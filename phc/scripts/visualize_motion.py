import sys
from pathlib import Path

# Add the project root to sys.path
# This assumes the script is in 'scripts/' and 'puffer_phc/' is in the parent directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
from types import SimpleNamespace
import random # Import random module

import mujoco
import numpy as np
import torch
import joblib # For loading the motion key directly if needed for verification

from puffer_phc import ASSET_DIR, ROOT_DIR
from puffer_phc.mjx_viser import MjxWebVis
from puffer_phc.motion_lib import FixHeightMode, MotionLibSMPL, MotionlibMode
from puffer_phc.poselib_skeleton import SkeletonTree
from smpl_sim.smpllib.smpl_joint_names import SMPL_MUJOCO_NAMES # For limb weights

def get_limb_weights(skeleton_tree, expected_node_names, device):
    """
    Computes dummy limb weights for MotionLibSMPL.
    Simplified from HumanoidPHC, using zeros for masses.
    """
    _body_names = expected_node_names # Use the passed names
    joint_groups = [
        ["L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
        ["R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
        ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
        ["L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
        ["R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
    ]
    
    # Create a mapping from name to index based on expected_node_names
    name_to_idx = {name: i for i, name in enumerate(expected_node_names)}

    limb_weight_group_indices = []
    for joint_group in joint_groups:
        group_indices = []
        for joint_name in joint_group:
            if joint_name in name_to_idx:
                group_indices.append(name_to_idx[joint_name])
            else:
                # This might happen if skeleton_tree.node_names differs significantly
                # or a name in joint_groups is not in the skeleton.
                # For visualization, this might be okay if not used by AMP.
                print(f"Warning: Joint name '{joint_name}' not found in skeleton_tree.node_names for limb_weights.")
        limb_weight_group_indices.append(group_indices)


    limb_lengths = torch.norm(skeleton_tree.local_translation.to(device), dim=-1)
    
    summed_limb_lengths = []
    for group in limb_weight_group_indices:
        if group: # If group is not empty
            summed_limb_lengths.append(limb_lengths[group].sum())
        else:
            summed_limb_lengths.append(torch.tensor(0.0, device=device)) # Add zero if group was empty

    # Using dummy masses (zeros) as they are not critical for qpos visualization
    # Number of groups for masses should match number of limb_length groups
    dummy_masses_sum = [torch.tensor(0.0, device=device)] * len(summed_limb_lengths) 
    
    humanoid_limb_weight = torch.tensor(summed_limb_lengths + dummy_masses_sum, device=device)
    return humanoid_limb_weight.unsqueeze(0) # Add batch dimension


def main(motion_file_name):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = "cpu" # Keep CPU for now if needed
    print(f"Using device: {DEVICE}")

    SMPL_XML_PATH = str(ASSET_DIR / "smpl_humanoid.xml")
    # MOTION_FILE_PATH = str(ROOT_DIR / "data" / "amass" / motion_file_name)
    MOTION_FILE_PATH = str(ROOT_DIR / "amass" / motion_file_name)
    # 1. Initialize MotionLibSMPL
    motion_lib_cfg = SimpleNamespace(
        motion_file=MOTION_FILE_PATH,
        device=DEVICE,
        fix_height=FixHeightMode.full_fix,
        min_length=-1,
        max_length=-1,
        im_eval=True,
        is_deterministic=True,
        smpl_type='smpl',
        step_dt=1/30,
        num_thread=1
    )
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    print(f"Loaded motion library from: {MOTION_FILE_PATH}")

    # Ensure SMPL parsers are on the correct device
    if motion_lib.mesh_parsers is not None:
        for parser_key in motion_lib.mesh_parsers:
            parser_instance = motion_lib.mesh_parsers[parser_key]
            parser_instance.to(DEVICE)

    # Pre-convert numpy arrays in loaded motion data to tensors on the target DEVICE
    if motion_lib.mode == MotionlibMode.file and hasattr(motion_lib, '_motion_data_list') and motion_lib._motion_data_list is not None:
        for i in range(len(motion_lib._motion_data_list)):
            motion_dict = motion_lib._motion_data_list[i]
            if isinstance(motion_dict, dict):
                keys_to_convert = ['pose_aa', 'root_trans_offset', 'pose_quat_global']
                for key_name in keys_to_convert:
                    if key_name in motion_dict and isinstance(motion_dict[key_name], np.ndarray):
                        motion_dict[key_name] = torch.tensor(
                            motion_dict[key_name], dtype=torch.float32, device=DEVICE
                        )
    elif motion_lib.mode == MotionlibMode.directory:
        print("Warning: MotionLib is in directory mode. Pre-conversion logic might need adjustment.")

    # Get the list of all available motion keys ONCE
    try:
        motion_data_keys_list = motion_lib._motion_data_keys.tolist()
        if not motion_data_keys_list:
            print("Error: No motion keys found in the motion library.")
            return
        print(f"Found {len(motion_data_keys_list)} motions.")
    except Exception as e:
        print(f"Error accessing motion keys: {e}")
        return

    # 2. Prepare common data for motion_lib.load_motions()
    sk_tree = SkeletonTree.from_mjcf(SMPL_XML_PATH)
    gender_beta = torch.zeros(17, device=DEVICE) # Neutral shape
    limb_weights = get_limb_weights(sk_tree, SMPL_MUJOCO_NAMES, DEVICE)

    # 3. Setup MuJoCo and Viser
    mj_model = mujoco.MjModel.from_xml_path(SMPL_XML_PATH)
    mj_data = mujoco.MjData(mj_model)
    viser_viewer = MjxWebVis(mj_model, batch_size=1)
    print("Viser viewer initialized. Open the provided URL in your browser.")

    # --- Main Loop ---
    try:
        while True: # Infinite loop
            # Select a random motion key
            motion_key_to_visualize = random.choice(motion_data_keys_list)
            print(f"\n--- Loading new motion: {motion_key_to_visualize} ---")

            # Find the index of the selected motion key
            try:
                motion_idx = motion_data_keys_list.index(motion_key_to_visualize)
            except ValueError:
                print(f"Warning: Randomly selected key '{motion_key_to_visualize}' not found in index list (should not happen). Skipping.")
                continue

            sample_idxes = torch.tensor([motion_idx], device=DEVICE, dtype=torch.long)

            # Load the selected motion (populates _motion_num_frames, _motion_dt etc.)
            motion_lib.load_motions(
                skeleton_trees=[sk_tree],
                gender_betas=gender_beta.unsqueeze(0),
                limb_weights=limb_weights,
                sample_idxes=sample_idxes,
                random_sample=False,
                same_motion_for_all=True
            )

            # Get properties of the *currently loaded* motion (at index 0)
            if not hasattr(motion_lib, '_motion_num_frames') or len(motion_lib._motion_num_frames) == 0:
                 print("Warning: Motion loaded but no frame data found (_motion_num_frames). Skipping.")
                 time.sleep(0.1) # Avoid busy-waiting if loading fails repeatedly
                 continue
            if not hasattr(motion_lib, '_motion_dt') or len(motion_lib._motion_dt) == 0:
                 print("Warning: Motion loaded but no frame data found (_motion_dt). Skipping.")
                 time.sleep(0.1)
                 continue

            num_frames = motion_lib._motion_num_frames[0].item()
            actual_motion_dt = motion_lib._motion_dt[0].item()
            motion_idx_in_batch = torch.tensor([0], device=DEVICE) # It's always index 0 after loading one motion

            print(f"Motion '{motion_key_to_visualize}' contains {num_frames} frames, dt = {actual_motion_dt:.4f}s. Starting playback...")

            # Animation Loop for the current motion
            for f_idx in range(num_frames):
                motion_time = torch.tensor([f_idx * actual_motion_dt], device=DEVICE)
                state = motion_lib.get_motion_state(motion_idx_in_batch, motion_time)

                # Extract and convert data for MuJoCo
                root_pos = state['root_pos'].squeeze().cpu().numpy()
                # Assuming state['root_rot'] is XYZW (scalar-last: x,y,z,w) based on render_env_viser.py
                root_rot_input_xyzw = state['root_rot'].squeeze().cpu().numpy() 
                
                # Convert XYZW to WXYZ for MuJoCo's qpos
                # (x,y,z,w) -> (w,x,y,z)
                root_rot_output_wxyz = root_rot_input_xyzw[[3, 0, 1, 2]]
                
                dof_pos = state['dof_pos'].squeeze().cpu().numpy()

                # Update MuJoCo data
                mj_data.qpos[:3] = root_pos
                mj_data.qpos[3:7] = root_rot_output_wxyz # Use the converted WXYZ quaternion
                mj_data.qpos[7:] = dof_pos
                
                mujoco.mj_forward(mj_model, mj_data)
                viser_viewer.update(mj_data, robot_idx=0)

                time.sleep(actual_motion_dt)


            print(f"Finished playback for motion: {motion_key_to_visualize}")

    except KeyboardInterrupt:
        print("\nVisualization stopped by user.")
    finally:
        print("Closing Viser viewer (Ctrl+C in terminal if it doesn't close automatically).")
        if hasattr(viser_viewer, 'viser_server') and viser_viewer.viser_server.is_running():
             print("Viser server might still be running. Press Ctrl+C in the terminal again if needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and visualize reference motions randomly and continuously.")
    parser.add_argument(
        "--motion_file",
        type=str,
        default="amass_train_11313_upright.pkl", # Example file
        help="Name of the motion .pkl file in data/amass/"
    )
    args = parser.parse_args()

    main(args.motion_file)