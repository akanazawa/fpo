import os
import sys
# Add the parent directory to the path so Python can find the puffer_phc module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Union

import numpy as np
import open3d as o3d
import trimesh
import viser
import viser.transforms as vtf
from mujoco import mj_id2name, mjtGeom, mjtObj  # type: ignore
import mujoco
import time
import argparse
import joblib
import torch
from puffer_phc import ASSET_DIR
from puffer_phc.poselib_skeleton import SkeletonTree, SkeletonState
from puffer_phc.torch_utils import quat_to_exp_map


class MjxWebVis:
    """A MuJoCo visualizer using Viser."""

    # global_servers: dict[int, viser.ViserServer] = {}
    global_servers = {}

    def __init__(self, mj_model, batch_size: int = 1, port: int = 8083):
        """Initialize visualizer with a MuJoCo model."""
        self.mj_model = mj_model
        self.batch_size = batch_size
        if port in MjxWebVis.global_servers:
            print(f"Found existing server on port {port}, shutting it down.")
            MjxWebVis.global_servers.pop(port).stop()

        self.server = viser.ViserServer(port=8083)
        MjxWebVis.global_servers[port] = self.server

        self.handles: dict[
            str,
            tuple[viser.BatchedMeshHandle, bool],
        ] = {}

        with self.server.gui.add_folder("Visibility"):
            cb_collision = self.server.gui.add_checkbox(
                "Collision geom", initial_value=True
            )
            cb_visual = self.server.gui.add_checkbox("Visual geom", initial_value=True)
            cb_floor = self.server.gui.add_checkbox("Floor geom", initial_value=True)

            @cb_collision.on_update
            def _(_) -> None:
                # Floor name is hack.
                for name, (handle, is_collision) in self.handles.items():
                    if is_collision and "floor" not in name:
                        handle.visible = cb_collision.value

            @cb_visual.on_update
            def _(_) -> None:
                for handle, is_collision in self.handles.values():
                    if not is_collision:
                        handle.visible = cb_visual.value

            @cb_floor.on_update
            def _(_) -> None:
                for name, handle in self.handles.items():
                    if name.startswith("floor"):
                        handle[0].visible = cb_floor.value

        # Process each geom in the model
        for i in range(mj_model.ngeom):
            # Get geom properties
            name = mj_id2name(mj_model, mjtObj.mjOBJ_GEOM, i)
            if not name:
                name = f"geom_{i}"

            pos = mj_model.geom_pos[i]
            quat = mj_model.geom_quat[i]  # (w, x, y, z)

            # Set color based on whether it's a collision geom
            is_collision = (
                mj_model.geom_contype[i] != 0 or mj_model.geom_conaffinity[i] != 0
            )
            color = (200, 100, 100) if is_collision else (100, 200, 200)

            # Handle different geom types using Viser primitives where possible
            mesh = self._create_mesh(mj_model, i)

            # Some hardcoded math for LOD choice.
            if mesh.faces.shape[0] > 1000:
                mesh = MjxWebVis._decimate_mesh(
                    mesh, target_faces=mesh.faces.shape[0] // 10
                )

            mesh_downsampled_0 = MjxWebVis._decimate_mesh(mesh, target_faces=50)
            mesh_downsampled_1 = MjxWebVis._decimate_mesh(mesh, target_faces=10)

            if self.batch_size > 10:
                mesh = mesh_downsampled_0

            # Remove the batched meshes call and just use add_mesh_simple
            handle = self.server.scene.add_mesh_simple(
                f"/geoms/{name}",
                vertices=mesh.vertices,
                faces=mesh.faces,
                flat_shading=False,
                wireframe=False,
                color=color,
                wxyz=quat,  # Set the initial orientation
                position=pos,  # Set the initial position
            )

            self.handles[name] = (handle, is_collision)

    def update(self, mj_data):
        """Update visualization with new MuJoCo data."""

        # We'll make a copy of the relevant state, then do the update itself asynchronously.
        geom_xpos = np.array(mj_data.geom_xpos)
        # Remove batch dimension for simple meshes
        assert geom_xpos.shape == (self.mj_model.ngeom, 3)
        geom_xmat = np.array(mj_data.geom_xmat.reshape((-1, 3, 3)))
        assert geom_xmat.shape == (self.mj_model.ngeom, 3, 3)

        async def update_mujoco() -> None:
            geom_xquat = vtf.SO3.from_matrix(geom_xmat).wxyz
            for i in range(self.mj_model.ngeom):
                name = mj_id2name(self.mj_model, mjtObj.mjOBJ_GEOM, i)
                if not name:
                    name = f"geom_{i}"

                if name not in self.handles:
                    continue

                # Update position and orientation for simple mesh
                handle = self.handles[name][0]
                handle.position = geom_xpos[i, :]
                handle.wxyz = geom_xquat[i, :]

        self.server._event_loop.create_task(update_mujoco())

    @staticmethod
    def _create_mesh(mj_model, idx: int) -> trimesh.Trimesh:
        """
        Create a trimesh object from a geom in the MuJoCo model.
        """
        size = mj_model.geom_size[idx]
        geom_type = mj_model.geom_type[idx]

        if geom_type == mjtGeom.mjGEOM_PLANE:
            # Create a plane mesh
            return trimesh.creation.box((20, 20, 0.01))
        elif geom_type == mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
            return trimesh.creation.icosphere(radius=radius, subdivisions=2)
        elif geom_type == mjtGeom.mjGEOM_BOX:
            dims = 2.0 * size
            return trimesh.creation.box(extents=dims)
        elif geom_type == mjtGeom.mjGEOM_MESH:
            mesh_id = mj_model.geom_dataid[idx]
            vert_start = mj_model.mesh_vertadr[mesh_id]
            vert_count = mj_model.mesh_vertnum[mesh_id]
            face_start = mj_model.mesh_faceadr[mesh_id]
            face_count = mj_model.mesh_facenum[mesh_id]

            verts = mj_model.mesh_vert[vert_start : (vert_start + vert_count), :]
            faces = mj_model.mesh_face[face_start : (face_start + face_count), :]

            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            mesh.fill_holes()
            mesh.fix_normals()
            return mesh

        elif geom_type == mjtGeom.mjGEOM_CAPSULE:
            r, half_len = size[0], size[1]
            return trimesh.creation.capsule(radius=r, height=2.0 * half_len)
        elif geom_type == mjtGeom.mjGEOM_CYLINDER:
            r, half_len = size[0], size[1]
            return trimesh.creation.cylinder(radius=r, height=2.0 * half_len)
        else:
            raise ValueError(f"Unsupported geom type {geom_type}")

    @staticmethod
    def _decimate_mesh(
        mesh: trimesh.Trimesh, target_faces: int = 500
    ) -> trimesh.Trimesh:
        """Decimate a mesh using Open3D's quartile decimation.

        Args:
            vertices: np.ndarray of shape (N, 3) containing vertex positions
            faces: np.ndarray of shape (M, 3) containing face indices
            target_ratio: Target ratio of vertices to keep (0 to 1)

        Returns:
            mesh: A decimated mesh
        """
        # Create Open3D mesh
        _mesh = o3d.geometry.TriangleMesh()
        _mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        _mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # Decimate mesh
        _mesh = _mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_faces
        )

        # Convert back to numpy arrays
        vertices_decimated = np.asarray(_mesh.vertices)
        faces_decimated = np.asarray(_mesh.triangles)

        return trimesh.Trimesh(vertices=vertices_decimated, faces=faces_decimated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--motion-file", type=str, default="sample_data/cmu_mocap_05_06.pkl", help="Path to motion file"
    )
    parser.add_argument("-i", "--motion-idx", type=int, default=0, help="Index of the motion to play")
    args = parser.parse_known_args()[0]

    motion_data = joblib.load(args.motion_file)
    keys = list(motion_data.keys())

    motion = motion_data[keys[args.motion_idx]]
    dt = 1.0 / motion["fps"]

    # Load skeleton data
    sk_tree = SkeletonTree.from_mjcf(SMPL_XML)
    sk_state = SkeletonState.from_rotation_and_root_translation(
        sk_tree, torch.from_numpy(motion["pose_quat_global"]), motion["root_trans_offset"], is_local=False
    )

    # Prepare MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(SMPL_XML)
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = dt

    # Initialize the visualizer
    vis = MjxWebVis(mj_model)
    
    # Add visual markers for keypoints
    keypoint_handles = []
    # for i in range(len(sk_tree._node_indices)):
        # marker = vis.server.scene.add_sphere(
        #     f"/keypoints/joint_{i}",
        #     radius=0.01,
        #     color=(255, 0, 0),
        #     position=(0, 0, 0),
        # )
        
        # keypoint_handles.append(marker)

    frame_idx = 0
    num_frames = motion["root_trans_offset"].shape[0]

    # Adjust height to prevent sinking into the floor
    z_offset = sk_state.global_translation[0, :, 2].min()
    global_translation = sk_state.global_translation.clone()
    global_translation[:, :, 2] -= z_offset

    print(f"Visualizing motion with {num_frames} frames at {motion['fps']} FPS")
    print("Open http://localhost:8083 in your browser to view the animation")

    while True:
        # Get the current frame data
        root_pos = global_translation[frame_idx]
        root_rot = sk_state.global_rotation[frame_idx][0]  # joint 0 is root
        local_rot = sk_state.local_rotation[frame_idx][1:]
        dof_pos = quat_to_exp_map(local_rot).flatten()

        # Set the pose in mujoco data
        print(f'ROOT POS SHAPE: {root_pos.shape}')
        mj_data.qpos[:3] = root_pos[0]
        mj_data.qpos[3:7] = root_rot[[3, 0, 1, 2]]  # xyzw -> wxyz
        mj_data.qpos[7:] = dof_pos

        print(f'MJ DATA SHAPE: {mj_data.qpos.shape}')

        # Forward kinematics (no physics)
        mujoco.mj_forward(mj_model, mj_data)

        # Update the visualization
        vis.update(mj_data)
        
        # Update keypoint markers
        async def update_keypoints():
            for i, handle in enumerate(keypoint_handles):
                if i < root_pos.shape[0]:
                    handle.position = root_pos[i].tolist()
        
        vis.server._event_loop.create_task(update_keypoints())

        # Move to next frame
        frame_idx = (frame_idx + 1) % num_frames
        time.sleep(dt)


if __name__ == "__main__":
    SMPL_XML = str(ASSET_DIR / "smpl_humanoid.xml")
    main()
