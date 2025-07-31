#!/usr/bin/env python3
"""
BVH to LocoMuJoCo NPZ converter for SkeletonTorque environment.
Converts BVH motion capture data to trajectory format compatible with LocoMuJoCo.
"""

import numpy as np
import jax.numpy as jnp
from bvh import Bvh
from scipy.spatial.transform import Rotation as sRot
import mujoco
import argparse
from pathlib import Path

from loco_mujoco.trajectory import Trajectory, TrajectoryData, TrajectoryInfo, TrajectoryModel
from loco_mujoco.environments import LocoEnv
from loco_mujoco.smpl.retargeting import load_robot_conf_file

# Constants
DEFAULT_TARGET_FREQUENCY = 40.0
DEFAULT_HEIGHT_OFFSET = 0.975
CM_TO_M = 0.01

def convert_bvh_to_trajectory(bvh_file_path, target_frequency=DEFAULT_TARGET_FREQUENCY):
    """Convert BVH file to LocoMuJoCo trajectory format."""
    
    with open(bvh_file_path) as f:
        mocap = Bvh(f.read())
    
    # Initialize SkeletonTorque environment
    env_name = 'SkeletonTorque'
    robot_conf = load_robot_conf_file(env_name)
    env_cls = LocoEnv.registered_envs[env_name]
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
    
    # Extract joint information
    joint_names = []
    joint_types = []
    for i in range(env._model.njnt):
        jnt_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_names.append(jnt_name)
        joint_types.append(env._model.jnt_type[i])
    
    # BVH to SkeletonTorque joint mapping
    joint_mapping = {
        'Character1_RightUpLeg': [
            ('hip_flexion_r', 'Xrotation', -1.0),
            ('hip_adduction_r', 'Zrotation', -1.0),
            ('hip_rotation_r', 'Yrotation', 1.0),
        ],
        'Character1_LeftUpLeg': [
            ('hip_flexion_l', 'Xrotation', -1.0),
            ('hip_adduction_l', 'Zrotation', 1.0),
            ('hip_rotation_l', 'Yrotation', -1.0),
        ],
        'Character1_RightLeg': [('knee_angle_r', 'Xrotation', -1.0)],
        'Character1_LeftLeg': [('knee_angle_l', 'Xrotation', -1.0)],
        'Character1_RightFoot': [('ankle_angle_r', 'Xrotation', 1.0)],
        'Character1_LeftFoot': [('ankle_angle_l', 'Xrotation', 1.0)],
        
        'Character1_RightArm': [
            ('arm_flex_r', 'Xrotation', 0.8),
            ('arm_add_r', 'Yrotation', -0.7),
            ('arm_rot_r', 'Zrotation', 0.2),
        ],
        'Character1_LeftArm': [
            ('arm_flex_l', 'Xrotation', 0.8),
            ('arm_add_l', 'Yrotation', 0.7),
            ('arm_rot_l', 'Zrotation', -0.2),
        ],
        
        'Character1_RightForeArm': [
            ('elbow_flex_r', 'Xrotation', -0.8),
            ('pro_sup_r', 'Yrotation', 0.3),
        ],
        'Character1_LeftForeArm': [
            ('elbow_flex_l', 'Xrotation', -0.8),
            ('pro_sup_l', 'Yrotation', -0.3),
        ],
        'Character1_RightHand': [
            ('wrist_flex_r', 'Xrotation', 0.5),
            ('wrist_dev_r', 'Zrotation', 0.3),
        ],
        'Character1_LeftHand': [
            ('wrist_flex_l', 'Xrotation', 0.5),
            ('wrist_dev_l', 'Zrotation', -0.3),
        ],
        
        'Character1_Spine': [
            ('lumbar_extension', 'Xrotation', 0.6),
            ('lumbar_bending', 'Zrotation', 0.6),
            ('lumbar_rotation', 'Yrotation', 0.6),
        ],
    }
    
    # Get initial position for centering
    root_joint = 'Character1_Hips'
    try:
        init_x = mocap.frame_joint_channel(0, root_joint, 'Xposition') * CM_TO_M
        init_y = mocap.frame_joint_channel(0, root_joint, 'Yposition') * CM_TO_M
        init_z = mocap.frame_joint_channel(0, root_joint, 'Zposition') * CM_TO_M
    except:
        init_x = init_y = init_z = 0.0
    
    # Process frames
    n_frames = mocap.nframes
    qpos_data = np.zeros((n_frames, env._model.nq))
    
    for frame_idx in range(n_frames):
        # Root transformation
        try:
            bvh_x = mocap.frame_joint_channel(frame_idx, root_joint, 'Xposition') * CM_TO_M
            bvh_y = mocap.frame_joint_channel(frame_idx, root_joint, 'Yposition') * CM_TO_M
            bvh_z = mocap.frame_joint_channel(frame_idx, root_joint, 'Zposition') * CM_TO_M
            
            root_pos = np.array([
                bvh_z - init_z,
                bvh_x - init_x,
                bvh_y - init_y + DEFAULT_HEIGHT_OFFSET
            ])
            
            rx = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Xrotation'))
            ry = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Yrotation'))
            rz = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Zrotation'))
            
            root_rot = sRot.from_euler('zyx', [rz, ry, rx])
            xml_align = sRot.from_euler('xyz', [np.pi/2, 0, 0])
            combined_rot = root_rot * xml_align
            
            root_quat_xyzw = combined_rot.as_quat()
            root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], 
                                      root_quat_xyzw[1], root_quat_xyzw[2]])
        except:
            root_pos = np.array([0.0, 0.0, DEFAULT_HEIGHT_OFFSET])
            root_quat_wxyz = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])
        
        qpos_data[frame_idx, :3] = root_pos
        qpos_data[frame_idx, 3:7] = root_quat_wxyz
        
        # Map joint angles from BVH to SkeletonTorque
        for bvh_joint, joint_mappings in joint_mapping.items():
            for mujoco_joint, bvh_component, scaling in joint_mappings:
                if mujoco_joint in joint_names:
                    try:
                        angle_deg = mocap.frame_joint_channel(frame_idx, bvh_joint, bvh_component) or 0.0
                        angle_rad = np.deg2rad(angle_deg)
                        
                        # Normalize angle to [-π, π]
                        angle_rad = ((angle_rad + np.pi) % (2 * np.pi)) - np.pi
                        
                        # Apply scaling
                        corrected_angle = angle_rad * scaling
                        
                        # Apply joint limits
                        if 'arm_rot' in mujoco_joint or 'arm_add' in mujoco_joint:
                            corrected_angle = np.clip(corrected_angle, -np.pi/2, np.pi/2)
                        elif 'arm_flex' in mujoco_joint:
                            corrected_angle = np.clip(corrected_angle, -np.pi, np.pi)
                        
                        # Set joint position
                        joint_idx = joint_names.index(mujoco_joint)
                        qpos_addr = env._model.jnt_qposadr[joint_idx]
                        qpos_data[frame_idx, qpos_addr] = corrected_angle
                        
                    except:
                        pass
    
    # Calculate velocities using finite differences
    qvel_data = np.zeros((n_frames, env._model.nv))
    if n_frames > 2:
        dt = 1.0 / target_frequency
        qvel_data[1:-1, :] = (qpos_data[2:, :env._model.nv] - qpos_data[:-2, :env._model.nv]) / (2 * dt)
    
    # Create trajectory model
    traj_model = TrajectoryModel(
        njnt=env._model.njnt,
        jnt_type=jnp.array(joint_types),
        nbody=env._model.nbody,
        body_rootid=jnp.array(env._model.body_rootid),
        body_weldid=jnp.array(env._model.body_weldid),
        body_mocapid=jnp.array(env._model.body_mocapid),
        body_pos=jnp.array(env._model.body_pos),
        body_quat=jnp.array(env._model.body_quat),
        body_ipos=jnp.array(env._model.body_ipos),
        body_iquat=jnp.array(env._model.body_iquat),
        nsite=env._model.nsite,
        site_bodyid=jnp.array(env._model.site_bodyid),
        site_pos=jnp.array(env._model.site_pos),
        site_quat=jnp.array(env._model.site_quat)
    )
    
    traj_info = TrajectoryInfo(
        joint_names=joint_names,
        model=traj_model,
        frequency=target_frequency
    )
    
    traj_data = TrajectoryData(
        qpos=jnp.array(qpos_data),
        qvel=jnp.array(qvel_data),
        split_points=jnp.array([0, n_frames])
    )
    
    return Trajectory(info=traj_info, data=traj_data)

def main():
    parser = argparse.ArgumentParser(description='Convert BVH file to LocoMuJoCo trajectory format')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input BVH file path')
    parser.add_argument('--output', '-o', type=str, help='Output NPZ file path (default: input filename with .npz extension)')
    parser.add_argument('--frequency', '-f', type=float, default=DEFAULT_TARGET_FREQUENCY, help='Output frequency in Hz')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return None
    
    # Set output file
    if args.output:
        output_file = args.output
    else:
        output_file = input_path.with_suffix('.npz').name
    
    try:
        print(f"Converting {args.input} to {output_file}...")
        trajectory = convert_bvh_to_trajectory(args.input, args.frequency)
        
        trajectory.save(output_file)
        print(f"Successfully saved trajectory to: {output_file}")
        print(f"Frames: {trajectory.data.qpos.shape[0]}, Frequency: {args.frequency} Hz")
        print(f"Test with: python simple_motion_viewer.py --input_file {output_file} --env_name SkeletonTorque")
        
        return trajectory
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()