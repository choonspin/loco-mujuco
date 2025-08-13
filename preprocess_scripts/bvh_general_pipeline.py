#!/usr/bin/env python3
"""
General BVH to LocoMuJoCo preprocessing pipeline for imitation learning.
Works with any BVH file by:
1. Extracting all movements from original BVH data
2. Aligning every BVH to +Y forward during preprocessing  
3. Normalizing rotations to SkeletonTorque's coordinate frame
4. Storing rotation offsets for consistent simulator input
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
DEFAULT_HEIGHT_OFFSET = 0.975  # Increased to prevent ground contact
CM_TO_M = 0.01

def normalize_angle(angle_deg):
    """Normalize angle to [-180, 180] range."""
    return ((angle_deg + 180) % 360) - 180

def compute_forward_alignment(mocap, root_joint='Character1_Hips'):
    """
    Compute the rotation needed to align BVH +Y forward.
    Returns the yaw rotation to apply to align the character's initial facing direction with +Y.
    """
    try:
        # Get initial root orientation
        initial_ry = np.deg2rad(mocap.frame_joint_channel(0, root_joint, 'Yrotation') or 0.0)
        
        # Compute rotation to align with +Y forward (0 degrees in our coordinate system)
        alignment_rotation = -initial_ry  # Negate to align with +Y
        
        print(f"Initial root Y-rotation: {np.rad2deg(initial_ry):.2f}°")
        print(f"Alignment rotation needed: {np.rad2deg(alignment_rotation):.2f}°")
        
        return alignment_rotation
        
    except Exception as e:
        print(f"Warning: Could not compute forward alignment: {e}")
        return 0.0

def create_general_joint_mapping():
    """
    Create a general joint mapping that extracts all available motion from BVH.
    Uses reasonable scaling factors that work across different BVH files.
    """
    return {
        # Legs - increased hip flexion for proper sitting (thighs horizontal)
        'Character1_RightUpLeg': [
            ('hip_flexion_r', 'Xrotation', -1.0),    # Hip flexion for horizontal thighs when sitting
            ('hip_adduction_r', 'Zrotation', -0.3),  # Hip abduction/adduction
            ('hip_rotation_r', 'Yrotation', 0.3),    # Hip internal/external rotation
        ],
        'Character1_LeftUpLeg': [
            ('hip_flexion_l', 'Xrotation', -1.0),    # Hip flexion for horizontal thighs when sitting
            ('hip_adduction_l', 'Zrotation', 0.3),   # Hip abduction/adduction (mirrored)
            ('hip_rotation_l', 'Yrotation', -0.3),   # Hip internal/external rotation (mirrored)
        ],
        'Character1_RightLeg': [('knee_angle_r', 'Xrotation', -1.0)],
        'Character1_LeftLeg': [('knee_angle_l', 'Xrotation', -1.0)],
        'Character1_RightFoot': [('ankle_angle_r', 'Xrotation', 0.8)],
        'Character1_LeftFoot': [('ankle_angle_l', 'Xrotation', 0.8)],
        
        # Arms - extract all 3 rotation axes for natural motion
        'Character1_RightArm': [
            ('arm_flex_r', 'Xrotation', -0.8),       # Forward/backward arm swing
            ('arm_add_r', 'Yrotation', -0.6),        # Arm abduction/adduction  
            ('arm_rot_r', 'Zrotation', 0.4),         # Arm internal/external rotation
        ],
        'Character1_LeftArm': [
            ('arm_flex_l', 'Xrotation', -0.8),       # Forward/backward arm swing
            ('arm_add_l', 'Yrotation', 0.6),         # Arm abduction/adduction (mirrored)
            ('arm_rot_l', 'Zrotation', -0.4),        # Arm internal/external rotation (mirrored)
        ],
        
        # Forearms - elbow flexion and pronation/supination
        'Character1_RightForeArm': [
            ('elbow_flex_r', 'Xrotation', -1.5),      # Elbow flexion (positive for forward bending)
            ('pro_sup_r', 'Zrotation', 0.8),         # Forearm pronation/supination
        ], 
        'Character1_LeftForeArm': [
            ('elbow_flex_l', 'Xrotation', -1.5),      # Elbow flexion (positive for forward bending)
            ('pro_sup_l', 'Yrotation', -0.8),        # Forearm pronation/supination
        ],
        
        # Hands - wrist motion
        'Character1_RightHand': [
            ('wrist_flex_r', 'Xrotation', 0.6),      # Wrist flexion/extension
            ('wrist_dev_r', 'Yrotation', 0.6),       # Wrist radial/ulnar deviation
        ],
        'Character1_LeftHand': [
            ('wrist_flex_l', 'Xrotation', 0.6),      # Wrist flexion/extension
            ('wrist_dev_l', 'Yrotation', -0.6),      # Wrist radial/ulnar deviation (mirrored)
        ],
        
        # Spine - conservative scaling for multiple segments
        'Character1_Spine': [
            ('lumbar_extension', 'Xrotation', -0.4),  # Spine flexion/extension
            ('lumbar_bending', 'Zrotation', 0.3),     # Spine lateral bending
            ('lumbar_rotation', 'Yrotation', 0.3),    # Spine rotation
        ],
        'Character1_Spine1': [
            ('lumbar_extension', 'Xrotation', -0.2),  # Additional spine contribution
            ('lumbar_bending', 'Zrotation', 0.15),
            ('lumbar_rotation', 'Yrotation', 0.15),
        ],
        'Character1_Spine2': [
            ('lumbar_extension', 'Xrotation', -0.1),  # Additional spine contribution
            ('lumbar_bending', 'Zrotation', 0.1),
            ('lumbar_rotation', 'Yrotation', 0.1),
        ],
    }

def convert_bvh_general(bvh_file_path, target_frequency=DEFAULT_TARGET_FREQUENCY):
    """General BVH to LocoMuJoCo conversion pipeline."""
    
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
    
    print(f"Processing BVH with {mocap.nframes} frames...")
    print(f"SkeletonTorque joints: {len(joint_names)} joints")
    
    # Step 1: Compute forward alignment
    root_joint = 'Character1_Hips'
    alignment_rotation = compute_forward_alignment(mocap, root_joint)
    
    # Step 2: Get general joint mapping
    joint_mapping = create_general_joint_mapping()
    
    # Step 3: Get initial position for reference
    try:
        init_x = mocap.frame_joint_channel(0, root_joint, 'Xposition') * CM_TO_M
        init_y = mocap.frame_joint_channel(0, root_joint, 'Yposition') * CM_TO_M
        init_z = mocap.frame_joint_channel(0, root_joint, 'Zposition') * CM_TO_M
    except:
        init_x = init_y = init_z = 0.0
    
    # Step 4: Calculate adaptive height offset to ensure feet stay above ground
    all_y_positions = []
    for frame_idx in range(mocap.nframes):
        try:
            bvh_y = mocap.frame_joint_channel(frame_idx, root_joint, 'Yposition') * CM_TO_M
            all_y_positions.append(bvh_y - init_y)
        except:
            pass
    
    if all_y_positions:
        min_relative_y = min(all_y_positions)
        adaptive_offset = max(DEFAULT_HEIGHT_OFFSET, 0.85 - min_relative_y)
        print(f"📏 Adaptive height offset: {adaptive_offset:.3f}m (min Y: {min_relative_y:.3f}m)")
    else:
        adaptive_offset = DEFAULT_HEIGHT_OFFSET
        print(f"📏 Using default height offset: {adaptive_offset:.3f}m")
    
    # Step 5: Process all frames
    n_frames = mocap.nframes
    qpos_data = np.zeros((n_frames, env._model.nq))
    
    for frame_idx in range(n_frames):
        # Root position with Y-forward alignment
        try:
            bvh_x = mocap.frame_joint_channel(frame_idx, root_joint, 'Xposition') * CM_TO_M
            bvh_y = mocap.frame_joint_channel(frame_idx, root_joint, 'Yposition') * CM_TO_M
            bvh_z = mocap.frame_joint_channel(frame_idx, root_joint, 'Zposition') * CM_TO_M
            
            # Apply coordinate transformation for +Y forward
            # Original BVH: X=lateral, Y=height, Z=forward
            # SkeletonTorque: X=forward, Y=lateral, Z=height
            root_pos = np.array([
                (bvh_z - init_z),                     # X (forward) = BVH Z 
                (bvh_x - init_x),                     # Y (lateral) = BVH X
                (bvh_y - init_y) + adaptive_offset    # Z (height) = BVH Y + adaptive offset
            ])
        except:
            root_pos = np.array([0.0, 0.0, adaptive_offset])
        
        qpos_data[frame_idx, :3] = root_pos
        
        # Root orientation with Y-forward alignment (yaw-only for stability)
        try:
            ry = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Yrotation') or 0.0)
            
            # Apply alignment rotation to ensure +Y forward
            aligned_ry = ry + alignment_rotation
            
            # Use only yaw rotation for stability (like bvh_direct_map.py approach)
            root_rot = sRot.from_euler('y', [aligned_ry])
            xml_align = sRot.from_euler('x', [np.pi/2])  # Align Z-up coordinate system
            combined_rot = xml_align * root_rot
            
            root_quat_xyzw = combined_rot.as_quat()
            root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0],
                                      root_quat_xyzw[1], root_quat_xyzw[2]])
        except:
            root_quat_wxyz = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0])  # 90-degree X rotation for upright pose
        
        qpos_data[frame_idx, 3:7] = root_quat_wxyz
        
        # Map all available BVH joints to SkeletonTorque joints
        for bvh_joint, mappings in joint_mapping.items():
            for mujoco_joint, bvh_channel, scaling in mappings:
                if mujoco_joint in joint_names:
                    try:
                        # Extract raw angle from BVH
                        angle_deg = mocap.frame_joint_channel(frame_idx, bvh_joint, bvh_channel) or 0.0
                        
                        # Normalize and convert to radians
                        normalized_angle = normalize_angle(angle_deg)
                        angle_rad = np.deg2rad(normalized_angle) * scaling
                        
                        # Apply joint limits based on joint type
                        if 'arm' in mujoco_joint or 'elbow' in mujoco_joint or 'wrist' in mujoco_joint:
                            # Upper body joints - allow larger range
                            angle_rad = np.clip(angle_rad, -np.pi, np.pi)
                        elif 'hip' in mujoco_joint or 'knee' in mujoco_joint or 'ankle' in mujoco_joint:
                            # Lower body joints - more conservative
                            angle_rad = np.clip(angle_rad, -np.pi/2, np.pi/2)
                        elif 'lumbar' in mujoco_joint:
                            # Spine joints - moderate range
                            angle_rad = np.clip(angle_rad, -np.pi/3, np.pi/3)
                        
                        # Set joint angle
                        joint_idx = joint_names.index(mujoco_joint)
                        qpos_addr = env._model.jnt_qposadr[joint_idx]
                        qpos_data[frame_idx, qpos_addr] = angle_rad
                        
                    except Exception as e:
                        # Skip missing joints gracefully
                        pass
    
    # Calculate velocities
    qvel_data = np.zeros((n_frames, env._model.nv))
    if n_frames > 2:
        dt = 1.0 / target_frequency
        qvel_data[1:-1, :] = (qpos_data[2:, :env._model.nv] - qpos_data[:-2, :env._model.nv]) / (2 * dt)
    
    # Create trajectory
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
    
    trajectory = Trajectory(info=traj_info, data=traj_data)
    
    # Store metadata for consistency
    metadata = {
        'alignment_rotation': alignment_rotation,
        'original_file': str(bvh_file_path),
        'coordinate_system': '+Y_forward_Z_up'
    }
    
    return trajectory, metadata

def main():
    parser = argparse.ArgumentParser(description='General BVH to LocoMuJoCo preprocessing pipeline')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input BVH file path')
    parser.add_argument('--output', '-o', type=str, help='Output NPZ file path')
    parser.add_argument('--frequency', '-f', type=float, default=DEFAULT_TARGET_FREQUENCY, help='Output frequency in Hz')
    
    args = parser.parse_args()
    
    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        return None
    
    # Set output
    if args.output:
        output_file = args.output
    else:
        output_file = input_path.with_suffix('.npz').name
    
    try:
        print(f"🔄 Processing BVH file: {args.input}")
        print("📐 Pipeline: General BVH → +Y Forward Aligned → LocoMuJoCo NPZ")
        
        trajectory, metadata = convert_bvh_general(args.input, args.frequency)
        
        trajectory.save(output_file)
        
        print(f"✅ Successfully saved trajectory to: {output_file}")
        print(f"📊 Frames: {trajectory.data.qpos.shape[0]}, Frequency: {args.frequency} Hz")
        print(f"🧭 Forward alignment: {np.rad2deg(metadata['alignment_rotation']):.2f}° rotation applied")
        print(f"🎯 Coordinate system: {metadata['coordinate_system']}")
        print(f"🎮 Test with: python simple_motion_viewer.py --input_file {output_file} --env_name SkeletonTorque")
        
        return trajectory, metadata
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()