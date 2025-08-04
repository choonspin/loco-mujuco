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
DEFAULT_HEIGHT_OFFSET = 1.0  # Increased from 0.975 to ensure feet touch ground
CM_TO_M = 0.01

def analyze_bvh_rest_pose(mocap):
    """
    Analyze the BVH rest pose (first frame) to determine if arm corrections are needed.
    Returns adaptive corrections based on the actual BVH data.
    """
    corrections = {}
    
    try:
        # Analyze first frame arm positions
        frame_0 = 0
        
        # Check right arm angles in first frame
        right_arm_angles = {}
        left_arm_angles = {}
        
        arm_joints = ['Character1_RightArm', 'Character1_LeftArm']
        for joint_name in arm_joints:
            try:
                x_rot = mocap.frame_joint_channel(frame_0, joint_name, 'Xrotation') or 0.0
                y_rot = mocap.frame_joint_channel(frame_0, joint_name, 'Yrotation') or 0.0  
                z_rot = mocap.frame_joint_channel(frame_0, joint_name, 'Zrotation') or 0.0
                
                if 'Right' in joint_name:
                    right_arm_angles = {'x': x_rot, 'y': y_rot, 'z': z_rot}
                else:
                    left_arm_angles = {'x': x_rot, 'y': y_rot, 'z': z_rot}
                    
            except:
                pass
        
        print(f"🔍 BVH Rest Pose Analysis:")
        print(f"  Right arm (X,Y,Z): ({right_arm_angles.get('x', 0):.1f}°, {right_arm_angles.get('y', 0):.1f}°, {right_arm_angles.get('z', 0):.1f}°)")
        print(f"  Left arm (X,Y,Z):  ({left_arm_angles.get('x', 0):.1f}°, {left_arm_angles.get('y', 0):.1f}°, {left_arm_angles.get('z', 0):.1f}°)")
        
        # Determine if arms are pointing backward (common issue)
        # In BVH, arms pointing backward often have Y rotations around ±180°
        right_y = right_arm_angles.get('y', 0)
        left_y = left_arm_angles.get('y', 0)
        
        # Check if arms appear to be pointing backward (Y rotation near ±180°)
        backward_threshold = 120  # degrees
        
        # Always apply a small forward correction for natural arm positioning
        # Most BVH files have arms too far back for natural human posture
        base_forward_correction = np.pi/8  # 22.5° forward for natural posture
        
        if abs(right_y) > backward_threshold or abs(left_y) > backward_threshold:
            print(f"  ⚠️  Arms appear to be pointing backward - will apply enhanced corrections")
            corrections['arm_flex_correction'] = base_forward_correction + np.pi/12  # Extra 15° for backward arms
            corrections['arm_rot_correction'] = np.pi/12   # 15° rotation
        else:
            print(f"  ✅ Arms in reasonable position - applying basic forward correction")
            corrections['arm_flex_correction'] = base_forward_correction  # Basic forward positioning
            corrections['arm_rot_correction'] = 0.0
            
        # Check for extreme angles that might need limiting
        max_angle = max(abs(right_arm_angles.get('x', 0)), abs(right_arm_angles.get('y', 0)), 
                       abs(right_arm_angles.get('z', 0)), abs(left_arm_angles.get('x', 0)),
                       abs(left_arm_angles.get('y', 0)), abs(left_arm_angles.get('z', 0)))
        
        if max_angle > 200:
            print(f"  ⚠️  Extreme arm angles detected (max: {max_angle:.1f}°) - will apply angle normalization")
            corrections['normalize_extreme_angles'] = True
        else:
            corrections['normalize_extreme_angles'] = False
            
    except Exception as e:
        print(f"  ⚠️  Could not analyze rest pose: {e}")
        # Conservative fallback - no corrections
        corrections = {
            'arm_flex_correction': 0.0,
            'arm_rot_correction': 0.0, 
            'normalize_extreme_angles': False
        }
    
    return corrections


def convert_bvh_to_trajectory(bvh_file_path, target_frequency=DEFAULT_TARGET_FREQUENCY):
    """Convert BVH file to LocoMuJoCo trajectory format."""
    
    with open(bvh_file_path) as f:
        mocap = Bvh(f.read())
    
    # Analyze BVH rest pose to determine adaptive corrections
    arm_corrections = analyze_bvh_rest_pose(mocap)
    
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
        
        # BALANCED ARM MAPPING: Use reasonable ranges for natural motion
        # Primary swing from X-rotation, subtle abduction from Y-rotation
        'Character1_RightArm': [
            ('arm_flex_r', 'Xrotation', -1.0),      # 368° range → forward/back swing
            ('arm_add_r', 'Yrotation', 0.5),        # 27° range → subtle up/down (scaled down)
            ('arm_rot_r', 'Yrotation', 0.5),        # Same Y but for rotation (shared)
        ],
        'Character1_LeftArm': [
            ('arm_flex_l', 'Xrotation', -1.0),      # 375° range → forward/back swing (mirrored)
            ('arm_add_l', 'Yrotation', -0.5),       # 36° range → subtle up/down (mirrored, scaled down)
            ('arm_rot_l', 'Yrotation', -0.5),       # Same Y but for rotation (shared, mirrored)
        ],
        
        # SIMPLIFIED FOREARM MAPPING: Follow the same pattern as knee/ankle
        'Character1_RightForeArm': [
            ('elbow_flex_r', 'Xrotation', -1.0),    # Same pattern as knee_angle_r
            ('pro_sup_r', 'Yrotation', 1.0),        # Simple Y-rotation for palm
        ], 
        'Character1_LeftForeArm': [
            ('elbow_flex_l', 'Xrotation', -1.0),    # Same pattern as knee_angle_l  
            ('pro_sup_l', 'Yrotation', -1.0),       # Mirrored Y-rotation for palm
        ],
        'Character1_RightHand': [
            ('wrist_flex_r', 'Xrotation', 1.0),     # BVH hand X-rotation → wrist flexion
            ('wrist_dev_r', 'Yrotation', 1.0),      # BVH hand Y-rotation → wrist deviation
        ],
        'Character1_LeftHand': [
            ('wrist_flex_l', 'Xrotation', 1.0),     # BVH hand X-rotation → wrist flexion
            ('wrist_dev_l', 'Yrotation', -1.0),     # BVH hand Y-rotation → wrist deviation (mirrored)
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
    
    # First pass: collect all Y positions to determine proper height offset
    all_y_positions = []
    for frame_idx in range(n_frames):
        try:
            bvh_y = mocap.frame_joint_channel(frame_idx, root_joint, 'Yposition') * CM_TO_M
            all_y_positions.append(bvh_y - init_y)
        except:
            pass
    
    # Calculate adaptive height offset to ensure feet stay above ground
    if all_y_positions:
        min_relative_y = min(all_y_positions)
        # Ensure minimum height is at least 0.85m (allowing for leg length)
        adaptive_offset = max(DEFAULT_HEIGHT_OFFSET, 0.85 - min_relative_y)
        print(f"Using adaptive height offset: {adaptive_offset:.3f}m (min relative Y: {min_relative_y:.3f}m)")
    else:
        adaptive_offset = DEFAULT_HEIGHT_OFFSET
        print(f"Using default height offset: {adaptive_offset:.3f}m")
    
    for frame_idx in range(n_frames):
        # Root transformation
        try:
            bvh_x = mocap.frame_joint_channel(frame_idx, root_joint, 'Xposition') * CM_TO_M
            bvh_y = mocap.frame_joint_channel(frame_idx, root_joint, 'Yposition') * CM_TO_M
            bvh_z = mocap.frame_joint_channel(frame_idx, root_joint, 'Zposition') * CM_TO_M
            
            # Convert BVH coordinate system to MuJoCo coordinate system
            # BVH: X=left/right, Y=up/down, Z=forward/back
            # MuJoCo: X=forward/back, Y=left/right, Z=up/down
            root_pos = np.array([
                bvh_z - init_z,                      # BVH Z → MuJoCo X (forward/back)
                bvh_x - init_x,                      # BVH X → MuJoCo Y (left/right)
                bvh_y - init_y + adaptive_offset  # BVH Y → MuJoCo Z (up/down) + adaptive height fix
            ])
            
            rx = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Xrotation'))
            ry = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Yrotation'))
            rz = np.deg2rad(mocap.frame_joint_channel(frame_idx, root_joint, 'Zrotation'))
            
            # Convert BVH euler angles to rotation
            root_rot = sRot.from_euler('zyx', [rz, ry, rx])
            # Use original alignment but fix order to prevent flipping
            xml_align = sRot.from_euler('xyz', [np.pi/2, 0, 0])
            combined_rot = root_rot * xml_align  # BVH rotation first, then alignment
            
            root_quat_xyzw = combined_rot.as_quat()
            root_quat_wxyz = np.array([root_quat_xyzw[3], root_quat_xyzw[0], 
                                      root_quat_xyzw[1], root_quat_xyzw[2]])
        except:
            root_pos = np.array([0.0, 0.0, adaptive_offset])
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
                        
                        # IMPROVED: Handle extreme angle normalization properly
                        # First normalize degrees to [-180, 180] before converting to radians
                        # This fixes the 368° → 8° issue
                        angle_deg_normalized = ((angle_deg + 180) % 360) - 180
                        angle_rad = np.deg2rad(angle_deg_normalized)
                        
                        # Debug: Log extreme angle corrections (only first few to avoid spam)
                        if abs(angle_deg) > 300 and frame_idx < 3:
                            print(f"  Frame {frame_idx}: Extreme angle correction: {angle_deg:.1f}° → {angle_deg_normalized:.1f}° for {mujoco_joint}")
                        
                        # Apply scaling
                        corrected_angle = angle_rad * scaling
                        
                        # NO HARDCODED CORRECTIONS - Use exact BVH data only
                        
                        # Apply joint limits
                        if 'arm' in mujoco_joint:
                            corrected_angle = np.clip(corrected_angle, -np.pi, np.pi)
                        
                        # Set joint position
                        joint_idx = joint_names.index(mujoco_joint)
                        qpos_addr = env._model.jnt_qposadr[joint_idx]
                        qpos_data[frame_idx, qpos_addr] = corrected_angle
                        
                        # DEBUG: Log first few successful mappings
                        if frame_idx < 3 and abs(corrected_angle) > 0.01:
                            print(f"    ✓ {mujoco_joint}: BVH {angle_deg:.1f}° → NPZ {np.rad2deg(corrected_angle):.1f}° (qpos[{qpos_addr}])")
                        
                    except Exception as e:
                        if frame_idx < 3:  # Only log errors for first few frames
                            print(f"    ❌ Failed to set {mujoco_joint}: {e}")
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
    
    trajectory = Trajectory(info=traj_info, data=traj_data)
    
    # Validate ground contact
    validate_ground_contact(trajectory, env)
    
    return trajectory


def validate_ground_contact(trajectory, env):
    """Validate that feet are properly grounded."""
    print("🔍 Validating ground contact...")
    
    qpos_data = np.array(trajectory.data.qpos)
    sample_frames = np.linspace(0, len(qpos_data)-1, min(10, len(qpos_data)), dtype=int)
    
    foot_heights = []
    for frame_idx in sample_frames:
        try:
            # Set joint positions
            env._data.qpos[:] = qpos_data[frame_idx]
            mujoco.mj_forward(env._model, env._data)
            
            # Check foot site positions if they exist
            for site_name in ['right_foot_mimic', 'left_foot_mimic']:
                try:
                    site_id = mujoco.mj_name2id(env._model, mujoco.mjtObj.mjOBJ_SITE, site_name)
                    if site_id >= 0:
                        site_pos = env._data.site_xpos[site_id]
                        foot_heights.append(site_pos[2])  # Z coordinate
                except:
                    pass
        except Exception as e:
            print(f"Warning: Could not validate frame {frame_idx}: {e}")
    
    if foot_heights:
        mean_foot_height = np.mean(foot_heights)
        min_foot_height = np.min(foot_heights)
        below_ground = sum(1 for h in foot_heights if h < -0.01)
        
        print(f"  Mean foot height: {mean_foot_height:.3f}m")
        print(f"  Min foot height: {min_foot_height:.3f}m")
        print(f"  Frames with feet below ground: {below_ground}/{len(foot_heights)}")
        
        if min_foot_height < -0.02:
            print("  ⚠️  Significant foot penetration detected - consider increasing height offset")
        elif min_foot_height > 0.1:
            print("  ⚠️  Feet may be floating - consider decreasing height offset")
        else:
            print("  ✅ Foot grounding looks reasonable")
    else:
        print("  ⚠️  Could not validate foot positions - no foot sites found")

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
        print(f"✅ Successfully saved trajectory to: {output_file}")
        print(f"📊 Frames: {trajectory.data.qpos.shape[0]}, Frequency: {args.frequency} Hz")
        print(f"🔧 Applied fixes: adaptive arm analysis + adaptive pelvis height")
        print(f"🎮 Test with: python simple_motion_viewer.py --input_file {output_file} --env_name SkeletonTorque")
        
        return trajectory
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()