#!/usr/bin/env python3
"""
Motion Capture Data Preprocessing Script for LocoMuJoCo

This script converts raw motion capture data (BVH, FBX, C3D, etc.) to LocoMuJoCo's 
NPZ format using the existing infrastructure, specifically the ExtendTrajData class 
for automatic kinematics computation.

Supported formats:
- BVH (Biovision Hierarchy)
- FBX (Autodesk FBX)
- C3D (3D biomechanics)
- JSON/YAML (custom format)

Usage:
    python preprocess_mocap.py --input_file motion.bvh --output_file trajectory.npz --env_name UnitreeH1
"""

import argparse
import json
import numpy as np
import yaml
from pathlib import Path
from dataclasses import replace
import warnings

import jax.numpy as jnp
import mujoco

# Motion capture parsing libraries
try:
    import bvh
    from bvh import Bvh
    BVH_AVAILABLE = True
except ImportError:
    BVH_AVAILABLE = False
    warnings.warn("BVH library not available. Install with: pip install bvh")

try:
    from fbx import FbxManager, FbxIOSettings, FbxImporter, FbxScene
    FBX_AVAILABLE = True
except ImportError:
    try:
        import fbx
        FBX_AVAILABLE = True
    except ImportError:
        FBX_AVAILABLE = False
        warnings.warn("FBX SDK not available. Install FBX SDK for Python")

try:
    import c3d
    C3D_AVAILABLE = True
except ImportError:
    C3D_AVAILABLE = False
    warnings.warn("C3D library not available. Install with: pip install c3d")

# Add AMC/ASF support
try:
    import amc_parser  # Custom parser needed
    AMC_AVAILABLE = True
except ImportError:
    AMC_AVAILABLE = False
    warnings.warn("AMC/ASF parser not available. Custom implementation needed")

from loco_mujoco.environments import LocoEnv
from loco_mujoco.datasets.data_generation import ExtendTrajData, calculate_qvel_with_finite_difference
from loco_mujoco.smpl.retargeting import load_robot_conf_file
from loco_mujoco.trajectory import (
    Trajectory,
    TrajectoryInfo,
    TrajectoryModel,
    TrajectoryData,
    interpolate_trajectories
)
from loco_mujoco.utils import setup_logger


def create_bvh_to_skeleton_mapping():
    """
    Create mapping from BVH Character1 joints to SkeletonTorque joints.
    BVH joints have 3 rotation channels (YXZ), SkeletonTorque has individual joints.
    
    Returns:
        dict: BVH joint index -> list of (SkeletonTorque joint, rotation channel index)
    """
    return {
        # BVH joint index -> [(skeleton_joint, rotation_channel), ...]
        # Character1_Hips (0) -> handled as free joint
        
        # Lower body - FIXED: Match BVH joint names with correct skeleton sides
        1: [('hip_rotation_l', 0), ('hip_flexion_l', 1), ('hip_adduction_l', 2)],   # Character1_LeftUpLeg (YXZ) → LEFT skeleton
        2: [('knee_angle_l', 1)],                                                    # Character1_LeftLeg (X rotation for flexion) → LEFT skeleton
        3: [('ankle_angle_l', 1)],                                                   # Character1_LeftFoot (X rotation for flexion) → LEFT skeleton
        4: [('hip_rotation_r', 0), ('hip_flexion_r', 1), ('hip_adduction_r', 2)],   # Character1_RightUpLeg (YXZ) → RIGHT skeleton
        5: [('knee_angle_r', 1)],                                                    # Character1_RightLeg (X rotation for flexion) → RIGHT skeleton  
        6: [('ankle_angle_r', 1)],                                                   # Character1_RightFoot (X rotation for flexion) → RIGHT skeleton
        
        # Torso - corrected for anatomical movement patterns
        7: [('lumbar_rotation', 0), ('lumbar_extension', 1), ('lumbar_bending', 2)], # Character1_Spine (YXZ)
        # Skip Spine1 (8) and Spine2 (9) - redundant
        
        # Left arm - corrected for anatomical movement patterns
        10: [('arm_flex_l', 1)],                                                     # Character1_LeftShoulder (X rotation for flexion)
        11: [('arm_rot_l', 0), ('arm_add_l', 2)],                                    # Character1_LeftArm (YZ)
        12: [('elbow_flex_l', 1)],                                                   # Character1_LeftForeArm (X rotation for flexion)
        13: [('pro_sup_l', 0), ('wrist_flex_l', 1), ('wrist_dev_l', 2)],            # Character1_LeftHand (YXZ)
        
        # Right arm - corrected for anatomical movement patterns
        25: [('arm_flex_r', 1)],                                                     # Character1_RightShoulder (X rotation for flexion)
        26: [('arm_rot_r', 0), ('arm_add_r', 2)],                                    # Character1_RightArm (YZ)
        27: [('elbow_flex_r', 1)],                                                   # Character1_RightForeArm (X rotation for flexion)
        28: [('pro_sup_r', 0), ('wrist_flex_r', 1), ('wrist_dev_r', 2)],           # Character1_RightHand (YXZ)
        
        # Finger joints (14-24, 29-38) ignored - SkeletonTorque has no fingers
        # Character1_Neck (24) ignored - SkeletonTorque has no neck
    }


def parse_bvh_file(file_path):
    """Parse BVH (Biovision Hierarchy) motion capture file."""
    if not BVH_AVAILABLE:
        raise ImportError("BVH library not available. Install with: pip install bvh")
    
    with open(file_path) as f:
        mocap = Bvh(f.read())
    
    # Extract frame data
    frame_count = mocap.nframes
    frame_time = mocap.frame_time
    frequency = 1.0 / frame_time
    
    # Get joint names
    joint_names = mocap.get_joints_names()
    
    # Get root joint (usually first joint - typically "Hips" or similar)
    root_joint = joint_names[0] if joint_names else None
    if not root_joint:
        raise ValueError("No joints found in BVH file")
    
    # Extract motion data
    qpos_data = []
    
    for frame_idx in range(frame_count):
        frame_values = []
        
        # Extract root joint data (position + rotation)
        root_channels = mocap.frame_joint_channels(frame_idx, root_joint, ['Xposition', 'Yposition', 'Zposition', 'Yrotation', 'Xrotation', 'Zrotation'])
        root_pos = root_channels[:3]
        
        # Convert from centimeters to meters (CeTI data appears to be in cm)
        root_pos = [pos / 100.0 for pos in root_pos]
        
        root_rot = root_channels[3:6]  # Euler angles in degrees
        
        # Convert Euler angles to quaternions (scalar-first)
        from scipy.spatial.transform import Rotation as R
        # BVH typically uses YXZ order for rotations (already in degrees, no conversion needed for R.from_euler)
        rot = R.from_euler('YXZ', root_rot, degrees=True)
        quat = rot.as_quat()  # [x,y,z,w] format
        quat_scalar_first = [quat[3], quat[0], quat[1], quat[2]]  # [w,x,y,z]
        
        # Add root position and quaternion
        frame_values.extend(root_pos)
        frame_values.extend(quat_scalar_first)
        
        # Extract other joint rotations
        for joint_name in joint_names[1:]:  # Skip root joint
            try:
                joint_channels = mocap.frame_joint_channels(frame_idx, joint_name, ['Yrotation', 'Xrotation', 'Zrotation'])
                
                # Normalize angles to [-180, 180] range to fix angle wrapping
                # BVH often stores angles near 360° instead of negative angles
                normalized_channels = []
                for angle in joint_channels:
                    # More robust normalization using modulo
                    normalized_angle = ((angle + 180) % 360) - 180
                    normalized_channels.append(normalized_angle)
                
                # Convert degrees to radians
                joint_channels = [np.deg2rad(angle) for angle in normalized_channels]
                frame_values.extend(joint_channels)
            except:
                # Some joints might not have all channels, pad with zeros
                frame_values.extend([0.0, 0.0, 0.0])
        
        qpos_data.append(frame_values)
    
    # Check if we should use specialized SkeletonTorque mapping
    skeleton_joint_mapping = create_bvh_to_skeleton_mapping()
    skeleton_joint_names = [
        'hip_flexion_r', 'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'ankle_angle_r',
        'hip_flexion_l', 'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'ankle_angle_l',
        'lumbar_extension', 'lumbar_bending', 'lumbar_rotation',
        'arm_flex_r', 'arm_add_r', 'arm_rot_r', 'elbow_flex_r', 'pro_sup_r', 'wrist_flex_r', 'wrist_dev_r',
        'arm_flex_l', 'arm_add_l', 'arm_rot_l', 'elbow_flex_l', 'pro_sup_l', 'wrist_flex_l', 'wrist_dev_l'
    ]
    
    # Rearrange qpos data to match SkeletonTorque joint order
    skeleton_qpos_data = []
    for frame_idx in range(frame_count):
        skeleton_frame = []
        
        # Add root (free joint) data - position and quaternion (7 DOF)
        skeleton_frame.extend(qpos_data[frame_idx][:7])  # root pos + quat
        
        # Map BVH joint rotations to SkeletonTorque joints in correct order
        skeleton_values = [0.0] * 27  # Initialize all 27 skeleton joints to zero
        
        for bvh_joint_idx, skeleton_mappings in skeleton_joint_mapping.items():
            if bvh_joint_idx < len(joint_names):  # Make sure joint exists in BVH
                # Get the 3 rotation values for this BVH joint (YXZ order)
                bvh_joint_rotations = qpos_data[frame_idx][7 + (bvh_joint_idx-1)*3 : 7 + bvh_joint_idx*3]
                
                for skeleton_joint, rotation_channel in skeleton_mappings:
                    if skeleton_joint in skeleton_joint_names:
                        skeleton_idx = skeleton_joint_names.index(skeleton_joint)
                        if rotation_channel < len(bvh_joint_rotations):
                            skeleton_values[skeleton_idx] = bvh_joint_rotations[rotation_channel]
        
        # Combine root + skeleton joint values
        skeleton_frame.extend(skeleton_values)
        skeleton_qpos_data.append(skeleton_frame)
    
    qpos_array = np.array(skeleton_qpos_data)
    
    # DEBUG: Check joint angles after skeleton mapping
    print(f"DEBUG: After skeleton mapping, qpos_array shape: {qpos_array.shape}")
    if qpos_array.shape[0] > 0:
        joint_angles_deg = np.rad2deg(qpos_array[:, 7:])  # Skip root (7 DOF)
        print(f"DEBUG: Joint angle ranges after mapping:")
        for i in range(min(10, joint_angles_deg.shape[1])):
            joint_range = joint_angles_deg[:, i].max() - joint_angles_deg[:, i].min()
            print(f"  Joint {i}: range = {joint_range:.1f}°")
    
    # Resample to 40Hz if needed (LocoMuJoCo standard)
    target_frequency = 40.0
    if frequency != target_frequency:
        print(f"[BVH Parser] INFO: Resampling from {frequency}Hz to {target_frequency}Hz")
        
        # Calculate resampling indices
        original_times = np.arange(frame_count) / frequency
        target_duration = original_times[-1]
        target_frames = int(target_duration * target_frequency) + 1
        target_times = np.linspace(0, target_duration, target_frames)
        
        # Interpolate each DOF
        resampled_qpos = np.zeros((target_frames, qpos_array.shape[1]))
        for dof in range(qpos_array.shape[1]):
            resampled_qpos[:, dof] = np.interp(target_times, original_times, qpos_array[:, dof])
        
        qpos_array = resampled_qpos
        frequency = target_frequency
        frame_count = target_frames
        print(f"[BVH Parser] INFO: Resampled to {frame_count} frames at {frequency}Hz")
    
    return {
        'qpos': qpos_array,
        'joint_names': skeleton_joint_names,
        'frequency': frequency,
        'n_timesteps': frame_count,
        'format': 'BVH_Skeleton'
    }


def parse_fbx_file(file_path):
    """Parse FBX motion capture file."""
    if not FBX_AVAILABLE:
        raise ImportError("FBX SDK not available. Install FBX SDK for Python")
    
    # Initialize the FBX SDK
    manager = FbxManager.Create()
    ios = FbxIOSettings.Create(manager, "IOSN")
    manager.SetIOSettings(ios)
    
    importer = FbxImporter.Create(manager, "")
    scene = FbxScene.Create(manager, "myScene")
    
    # Import the FBX file
    if not importer.Initialize(str(file_path), -1, manager.GetIOSettings()):
        raise ValueError(f"Cannot initialize FBX importer: {importer.GetStatus().GetErrorString()}")
    
    importer.Import(scene)
    importer.Destroy()
    
    # Extract animation data
    # This is a simplified implementation - real FBX parsing is more complex
    # You may need to customize this based on your specific FBX structure
    
    root_node = scene.GetRootNode()
    joint_names = []
    animation_data = []
    
    # Extract joint hierarchy and animation curves
    # (Implementation depends on specific FBX structure)
    
    manager.Destroy()
    
    # Placeholder return - implement based on your FBX structure
    return {
        'qpos': np.array([]),
        'joint_names': joint_names,
        'frequency': 30.0,  # Default FBX framerate
        'n_timesteps': 0,
        'format': 'FBX'
    }


def parse_c3d_file(file_path):
    """Parse C3D (3D biomechanics) motion capture file."""
    if not C3D_AVAILABLE:
        raise ImportError("C3D library not available. Install with: pip install c3d")
    
    with open(file_path, 'rb') as f:
        reader = c3d.Reader(f)
        
        # Extract basic info
        frequency = reader.frame_rate
        n_frames = reader.last_frame - reader.first_frame + 1
        
        # Extract marker data
        frames = []
        for frame_no, points, analog in reader.read_frames():
            # points contains 3D marker positions
            # analog contains force plate/EMG data (if available)
            frames.append(points[:, :3])  # X, Y, Z coordinates
        
        # Note: C3D files contain marker positions, not joint angles
        # You'll need inverse kinematics to convert to joint space
        # This is a placeholder implementation
        
        return {
            'qpos': np.array(frames),  # This would need IK processing
            'joint_names': [f"marker_{i}" for i in range(points.shape[0])],
            'frequency': frequency,
            'n_timesteps': n_frames,
            'format': 'C3D',
            'marker_data': True  # Flag indicating this needs IK processing
        }


def parse_mocap_data(input_file):
    """
    Parse raw motion capture data from various formats.
    
    Supported formats:
    - BVH: Biovision Hierarchy files with joint rotations
    - FBX: Autodesk FBX files (requires FBX SDK)
    - C3D: 3D biomechanics files with marker positions
    - JSON/YAML: Custom format with joint positions
    
    Args:
        input_file (str): Path to input mocap data file
        
    Returns:
        dict: Parsed mocap data with qpos array and metadata
    """
    logger = setup_logger("preprocess", identifier="[Mocap Preprocessing]")
    
    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    suffix = file_path.suffix.lower()
    
    if suffix == '.bvh':
        logger.info("Parsing BVH file...")
        return parse_bvh_file(file_path)
        
    elif suffix == '.fbx':
        logger.info("Parsing FBX file...")
        return parse_fbx_file(file_path)
        
    elif suffix == '.c3d':
        logger.info("Parsing C3D file...")
        data = parse_c3d_file(file_path)
        if data.get('marker_data'):
            logger.warning("C3D contains marker data - inverse kinematics processing needed")
        return data
        
    elif suffix == '.json':
        logger.info("Parsing JSON file...")
        with open(file_path, 'r') as f:
            data = json.load(f)
            
    elif suffix in ['.yaml', '.yml']:
        logger.info("Parsing YAML file...")
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
            
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    # Handle JSON/YAML format
    if 'trajectory' in data:
        # Extract trajectory data
        trajectory_data = data['trajectory']
        joint_names = data['joint_names']
        frequency = data.get('frequency', 40.0)
        
        # Convert to qpos array
        n_timesteps = len(trajectory_data)
        n_joints = len(joint_names)
        
        # Add 7 DOF for free joint (3 pos + 4 quat) if not present
        if 'root_position' in trajectory_data[0] and 'root_orientation' in trajectory_data[0]:
            qpos_dim = n_joints + 7  # joint positions + free joint
            qpos = np.zeros((n_timesteps, qpos_dim))
            
            for i, frame in enumerate(trajectory_data):
                # Free joint: position (3) + quaternion (4, scalar-first)
                qpos[i, :3] = frame['root_position']
                qpos[i, 3:7] = frame['root_orientation']  # Assume scalar-first quaternion
                qpos[i, 7:] = frame['joint_positions']
        else:
            # Assume data includes free joint already
            qpos_dim = n_joints
            qpos = np.array([frame['joint_positions'] for frame in trajectory_data])
        
        return {
            'qpos': qpos,
            'joint_names': joint_names,
            'frequency': frequency,
            'n_timesteps': n_timesteps,
            'format': 'JSON/YAML'
        }
    else:
        raise ValueError("JSON/YAML file must contain 'trajectory' key")
    
    logger.info(f"Parsed mocap data: {data['n_timesteps']} timesteps, "
                f"{data['qpos'].shape[1]} DOF, {data['frequency']} Hz, format: {data.get('format', 'Unknown')}")


def create_joint_mapping(mocap_joint_names, robot_joint_names):
    """
    Create mapping between mocap joint names and robot joint names.
    
    Args:
        mocap_joint_names (list): Joint names from mocap data
        robot_joint_names (list): Joint names from robot model
        
    Returns:
        dict: Mapping from mocap indices to robot indices
    """
    logger = setup_logger("preprocess", identifier="[Joint Mapping]")
    
    # BVH to SkeletonTorque joint mapping
    # BVH joints have multiple rotation channels (X,Y,Z) that map to separate SkeletonTorque joints
    # NOTE: Coordinate system conversion applied - BVH axes remapped to MuJoCo axes:
    # BVH X (left/right) → MuJoCo Y, BVH Y (up/down) → MuJoCo Z, BVH Z (forward/back) → MuJoCo X
    bvh_joint_mappings = {
        # BVH hip joints: Character1_LeftUpLeg/RightUpLeg (3 channels each) → 3 separate SkeletonTorque joints
        'Character1_LeftUpLeg': {
            'X': 'hip_flexion_l',     # BVH X rotation (left/right axis) = flexion (forward/back movement)
            'Y': 'hip_rotation_l',    # BVH Y rotation (up/down axis) = rotation (twist)
            'Z': 'hip_adduction_l',   # BVH Z rotation (forward/back axis) = adduction (side-to-side movement)
        },
        'Character1_RightUpLeg': {
            'X': 'hip_flexion_r',
            'Y': 'hip_rotation_r',
            'Z': 'hip_adduction_r',
        },
        
        # BVH knee joints: Character1_LeftLeg/RightLeg → single SkeletonTorque knee joint
        'Character1_LeftLeg': {
            'X': 'knee_angle_l',      # BVH X rotation (left/right) for knee flexion
            'Y': None,                # Ignore Y rotation
            'Z': None,                # Ignore Z rotation
        },
        'Character1_RightLeg': {
            'X': 'knee_angle_r',
            'Y': None,
            'Z': None,
        },
        
        # BVH ankle joints: Character1_LeftFoot/RightFoot → single SkeletonTorque ankle joint
        'Character1_LeftFoot': {
            'X': 'ankle_angle_l',     # BVH X rotation (left/right) for ankle flexion
            'Y': None,
            'Z': None,
        },
        'Character1_RightFoot': {
            'X': 'ankle_angle_r',
            'Y': None,
            'Z': None,
        },
        
        # BVH spine joints: Character1_Spine/Spine1/Spine2 → lumbar joints
        'Character1_Spine': {
            'X': 'lumbar_bending',    # BVH X rotation (left/right) = bending (side-to-side)
            'Y': 'lumbar_rotation',   # BVH Y rotation (up/down) = rotation (twist)
            'Z': 'lumbar_extension',  # BVH Z rotation (forward/back) = extension
        },
        'Character1_Spine1': {
            'X': None,  # Use only first spine joint for now
            'Y': None,
            'Z': None,
        },
        'Character1_Spine2': {
            'X': None,
            'Y': None, 
            'Z': None,
        },
        
        # BVH shoulder joints: Character1_LeftShoulder/RightShoulder → arm joints
        'Character1_LeftShoulder': {
            'X': 'arm_flex_l',        # BVH X rotation (left/right) = flexion
            'Y': 'arm_rot_l',         # BVH Y rotation (up/down) = rotation
            'Z': 'arm_add_l',         # BVH Z rotation (forward/back) = adduction
        },
        'Character1_RightShoulder': {
            'X': 'arm_flex_r',
            'Y': 'arm_rot_r',
            'Z': 'arm_add_r',
        },
        
        # BVH elbow joints: Character1_LeftArm/LeftForeArm → elbow joints
        'Character1_LeftArm': {
            'X': 'elbow_flex_l',      # BVH X rotation (left/right) for elbow flexion
            'Y': None,
            'Z': None,
        },
        'Character1_LeftForeArm': {
            'X': 'pro_sup_l',         # BVH X rotation (left/right) for pronation/supination
            'Y': None,
            'Z': None,
        },
        'Character1_RightArm': {
            'X': 'elbow_flex_r',
            'Y': None,
            'Z': None,
        },
        'Character1_RightForeArm': {
            'X': 'pro_sup_r',
            'Y': None,
            'Z': None,
        },
        
        # BVH hand joints: Character1_LeftHand/RightHand → wrist joints
        'Character1_LeftHand': {
            'X': 'wrist_flex_l',      # BVH X rotation (left/right) = flexion
            'Y': 'wrist_dev_l',       # BVH Y rotation (up/down) = deviation
            'Z': None,
        },
        'Character1_RightHand': {
            'X': 'wrist_flex_r',
            'Y': 'wrist_dev_r', 
            'Z': None,
        },
    }
    
    # Legacy mappings for non-BVH formats
    common_mappings = {
        'hip_rotation_l': ['left_hip_yaw', 'l_hip_yaw', 'LeftHipYaw', 'left_leg_hip_y'],
        'hip_adduction_l': ['left_hip_roll', 'l_hip_roll', 'LeftHipRoll', 'left_leg_hip_r'], 
        'hip_flexion_l': ['left_hip_pitch', 'l_hip_pitch', 'LeftHipPitch', 'left_leg_hip_p'],
        'hip_rotation_r': ['right_hip_yaw', 'r_hip_yaw', 'RightHipYaw', 'right_leg_hip_y'],
        'hip_adduction_r': ['right_hip_roll', 'r_hip_roll', 'RightHipRoll', 'right_leg_hip_r'],
        'hip_flexion_r': ['right_hip_pitch', 'r_hip_pitch', 'RightHipPitch', 'right_leg_hip_p'],
        'knee_angle_l': ['left_knee', 'l_knee', 'LeftKnee', 'left_leg_knee'],
        'knee_angle_r': ['right_knee', 'r_knee', 'RightKnee', 'right_leg_knee'],
        'ankle_angle_l': ['left_ankle_pitch', 'l_ankle_pitch', 'LeftAnklePitch', 'left_leg_ankle_p'],
        'ankle_angle_r': ['right_ankle_pitch', 'r_ankle_pitch', 'RightAnklePitch', 'right_leg_ankle_p'],
    }
    
    # Detect if this is BVH channel-based data
    is_bvh_channels = any(name.startswith(('Character1_', 'Xposition', 'Yposition', 'Zposition')) for name in mocap_joint_names)
    
    if is_bvh_channels:
        logger.info("Detected BVH channel-based data, using BVH joint mapping")
        return create_bvh_channel_mapping(mocap_joint_names, robot_joint_names, bvh_joint_mappings, logger)
    else:
        logger.info("Using legacy joint name mapping")
        return create_legacy_joint_mapping(mocap_joint_names, robot_joint_names, common_mappings, logger)


def create_bvh_channel_mapping(mocap_joint_names, robot_joint_names, bvh_joint_mappings, logger):
    """
    Create mapping for BVH channel-based data where each joint has multiple rotation channels
    """
    mapping = {}
    unmapped_mocap = []
    unmapped_robot = set(robot_joint_names)
    
    # Parse BVH channel structure: root (6 channels) + joints (3 channels each)
    # Channels: Xposition, Yposition, Zposition, Zrotation, Xrotation, Yrotation, 
    #           Joint1_Zrot, Joint1_Xrot, Joint1_Yrot, Joint2_Zrot, ...
    
    channel_idx = 0
    
    # Skip root channels (first 6: position + rotation)
    if channel_idx < len(mocap_joint_names) and mocap_joint_names[channel_idx] in ['Xposition', 'Yposition', 'Zposition']:
        channel_idx += 6  # Skip root position (3) + root rotation (3)
        logger.info("Skipped root channels (6 total)")
    
    # Process BVH joint rotation channels
    # Each BVH joint has 3 rotation channels: Z, X, Y (in that order)
    bvh_joint_names = [name for name in bvh_joint_mappings.keys()]
    
    for joint_name in bvh_joint_names:
        if joint_name not in bvh_joint_mappings:
            continue
            
        joint_mapping = bvh_joint_mappings[joint_name]
        
        # Map the 3 rotation channels for this joint: Z, X, Y
        for rotation_axis in ['Z', 'X', 'Y']:
            if channel_idx >= len(mocap_joint_names):
                logger.warning(f"Ran out of mocap channels at joint {joint_name}, axis {rotation_axis}")
                break
                
            target_joint = joint_mapping.get(rotation_axis)
            if target_joint and target_joint in robot_joint_names:
                robot_idx = robot_joint_names.index(target_joint)
                mapping[channel_idx] = robot_idx
                unmapped_robot.discard(target_joint)
                logger.debug(f"Mapped channel {channel_idx} ({joint_name}_{rotation_axis}) → {target_joint}")
            else:
                # Channel doesn't map to anything or target joint doesn't exist
                if target_joint:
                    logger.debug(f"Target joint {target_joint} not found in robot joints")
                unmapped_mocap.append(f"{joint_name}_{rotation_axis}")
                
            channel_idx += 1
    
    logger.info(f"Mapped {len(mapping)}/{len(mocap_joint_names)} channels using BVH mapping")
    if unmapped_mocap:
        logger.warning(f"Unmapped mocap channels: {unmapped_mocap[:20]}{'...' if len(unmapped_mocap) > 20 else ''}")
    if unmapped_robot:
        logger.warning(f"Unmapped robot joints: {list(unmapped_robot)[:15]}{'...' if len(unmapped_robot) > 15 else ''}")
    
    return mapping


def create_legacy_joint_mapping(mocap_joint_names, robot_joint_names, common_mappings, logger):
    """
    Create mapping for legacy joint name-based data
    """
    # Create reverse mapping (all possible names -> standard name)
    name_to_standard = {}
    for standard_name, variants in common_mappings.items():
        for variant in variants:
            name_to_standard[variant.lower()] = standard_name
    
    # Map mocap joints to robot joints
    mapping = {}
    unmapped_mocap = []
    unmapped_robot = set(robot_joint_names)
    
    for mocap_idx, mocap_name in enumerate(mocap_joint_names):
        # Try direct match first
        if mocap_name in robot_joint_names:
            robot_idx = robot_joint_names.index(mocap_name)
            mapping[mocap_idx] = robot_idx
            unmapped_robot.discard(mocap_name)
            continue
            
        # Try standard name mapping
        standard_name = name_to_standard.get(mocap_name.lower())
        if standard_name and standard_name in robot_joint_names:
            robot_idx = robot_joint_names.index(standard_name)
            mapping[mocap_idx] = robot_idx
            unmapped_robot.discard(standard_name)
            continue
            
        # Try fuzzy matching
        best_match = None
        best_score = 0
        for robot_idx, robot_name in enumerate(robot_joint_names):
            # Simple similarity check
            common_words = set(mocap_name.lower().split('_')) & set(robot_name.lower().split('_'))
            score = len(common_words)
            if score > best_score and score > 0:
                best_score = score
                best_match = robot_idx
        
        if best_match is not None:
            mapping[mocap_idx] = best_match
            unmapped_robot.discard(robot_joint_names[best_match])
        else:
            unmapped_mocap.append(mocap_name)
    
    logger.info(f"Mapped {len(mapping)}/{len(mocap_joint_names)} joints using legacy mapping")
    if unmapped_mocap:
        logger.warning(f"Unmapped mocap joints: {unmapped_mocap}")
    if unmapped_robot:
        logger.warning(f"Unmapped robot joints: {list(unmapped_robot)}")
    
    return mapping


def create_minimal_trajectory(mocap_data, env_name):
    """
    Create a minimal trajectory structure that can be extended by ExtendTrajData.
    
    Args:
        mocap_data (dict): Parsed mocap data
        env_name (str): Environment name (e.g., "UnitreeH1")
        
    Returns:
        Trajectory: Minimal trajectory with qpos and basic info
    """
    logger = setup_logger("preprocess", identifier="[Mocap Preprocessing]")
    
    # Load environment to get model information
    env_cls = LocoEnv.registered_envs[env_name]
    robot_conf = load_robot_conf_file(env_name)
    env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
    
    # Get robot joint names from the model (excluding free joint)
    robot_joint_names = []
    for i in range(env._model.njnt):
        if env._model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:  # Skip free joint
            joint_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name:
                robot_joint_names.append(joint_name)
    
    logger.info(f"Robot has {len(robot_joint_names)} joints: {robot_joint_names}")
    
    # Handle joint mapping if needed
    mocap_qpos = mocap_data['qpos']
    
    if mocap_data.get('format') in ['BVH', 'BVH_Skeleton', 'FBX']:
        # For BVH/FBX, data has 6-DOF root (position + Euler) that needs conversion to 7-DOF (position + quaternion)
        if mocap_qpos.shape[1] >= 6:
            # Extract root position (3 DOF) and Euler rotation (3 DOF)
            bvh_root_position = mocap_qpos[:, :3]  # X, Y, Z position in BVH coordinates
            bvh_root_euler = mocap_qpos[:, 3:6]    # Euler rotations (ZXY order for BVH)
            joint_data = mocap_qpos[:, 6:]         # Remaining joint rotations
            
            logger.info(f"Original BVH root position range: X[{bvh_root_position[:, 0].min():.3f}, {bvh_root_position[:, 0].max():.3f}], Y[{bvh_root_position[:, 1].min():.3f}, {bvh_root_position[:, 1].max():.3f}], Z[{bvh_root_position[:, 2].min():.3f}, {bvh_root_position[:, 2].max():.3f}]")
            logger.info(f"First frame BVH root position: {bvh_root_position[0]}")
            
            # Convert BVH coordinate system to MuJoCo coordinate system
            # BVH: X=left/right, Y=up/down, Z=forward/back (Y-up, right-handed)
            # MuJoCo: X=forward/back, Y=left/right, Z=up/down (Z-up, right-handed)
            # Conversion: BVH_Z → MuJoCo_X, BVH_X → MuJoCo_Y, BVH_Y → MuJoCo_Z
            
            # NOTE: BVH data was already converted from cm to meters in parse_bvh_file()
            # Only need to remap coordinate axes here
            root_position = np.zeros_like(bvh_root_position)
            root_position[:, 0] = bvh_root_position[:, 2]  # BVH Z → MuJoCo X (forward/back)
            root_position[:, 1] = bvh_root_position[:, 0]  # BVH X → MuJoCo Y (left/right) 
            root_position[:, 2] = bvh_root_position[:, 1]  # BVH Y → MuJoCo Z (up/down)
            
            logger.info(f"Converted MuJoCo root position range: X[{root_position[:, 0].min():.3f}, {root_position[:, 0].max():.3f}], Y[{root_position[:, 1].min():.3f}, {root_position[:, 1].max():.3f}], Z[{root_position[:, 2].min():.3f}, {root_position[:, 2].max():.3f}]")
            logger.info(f"First frame MuJoCo root position: {root_position[0]}")
            
            # Convert Euler angles to quaternions with coordinate system conversion
            from scipy.spatial.transform import Rotation as R
            
            # IMPORTANT: For coordinate system conversion, we need to be very careful
            # BVH: Y-up, X-left/right, Z-forward/back
            # MuJoCo: Z-up, Y-left/right, X-forward/back
            
            # The key insight: when coordinate system changes, rotations also need to change
            # A rotation around BVH-Y (up) becomes a rotation around MuJoCo-Z (up)
            # A rotation around BVH-X (left/right) becomes a rotation around MuJoCo-Y (left/right)
            # A rotation around BVH-Z (forward/back) becomes a rotation around MuJoCo-X (forward/back)
            
            # But BVH stores rotations as YXZ order, so we need to remap carefully
            root_euler_remapped = np.zeros_like(bvh_root_euler)
            root_euler_remapped[:, 0] = bvh_root_euler[:, 2]  # BVH Z-rot → MuJoCo X-rot (forward/back axis)
            root_euler_remapped[:, 1] = bvh_root_euler[:, 0]  # BVH Y-rot → MuJoCo Z-rot (up/down axis) 
            root_euler_remapped[:, 2] = bvh_root_euler[:, 1]  # BVH X-rot → MuJoCo Y-rot (left/right axis)
            
            # Convert to quaternions using XZY order to match the remapping
            root_rotations = R.from_euler('XZY', root_euler_remapped, degrees=True)
            
            # CRITICAL FIX: SkeletonTorque model needs 90° X-rotation to be upright
            # The reference upright skeleton has quaternion [0.707, 0.707, 0, 0] = 90° X-rotation
            # We need to apply this corrective rotation to our BVH data
            upright_correction = R.from_euler('X', 90, degrees=True)
            root_rotations_corrected = upright_correction * root_rotations
            
            root_quaternions = root_rotations_corrected.as_quat()  # [x, y, z, w] format
            
            # Convert to MuJoCo's [w, x, y, z] format
            root_quaternions_mujoco = np.zeros_like(root_quaternions)
            root_quaternions_mujoco[:, 0] = root_quaternions[:, 3]  # w
            root_quaternions_mujoco[:, 1] = root_quaternions[:, 0]  # x  
            root_quaternions_mujoco[:, 2] = root_quaternions[:, 1]  # y
            root_quaternions_mujoco[:, 3] = root_quaternions[:, 2]  # z
            
            # Combine position + quaternion for 7-DOF root
            root_data = np.concatenate([root_position, root_quaternions_mujoco], axis=1)
            
            logger.info(f"Converted BVH root: {root_data.shape[0]} timesteps, root_data shape: {root_data.shape}")
            logger.info(f"First frame root_data (pos + quat): {root_data[0]}")
            logger.info(f"Root Z position (height): first={root_data[0, 2]:.6f}, last={root_data[-1, 2]:.6f}")
            
            # Joint data is already in radians from BVH parser - no conversion needed
            joint_data_rad = joint_data
            logger.info(f"Joint angles already in radians from BVH parser")
            
            # Joint mapping is handled by BVH parser - joint_data_rad is already in SkeletonTorque format
            final_qpos = np.concatenate([root_data, joint_data_rad], axis=1)
        else:
            raise ValueError("BVH/FBX data must include root transform (6 DOF minimum)")
            
    elif mocap_data.get('format') == 'C3D':
        logger.error("C3D format requires inverse kinematics - not yet implemented")
        raise NotImplementedError("C3D inverse kinematics not implemented")
        
    else:
        # JSON/YAML format - assume data is already in correct format
        final_qpos = mocap_qpos
    
    # Validate qpos dimensions
    expected_qpos_dim = 7 + len(robot_joint_names)  # free joint + robot joints
    if final_qpos.shape[1] != expected_qpos_dim:
        logger.warning(f"qpos dimension mismatch: got {final_qpos.shape[1]}, expected {expected_qpos_dim}")
        # Pad or truncate as needed
        if final_qpos.shape[1] < expected_qpos_dim:
            padding = np.zeros((final_qpos.shape[0], expected_qpos_dim - final_qpos.shape[1]))
            final_qpos = np.concatenate([final_qpos, padding], axis=1)
        else:
            final_qpos = final_qpos[:, :expected_qpos_dim]
    
    # Create basic trajectory model exactly matching the environment model
    # Debug: print model dimensions
    logger.info(f"Creating trajectory model - env.njnt: {env._model.njnt}, expected joints: {len(robot_joint_names) + 1}")
    
    traj_model = TrajectoryModel(
        njnt=env._model.njnt,  # This should be 28 for SkeletonTorque
        jnt_type=jnp.array(env._model.jnt_type),
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
    
    # Create trajectory info with root joint included
    # The environment expects joint_names to include the root joint
    full_joint_names = ['root'] + robot_joint_names
    logger.info(f"Creating trajectory info with {len(full_joint_names)} joint names: {full_joint_names[:5]}...")
    
    traj_info = TrajectoryInfo(
        joint_names=full_joint_names,
        frequency=mocap_data['frequency'],
        model=traj_model
    )
    
    # Calculate qvel using finite difference (optional - ExtendTrajData can handle this)
    if final_qpos.shape[0] > 2:  # Need at least 3 timesteps for central difference
        qpos_fd, qvel_fd = calculate_qvel_with_finite_difference(final_qpos, mocap_data['frequency'])
        logger.info(f"Calculated qvel using finite difference: {qvel_fd.shape}")
    else:
        qpos_fd = final_qpos
        qvel_fd = np.zeros((final_qpos.shape[0], final_qpos.shape[1] - 1))  # qvel has one less dimension
        logger.warning("Insufficient data for finite difference, using zero velocities")
    
    # Create trajectory data (minimal - only qpos/qvel)
    traj_data = TrajectoryData(
        qpos=jnp.array(qpos_fd),
        qvel=jnp.array(qvel_fd),
        split_points=jnp.array([0, len(qpos_fd)])
    )
    
    trajectory = Trajectory(info=traj_info, data=traj_data)
    
    logger.info(f"Created minimal trajectory: {len(qpos_fd)} timesteps, {final_qpos.shape[1]} DOF")
    return trajectory, env


def extend_trajectory_with_kinematics(trajectory, env):
    """
    Use LocoMuJoCo's ExtendTrajData class to compute full kinematics.
    
    Args:
        trajectory (Trajectory): Minimal trajectory with qpos/qvel
        env: LocoMuJoCo environment
        
    Returns:
        Trajectory: Extended trajectory with full kinematics data
    """
    logger = setup_logger("preprocess", identifier="[Mocap Preprocessing]")
    
    # Interpolate to match environment frequency if needed
    traj_data, traj_info = interpolate_trajectories(
        trajectory.data, 
        trajectory.info, 
        1.0 / env.dt
    )
    trajectory = Trajectory(info=traj_info, data=traj_data)
    
    # Load trajectory into environment
    env.load_trajectory(trajectory, warn=False)
    traj_data, traj_info = env.th.traj.data, env.th.traj.info
    
    logger.info("Extending trajectory with full kinematics using ExtendTrajData...")
    
    # Debug information
    logger.info(f"Environment model njnt: {env._model.njnt}")
    logger.info(f"Trajectory model njnt: {traj_info.model.njnt}")
    logger.info(f"Trajectory qpos shape: {traj_data.qpos.shape}")
    logger.info(f"Environment expected qpos dim: {env._model.nq}")
    logger.info(f"Environment expected qvel dim: {env._model.nv}")
    
    # Print joint names comparison
    env_joint_names = []
    for i in range(env._model.njnt):
        joint_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if joint_name:
            env_joint_names.append(joint_name)
    
    print(f"Trajectory joint names ({len(traj_info.joint_names)}): {traj_info.joint_names}")
    print(f"Environment joint names ({len(env_joint_names)}): {env_joint_names}")
    
    # Use ExtendTrajData to compute full kinematics
    callback = ExtendTrajData(env, model=env._model, n_samples=traj_data.n_samples)
    env.play_trajectory(
        n_episodes=env.th.n_trajectories,
        render=False,
        callback_class=callback
    )
    
    # Get extended trajectory data
    extended_traj_data, extended_traj_info = callback.extend_trajectory_data(traj_data, traj_info)
    extended_trajectory = replace(trajectory, data=extended_traj_data, info=extended_traj_info)
    
    logger.info("Trajectory extension completed!")
    logger.info(f"Extended data contains: qpos{extended_traj_data.qpos.shape}, "
                f"xpos{extended_traj_data.xpos.shape}, sites{extended_traj_data.site_xpos.shape}")
    
    return extended_trajectory


def validate_trajectory(trajectory, reference_file=None):
    """
    Validate the processed trajectory against expected format.
    
    Args:
        trajectory (Trajectory): Processed trajectory
        reference_file (str, optional): Path to reference NPZ file for comparison
    """
    logger = setup_logger("preprocess", identifier="[Mocap Preprocessing]")
    
    data = trajectory.data
    info = trajectory.info
    
    # Basic validation
    assert data.qpos is not None, "qpos data missing"
    assert data.qvel is not None, "qvel data missing"
    assert data.xpos is not None, "xpos data missing (body positions)"
    assert data.xquat is not None, "xquat data missing (body orientations)"
    assert data.site_xpos is not None, "site_xpos data missing"
    
    logger.info("✓ All required data fields present")
    
    # Check dimensions
    n_timesteps = data.qpos.shape[0]
    assert data.qvel.shape[0] == n_timesteps, "qvel timestep mismatch"
    assert data.xpos.shape[0] == n_timesteps, "xpos timestep mismatch"
    
    logger.info(f"✓ Consistent timesteps: {n_timesteps}")
    
    # Check data types
    assert data.qpos.dtype in [np.float32, jnp.float32], f"qpos wrong dtype: {data.qpos.dtype}"
    logger.info("✓ Correct data types")
    
    if reference_file:
        logger.info(f"Comparing against reference: {reference_file}")
        ref_traj = Trajectory.load(reference_file)
        
        # Compare dimensions
        assert data.qpos.shape[1] == ref_traj.data.qpos.shape[1], "qpos dimension mismatch with reference"
        assert data.xpos.shape[1:] == ref_traj.data.xpos.shape[1:], "xpos shape mismatch with reference"
        logger.info("✓ Dimensions match reference data")
    
    logger.info("Trajectory validation passed!")


def main(args):
    """Main preprocessing pipeline."""
    logger = setup_logger("preprocess", identifier="[Mocap Preprocessing]")
    
    logger.info(f"Starting preprocessing pipeline...")
    logger.info(f"Input: {args.input_file}")
    logger.info(f"Output: {args.output_file}")
    logger.info(f"Environment: {args.env_name}")
    
    # Step 1: Parse raw mocap data
    logger.info("Step 1: Parsing mocap data...")
    mocap_data = parse_mocap_data(args.input_file)
    
    # Step 2: Create minimal trajectory
    logger.info("Step 2: Creating minimal trajectory...")
    trajectory, env = create_minimal_trajectory(mocap_data, args.env_name)
    
    # Step 3: Extend with full kinematics
    logger.info("Step 3: Computing full kinematics...")
    extended_trajectory = extend_trajectory_with_kinematics(trajectory, env)
    
    # Step 4: Validate result
    logger.info("Step 4: Validating trajectory...")
    validate_trajectory(extended_trajectory, args.reference_file)
    
    # Step 5: Save to NPZ
    logger.info(f"Step 5: Saving to {args.output_file}...")
    extended_trajectory.save(args.output_file)
    
    logger.info("✅ Preprocessing completed successfully!")
    
    # Print summary
    data = extended_trajectory.data
    print(f"\nGenerated trajectory summary:")
    print(f"  Timesteps: {data.qpos.shape[0]}")
    print(f"  Joint DOF: {data.qpos.shape[1]}")
    print(f"  Bodies: {data.xpos.shape[1]}")
    print(f"  Sites: {data.site_xpos.shape[1]}")
    print(f"  Frequency: {extended_trajectory.info.frequency} Hz")
    print(f"  Duration: {data.qpos.shape[0] / extended_trajectory.info.frequency:.2f} seconds")
    print(f"  File size: {Path(args.output_file).stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess motion capture data for LocoMuJoCo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported input formats:
  BVH    - Biovision Hierarchy files (.bvh)
  FBX    - Autodesk FBX files (.fbx) - requires FBX SDK
  C3D    - 3D biomechanics files (.c3d) - requires inverse kinematics
  JSON   - Custom JSON format (.json)
  YAML   - Custom YAML format (.yaml/.yml)

Examples:
  # Process BVH file
  python preprocess_mocap.py --input_file walk.bvh --output_file walk.npz --env_name UnitreeH1
  
  # Process with validation
  python preprocess_mocap.py --input_file dance.bvh --output_file dance.npz --env_name UnitreeH1 --reference_file cached_datasets/UnitreeH1/squat.npz
  
  # Check required dependencies
  python preprocess_mocap.py --check_deps

Required packages for different formats:
  BVH: pip install bvh
  FBX: Install FBX SDK for Python
  C3D: pip install c3d
        """)
    
    parser.add_argument("--input_file", 
                        help="Path to input mocap data file (BVH/FBX/C3D/JSON/YAML)")
    parser.add_argument("--output_file",
                        help="Path to output NPZ file")
    parser.add_argument("--env_name", 
                        help="Environment name (e.g., UnitreeH1, UnitreeG1)")
    parser.add_argument("--reference_file", 
                        help="Optional reference NPZ file for validation")
    parser.add_argument("--check_deps", action="store_true",
                        help="Check availability of optional dependencies")
    parser.add_argument("--list_joints", action="store_true",
                        help="List joint names for the specified environment")
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps:
        print("Checking dependencies...")
        print(f"BVH support: {'✓' if BVH_AVAILABLE else '✗'}")
        print(f"FBX support: {'✓' if FBX_AVAILABLE else '✗'}")
        print(f"C3D support: {'✓' if C3D_AVAILABLE else '✗'}")
        
        if not BVH_AVAILABLE:
            print("  Install BVH: pip install bvh")
        if not FBX_AVAILABLE:
            print("  Install FBX: Download FBX SDK for Python from Autodesk")
        if not C3D_AVAILABLE:
            print("  Install C3D: pip install c3d")
        exit(0)
    
    # List joints for environment
    if args.list_joints:
        if not args.env_name:
            print("Error: --env_name required with --list_joints")
            exit(0)
            
        try:
            from loco_mujoco.environments import LocoEnv
            from loco_mujoco.smpl.retargeting import load_robot_conf_file
            
            env_cls = LocoEnv.registered_envs[args.env_name]
            robot_conf = load_robot_conf_file(args.env_name)
            env = env_cls(**robot_conf.env_params, th_params=dict(random_start=False, fixed_start_conf=(0, 0)))
            
            # Get joint names from the model
            joint_names = []
            for i in range(env._model.njnt):
                if env._model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE:  # Skip free joint
                    joint_name = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name:
                        joint_names.append(joint_name)
            
            print(f"\nJoint names for {args.env_name}:")
            for i, name in enumerate(joint_names):
                print(f"  {i:2d}: {name}")
            print(f"\nTotal joints: {len(joint_names)}")
            
        except Exception as e:
            print(f"Error loading environment {args.env_name}: {e}")
        exit(0)
    
    # Validate required arguments
    if not args.input_file or not args.output_file or not args.env_name:
        parser.print_help()
        exit(0)
    
    main(args)