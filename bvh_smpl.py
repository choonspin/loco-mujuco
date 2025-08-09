#!/usr/bin/env python3
"""
BVH to LocoMuJoCo Converter

This script converts BVH motion capture files to LocoMuJoCo NPZ format by:
1. Loading and parsing BVH files using pymo
2. Mapping BVH joints to LocoMuJoCo canonical humanoid joints (SMPLH format)
3. Adapting BVH data to match AMASS loader output format
4. Using LocoMuJoCo's existing retargeting pipeline (retarget.py)
5. Exporting final NPZ in AMASS-compatible format

The goal is to reuse LocoMuJoCo's existing retargeting pipeline completely,
only replacing the AMASS data loading stage with BVH loading.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation as R

# BVH parsing
try:
    import pymo
    from pymo.parsers import BVHParser
    from pymo.data import Joint, MocapData
    PYMO_AVAILABLE = True
except ImportError:
    PYMO_AVAILABLE = False
    print("Warning: pymo not available. Install with: pip install pymo")

# LocoMuJoCo imports
from loco_mujoco.smpl.retargeting import (
    fit_smpl_motion,
    extend_motion, 
    load_robot_conf_file,
    check_optional_imports,
    get_smpl_model_path,
    get_converted_amass_dataset_path
)
from loco_mujoco.smpl.const import SMPLH_BONE_ORDER_NAMES
from loco_mujoco.trajectory import Trajectory
from loco_mujoco.utils import setup_logger


# LocoMuJoCo canonical joint names (based on SMPLH but mapped to SkeletonTorque)
HUMANOID_JOINT_NAMES = [
    # Root (pelvis)
    "pelvis",
    # Lower body
    "hip_flexion_r", "hip_adduction_r", "hip_rotation_r", "knee_angle_r", "ankle_angle_r", 
    "hip_flexion_l", "hip_adduction_l", "hip_rotation_l", "knee_angle_l", "ankle_angle_l",
    # Spine/torso
    "lumbar_extension", "lumbar_bending", "lumbar_rotation",
    # Upper body
    "arm_flex_r", "arm_add_r", "arm_rot_r", "elbow_flex_r", "pro_sup_r", "wrist_flex_r", "wrist_dev_r",
    "arm_flex_l", "arm_add_l", "arm_rot_l", "elbow_flex_l", "pro_sup_l", "wrist_flex_l", "wrist_dev_l"
]

# BVH to SMPLH joint mapping - maps BVH joint names to SMPLH bone names  
# Based on the Character1_* naming convention from the actual BVH file
BVH_TO_SMPLH_MAPPING = {
    # Root and pelvis
    'Character1_Hips': 'Pelvis',
    'Hips': 'Pelvis',
    'hip': 'Pelvis', 
    'pelvis': 'Pelvis',
    'root': 'Pelvis',
    
    # Spine and torso
    'Character1_Spine': 'Torso',
    'Character1_Spine1': 'Spine', 
    'Character1_Spine2': 'Chest',
    'Character1_Spine3': 'Chest',
    'Character1_Neck': 'Neck',
    'Character1_Head': 'Head',
    'Spine': 'Torso',
    'Spine1': 'Spine', 
    'Spine2': 'Chest',
    'Spine3': 'Chest',
    'Chest': 'Chest',
    'Neck': 'Neck',
    'Neck1': 'Neck',
    'Head': 'Head',
    
    # Left leg
    'Character1_LeftUpLeg': 'L_Hip',
    'Character1_LeftLeg': 'L_Knee', 
    'Character1_LeftFoot': 'L_Ankle',
    'Character1_LeftToeBase': 'L_Toe',
    'LeftUpLeg': 'L_Hip',
    'LeftLeg': 'L_Knee', 
    'LeftFoot': 'L_Ankle',
    'LeftToeBase': 'L_Toe',
    'LeftToe_End': 'L_Toe',
    
    # Right leg  
    'Character1_RightUpLeg': 'R_Hip',
    'Character1_RightLeg': 'R_Knee',
    'Character1_RightFoot': 'R_Ankle', 
    'Character1_RightToeBase': 'R_Toe',
    'RightUpLeg': 'R_Hip',
    'RightLeg': 'R_Knee',
    'RightFoot': 'R_Ankle', 
    'RightToeBase': 'R_Toe',
    'RightToe_End': 'R_Toe',
    
    # Left arm
    'Character1_LeftShoulder': 'L_Thorax',
    'Character1_LeftArm': 'L_Shoulder',
    'Character1_LeftForeArm': 'L_Elbow',
    'Character1_LeftHand': 'L_Wrist',
    'LeftShoulder': 'L_Thorax',
    'LeftArm': 'L_Shoulder',
    'LeftForeArm': 'L_Elbow',
    'LeftHand': 'L_Wrist',
    
    # Right arm
    'Character1_RightShoulder': 'R_Thorax', 
    'Character1_RightArm': 'R_Shoulder',
    'Character1_RightForeArm': 'R_Elbow',
    'Character1_RightHand': 'R_Wrist',
    'RightShoulder': 'R_Thorax', 
    'RightArm': 'R_Shoulder',
    'RightForeArm': 'R_Elbow',
    'RightHand': 'R_Wrist',
    
    # Common alternative names
    'LHipJoint': 'L_Hip', 'RHipJoint': 'R_Hip',
    'LKnee': 'L_Knee', 'RKnee': 'R_Knee', 
    'LAnkle': 'L_Ankle', 'RAnkle': 'R_Ankle',
    'LShoulder': 'L_Shoulder', 'RShoulder': 'R_Shoulder',
    'LElbow': 'L_Elbow', 'RElbow': 'R_Elbow',
    'LWrist': 'L_Wrist', 'RWrist': 'R_Wrist',
}


def load_bvh(bvh_path: str) -> Tuple[MocapData, Dict]:
    """
    Load BVH file and extract motion data using pymo.
    
    Args:
        bvh_path: Path to BVH file
        
    Returns:
        Tuple of (MocapData object, metadata dict)
    """
    if not PYMO_AVAILABLE:
        raise ImportError("pymo is required for BVH loading. Install with: pip install pymo")
    
    parser = BVHParser()
    data = parser.parse(bvh_path)
    
    # PyMO framerate is frame time in seconds, so actual fps = 1/framerate
    actual_fps = 1.0 / data.framerate if data.framerate > 0 else 30.0  # fallback to 30fps
    
    metadata = {
        'framerate': actual_fps,
        'n_frames': data.values.shape[0],
        'joint_names': list(data.traverse()),  # traverse() returns joint names as strings
        'skeleton': data.skeleton
    }
    
    return data, metadata


def map_bvh_to_canonical(bvh_data: MocapData, mapping: Dict[str, str]) -> Dict:
    """
    Map BVH joint positions to canonical SMPLH format.
    
    Args:
        bvh_data: Loaded BVH motion data
        mapping: Dictionary mapping BVH joint names to SMPLH names
        
    Returns:
        Dictionary with canonical joint positions and metadata
    """
    n_frames = bvh_data.values.shape[0]
    joint_names = list(bvh_data.traverse())
    column_names = list(bvh_data.values.columns)
    
    # Extract global positions for mapped joints
    canonical_positions = {}
    missing_joints = []
    
    print(f"Available joints: {joint_names[:10]}...")  # Debug info
    print(f"Sample columns: {column_names[:10]}...")  # Debug info
    
    # Create mapping of available joints to their position columns
    available_positions = {}
    for joint_name in joint_names:
        # Look for position columns for this joint
        x_col = None
        y_col = None  
        z_col = None
        
        for col in column_names:
            if joint_name in col:
                if 'Xposition' in col:
                    x_col = col
                elif 'Yposition' in col:
                    y_col = col
                elif 'Zposition' in col:
                    z_col = col
        
        # If we found position columns, store the position data
        if x_col and y_col and z_col:
            # BVH coordinate system: X-right, Y-up, Z-forward
            # LocoMuJoCo coordinate system: X-right, Y-forward, Z-up
            # Also convert from centimeters to meters and ground the character
            
            x_pos = bvh_data.values[x_col].values / 100.0  # cm to m
            y_pos = bvh_data.values[y_col].values / 100.0  # cm to m (BVH Y = height)
            z_pos = bvh_data.values[z_col].values / 100.0  # cm to m
            
            # Ground the character by subtracting the initial height
            y_pos_grounded = y_pos - y_pos[0]
            
            # Convert coordinate system: BVH (X,Y,Z) → LocoMuJoCo (X,Z,Y_grounded)
            positions = np.column_stack([
                x_pos,           # X stays the same
                z_pos,           # BVH Z (forward) → LocoMuJoCo Y (forward)  
                y_pos_grounded   # BVH Y (up) → LocoMuJoCo Z (up), grounded
            ])
            available_positions[joint_name] = positions
    
    print(f"Found position data for {len(available_positions)} joints")
    
    # Map BVH joints to SMPLH canonical format
    for bvh_name, smplh_name in mapping.items():
        found = False
        
        # Direct match
        if bvh_name in available_positions:
            canonical_positions[smplh_name] = available_positions[bvh_name]
            found = True
            print(f"Mapped {bvh_name} -> {smplh_name}")
        else:
            # Try fuzzy matching for common joint name variations
            for available_joint in available_positions.keys():
                if bvh_name.lower() in available_joint.lower() or available_joint.lower().endswith(bvh_name.lower()):
                    canonical_positions[smplh_name] = available_positions[available_joint]
                    found = True
                    print(f"Fuzzy mapped {bvh_name} -> {available_joint} -> {smplh_name}")
                    break
        
        if not found:
            missing_joints.append(smplh_name)
    
    # Handle missing joints with defaults or interpolation
    for smplh_name in SMPLH_BONE_ORDER_NAMES:
        if smplh_name not in canonical_positions:
            missing_joints.append(smplh_name)
            # Create default position (can be improved with interpolation)
            canonical_positions[smplh_name] = np.zeros((n_frames, 3))
    
    if missing_joints:
        print(f"Warning: {len(missing_joints)} joints not found in BVH, using defaults: {missing_joints[:5]}...")
    
    # Use actual fps, not frame time
    actual_fps = 1.0 / bvh_data.framerate if bvh_data.framerate > 0 else 30.0
    
    return {
        'positions': canonical_positions,
        'n_frames': n_frames,
        'framerate': actual_fps,
        'missing_joints': missing_joints
    }

# Reverse mapping for rotations: SMPLH joint -> example BVH joint name
REVERSE_SMPLH_TO_BVH = {}
for bvh_joint, smplh_joint in BVH_TO_SMPLH_MAPPING.items():
    if smplh_joint not in REVERSE_SMPLH_TO_BVH:
        REVERSE_SMPLH_TO_BVH[smplh_joint] = bvh_joint

def extract_rotations_to_pose_aa(bvh_data: MocapData, joint_order: List[str]) -> np.ndarray:
    """
    Extract Euler rotations from BVH and convert to SMPL axis-angle pose vectors.
    
    Args:
        bvh_data: Loaded BVH motion data
        joint_order: List of SMPLH joint names in desired order
    
    Returns:
        pose_aa: np.ndarray of shape (n_frames, n_joints*3) with axis-angle rotations
    """
    n_frames = bvh_data.values.shape[0]
    n_joints = len(joint_order)
    pose_aa = np.zeros((n_frames, n_joints * 3))
    columns = bvh_data.values.columns
    
    for i, smplh_joint in enumerate(joint_order):
        bvh_joint = REVERSE_SMPLH_TO_BVH.get(smplh_joint, None)
        if bvh_joint:
            x_col = f"{bvh_joint}_Xrotation"
            y_col = f"{bvh_joint}_Yrotation"
            z_col = f"{bvh_joint}_Zrotation"
            if all(col in columns for col in [x_col, y_col, z_col]):
                euler_angles = np.stack([
                    bvh_data.values[x_col].values,
                    bvh_data.values[y_col].values,
                    bvh_data.values[z_col].values,
                ], axis=1)
                
                # Convert Euler angles (degrees) to axis-angle
                # You may need to adjust 'XYZ' order depending on your BVH
                rotations = R.from_euler('XYZ', euler_angles, degrees=True)
                pose_aa[:, i*3:(i+1)*3] = rotations.as_rotvec()
            else:
                # Missing rotation columns; leave zeros
                pass
        else:
            # No BVH joint mapped; leave zeros
            pass

    return pose_aa


def prepare_for_retarget(canonical_data: Dict, bvh_data: MocapData, target_fps: float = 40.0) -> Dict:
    """
    Prepare canonical joint data in AMASS-compatible format for retargeting.
    
    Args:
        canonical_data: Output from map_bvh_to_canonical (used here for root translation)
        bvh_data: Loaded BVH mocap data for extracting rotations
        target_fps: Target framerate for output
        
    Returns:
        Dictionary in AMASS format (pose_aa, trans, gender, betas, fps)
    """
    n_frames = canonical_data['n_frames']
    original_fps = canonical_data['framerate']
    
    # Extract root translation from pelvis
    if 'Pelvis' in canonical_data['positions']:
        trans = canonical_data['positions']['Pelvis'].copy()
    else:
        trans = np.zeros((n_frames, 3))
    
    # Convert coordinate system if needed (here assumed done in canonical_data)
    trans_converted = trans.copy()
    
    # Extract full pose axis-angle from BVH rotations
    pose_aa = extract_rotations_to_pose_aa(bvh_data, SMPLH_BONE_ORDER_NAMES)
    
    # Interpolate if fps differs
    if abs(original_fps - target_fps) > 0.1:
        print(f"Interpolating from {original_fps:.1f} to {target_fps:.1f} fps")
        original_times = np.linspace(0, n_frames / original_fps, n_frames)
        target_n_frames = int(n_frames * target_fps / original_fps)
        target_times = np.linspace(0, n_frames / original_fps, target_n_frames)
        
        trans_interp = np.zeros((target_n_frames, 3))
        for i in range(3):
            trans_interp[:, i] = np.interp(target_times, original_times, trans_converted[:, i])
        
        pose_aa_interp = np.zeros((target_n_frames, pose_aa.shape[1]))
        for i in range(pose_aa.shape[1]):
            pose_aa_interp[:, i] = np.interp(target_times, original_times, pose_aa[:, i])
        
        trans_converted = trans_interp
        pose_aa = pose_aa_interp
    
    amass_data = {
        'pose_aa': pose_aa,
        'trans': trans_converted,
        'gender': np.array(['neutral']),
        'betas': np.zeros(10),
        'fps': target_fps
    }
    
    print(f"Prepared AMASS data: {pose_aa.shape[0]} frames at {target_fps:.1f} fps")
    print(f"Trans shape: {trans_converted.shape}, Pose shape: {pose_aa.shape}")
    
    return amass_data


def run_retarget(amass_data: Dict, env_name: str, logger: logging.Logger, 
                 visualize: bool = False) -> Trajectory:
    """
    Run LocoMuJoCo's retargeting pipeline on AMASS-format data.
    
    Args:
        amass_data: Data in AMASS format 
        env_name: Target environment name
        logger: Logger instance
        visualize: Whether to visualize retargeting
        
    Returns:
        Retargeted Trajectory object
    """
    # Check imports
    check_optional_imports()
    
    # Load robot configuration
    robot_conf = load_robot_conf_file(env_name)
    
    # Get SMPL paths
    path_to_smpl_model = get_smpl_model_path()
    path_to_converted_amass_datasets = get_converted_amass_dataset_path()
    
    # Set up paths for shape optimization  
    path_robot_smpl_data = os.path.join(path_to_converted_amass_datasets, env_name)
    os.makedirs(path_robot_smpl_data, exist_ok=True)
    path_to_robot_smpl_shape = os.path.join(path_robot_smpl_data, "shape_optimized.pkl")
    
    # Check if shape optimization file exists, if not use None to trigger defaults
    if not os.path.exists(path_to_robot_smpl_shape):
        logger.info("No shape optimization file found, using default shape parameters")
        path_to_robot_smpl_shape = None
    else:
        logger.info(f"Using existing shape optimization: {path_to_robot_smpl_shape}")
    
    # Run SMPL retargeting - this is the core LocoMuJoCo pipeline
    logger.info("Running SMPL motion fitting...")
    trajectory = fit_smpl_motion(
        env_name=env_name,
        robot_conf=robot_conf,
        path_to_smpl_model=path_to_smpl_model,
        motion=amass_data,
        use_shape_optimization=True,
        path_to_shape_opt=path_to_robot_smpl_shape,
        visualize=visualize,
        logger=logger
    )
    
    return trajectory


def main(args):
    logger = setup_logger(name="BVHtoLocoMuJoCo", level=logging.INFO)
    
    logger.info(f"Loading BVH from {args.bvh_path}")
    bvh_data, bvh_metadata = load_bvh(args.bvh_path)
    
    logger.info("Mapping BVH data to canonical SMPLH joint positions")
    canonical_data = map_bvh_to_canonical(bvh_data, BVH_TO_SMPLH_MAPPING)
    
    logger.info("Preparing AMASS data from canonical positions and rotations")
    amass_data = prepare_for_retarget(canonical_data, bvh_data, target_fps=args.fps)
    
    logger.info(f"Running retargeting for environment '{args.env_name}'")
    trajectory = run_retarget(amass_data, args.env_name, logger, visualize=args.visualize)
    
    # Save output trajectory NPZ
    out_path = Path(args.out_path)
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.save(str(out_path))
    logger.info(f"Saved retargeted trajectory to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BVH to LocoMuJoCo NPZ format")
    parser.add_argument("bvh_path", type=str, help="Input BVH file path")
    parser.add_argument("env_name", type=str, help="LocoMuJoCo environment name")
    parser.add_argument("out_path", type=str, help="Output NPZ file path")
    parser.add_argument("--fps", type=float, default=40.0, help="Output FPS")
    parser.add_argument("--visualize", action="store_true", help="Visualize retargeting")
    args = parser.parse_args()
    
    main(args)
