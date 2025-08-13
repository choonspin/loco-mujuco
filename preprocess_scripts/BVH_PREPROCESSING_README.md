# BVH to LocoMuJoCo Preprocessing Pipeline

A robust, general-purpose pipeline for converting BVH motion capture files to LocoMuJoCo NPZ format for imitation learning.

## Features

✅ **Universal BVH Support** - Works with any BVH file format  
✅ **Motion Extraction** - Captures all movements (not hardcoded patterns)  
✅ **Forward Alignment** - Automatic +Y forward alignment for consistency  
✅ **Adaptive Height** - Prevents ground penetration for all motion types  
✅ **Natural Postures** - Proper sitting, walking, and gesturing motions  
✅ **Cross-Subject** - Compatible across different people and recording setups  
✅ **Robust Mapping** - Conservative joint scaling for stability  

## Quick Start

### Prerequisites
```bash
# Activate LocoMuJoCo environment
source /home/choonspin/miniconda3/etc/profile.d/conda.sh
conda activate luco
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH
```

### Convert BVH to NPZ
```bash
python bvh_general_pipeline.py --input motion.bvh --output motion.npz
```

### View Motion
```bash
export MUJOCO_GL=osmesa
python simple_motion_viewer.py --input_file motion.npz --env_name SkeletonTorque
```

## Usage Examples

### Basic Conversion
```bash
python bvh_general_pipeline.py --input walking.bvh --output walking_motion.npz
```

### Custom Frequency
```bash
python bvh_general_pipeline.py --input motion.bvh --output motion.npz --frequency 30
```

### One-Line Convert + View
```bash
python bvh_general_pipeline.py --input motion.bvh --output motion.npz && export MUJOCO_GL=osmesa && python simple_motion_viewer.py --input_file motion.npz --env_name SkeletonTorque
```

## Pipeline Features

### Motion Types Supported
- **Walking/Running** - Locomotion with proper gait patterns
- **Sitting/Standing** - Postural transitions with horizontal thigh positioning
- **Gesturing** - Upper body movements and arm coordination
- **Object Manipulation** - Complex hand and arm interactions
- **General Movement** - Any recorded human motion

### Technical Capabilities
- **Adaptive Height Offset** - Automatically prevents ground contact
- **Forward Alignment** - Ensures consistent +Y forward orientation
- **Coordinate Transformation** - Proper BVH to SkeletonTorque mapping
- **Joint Limits** - Prevents unrealistic joint angles
- **Multi-Segment Spine** - Natural torso movement
- **Yaw-Only Root** - Stable root orientation

### Output Information
```
🔄 Processing BVH file: motion.bvh
📐 Pipeline: General BVH → +Y Forward Aligned → LocoMuJoCo NPZ
Processing BVH with 4972 frames...
SkeletonTorque joints: 28 joints
Initial root Y-rotation: 0.00°
Alignment rotation needed: -0.00°
📏 Adaptive height offset: 0.975m (min Y: -0.019m)
✅ Successfully saved trajectory to: motion.npz
📊 Frames: 4972, Frequency: 40.0 Hz
🧭 Forward alignment: -0.00° rotation applied
🎯 Coordinate system: +Y_forward_Z_up
```

## Joint Mapping

The pipeline maps BVH joints to SkeletonTorque format:

| BVH Joint | SkeletonTorque Joint | Scaling | Purpose |
|-----------|---------------------|---------|---------|
| `Character1_RightUpLeg` | `hip_flexion_r` | -1.0 | Hip forward/back |
| `Character1_RightLeg` | `knee_angle_r` | -1.0 | Knee flexion |
| `Character1_RightArm` | `arm_flex_r` | -0.8 | Arm swing |
| `Character1_RightForeArm` | `elbow_flex_r` | -1.5 | Elbow bending |
| `Character1_Spine` | `lumbar_extension` | -0.4 | Spine flexion |

## Configuration

### Environment Variables
```bash
# Required for LocoMuJoCo
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Required for headless rendering
export MUJOCO_GL=osmesa
```

### Parameters
- `--input`: Path to BVH file (required)
- `--output`: Output NPZ file path (optional, defaults to input name with .npz)
- `--frequency`: Target frequency in Hz (optional, default: 40.0)

## Troubleshooting

### Common Issues

**Segmentation Fault in Viewer**
- Normal after motion completes or is interrupted
- Motion playback works correctly before crash

**Ground Penetration**
- Pipeline automatically calculates adaptive height offset
- Minimum Y position is analyzed and compensated

**Wrong Facing Direction**
- Pipeline computes forward alignment automatically
- +Y forward orientation is applied consistently

**Unrealistic Joint Angles**
- Conservative scaling factors prevent extreme poses
- Joint limits are applied based on anatomical constraints

### Performance Notes
- Processing time scales with number of frames
- Memory usage depends on motion duration
- Viewer requires GPU for rendering

## Integration with LocoMuJoCo

The generated NPZ files are fully compatible with:
- LocoMuJoCo imitation learning algorithms
- JAX-based training (PPO, GAIL, AMP, DeepMimic)
- SkeletonTorque environment
- Trajectory replay and analysis tools

## File Structure

```
bvh_general_pipeline.py     # Main conversion script
simple_motion_viewer.py     # Motion visualization tool
output.npz                  # Generated trajectory file
```

## Dependencies

- LocoMuJoCo environment
- BVH parsing library (`bvh`)
- SciPy spatial transformations
- NumPy and JAX
- MuJoCo physics engine

---

**For support or issues, refer to the LocoMuJoCo documentation or project repository.**