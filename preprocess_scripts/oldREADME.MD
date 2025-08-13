LocoMuJoCo Data Preprocessing & Visualization Tools

This directory contains two Python scripts to support motion capture data preprocessing and visualization for the LocoMuJoCo imitation learning benchmark:

    BVH to LocoMuJoCo NPZ Converter

    Simple Motion Viewer

1. BVH to LocoMuJoCo NPZ Converter

Description:
Convert BVH motion capture files into the LocoMuJoCo trajectory .npz format compatible with the SkeletonTorque humanoid environment.

Key Features:

    Parses BVH files using the bvh Python package

    Maps BVH joint rotations to SkeletonTorque joint angles with adaptive scaling and normalization

    Aligns root positions and rotations to LocoMuJoCo coordinate conventions

    Applies adaptive pelvis height offset for stable ground contact

    Computes joint velocities by finite differences

    Validates foot grounding via MuJoCo site checks

    Outputs .npz files ready for LocoMuJoCo training pipelines

Usage:

python bvh_to_locomujoco.py --input <input_bvh_file> [--output <output_npz_file>] [--frequency <hz>]

    --input (-i): Input BVH file path (required)

    --output (-o): Output NPZ file path (optional, defaults to input filename with .npz)

    --frequency (-f): Output frequency in Hz (default: 40.0)

Example:

python bvh_to_locomujoco.py -i walk.bvh -o walk.npz -f 40

2. Simple Motion Viewer

Description:
Visualize LocoMuJoCo .npz trajectory files by playing back motions in a specified environment (default: SkeletonTorque) with real-time rendering.

Key Features:

    Loads .npz trajectories using LocoMuJoCo’s Trajectory.load()

    Creates robot environment and loads motion

    Plays motion at half speed for clearer viewing

    Continuously loops playback until interrupted by the user

    Displays root position and playback progress in console

Usage:

python simple_motion_viewer.py --input_file <trajectory.npz> [--env_name SkeletonTorque]

    --input_file: Path to .npz trajectory file (required)

    --env_name: Environment to load (default: SkeletonTorque)

Example:

python simple_motion_viewer.py --input_file walk.npz --env_name SkeletonTorque

Requirements

    Python 3.10+

    numpy

    jax

    bvh (for BVH parsing)

    scipy

    mujoco Python bindings (MuJoCo 3.2.7 recommended)

    LocoMuJoCo installed and accessible

Notes & Recommendations

    The BVH converter assumes BVH files with joint names prefixed by Character1_. Update joint mapping in the script if your BVH uses different naming conventions.

    Adaptive height offset tuning helps prevent foot penetration or floating — check grounding validation output after conversion.

    The motion viewer runs playback at half the trajectory frequency by default to aid visualization. Adjust sleep timing if needed.

    Use Ctrl+C to stop motion playback gracefully.

    These tools integrate seamlessly with the LocoMuJoCo imitation learning training and evaluation pipelines.

Troubleshooting

    File not found errors: Verify input paths are correct.

    Joint mapping mismatches: Adjust the joint mapping dictionary in bvh_to_locomujoco.py as needed.

    Rendering issues: Ensure MuJoCo Python bindings and rendering libraries are installed properly, and environment variables (e.g., MUJOCO_GL) are set.

    Installation: Use editable install of LocoMuJoCo (pip install -e .) in your working environment.

For further assistance or customization requests, feel free to reach out!