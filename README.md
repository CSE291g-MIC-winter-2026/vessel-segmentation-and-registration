# vessel-segmentation-and-registration

### script.py

This script moves our pipeline from CPU to GPU. On top of the Registration_GPU_multiscale.ipynb, this script adds 2D-3D medical image registration to align a 3D volume to a 2D projection. It mimics the high-level strategy of DiffPose with finite differences instead of the deep-learning "differentiable" backbone. This script also connects the process of building synthetic data -> registaion -> evalutaion on the TopBrain 2025 dataset.

1. The Renderer (render_at_pose): This is the "forward model." It takes a 3D volume, applies 3D rotations and translations, blurs it to simulate camera effects, and then sums the voxels along one axis to create a 2D "pseudo-X-ray."

2. Similarity Metric (ncc): It uses Normalized Cross-Correlation to measure how well the rendered 2D image matches the real observed image.

3. Numerical Gradient Engine (finite_difference_gradient): Since the renderer is not "differentiable", the code calculates gradients by each of the 6 pose parameters (Rx, Ry, Rz, Tx, Ty, Tz) and measuring the change in NCC score.

4. Multiscale Strategy (multiscale_register): To prevent getting stuck in local errors, it runs the optimization in three stages: Coarse (Low resolution, high blur), Medium (Mid resolution, mid blur), Fine (High resolution, low blur).

5. Evaluation (compute_mtre): It calculates the Mean Target Registration Error (mTRE) in millimeters to see how far the final prediction is from the actual ground truth.
