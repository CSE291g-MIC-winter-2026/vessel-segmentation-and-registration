# vessel-segmentation-and-registration

### script.py

This script moves our pipeline from CPU to GPU. On top of the Registration_GPU_multiscale.ipynb, this script adds 2D-3D medical image registration to align a 3D volume to a 2D projection. It mimics the high-level strategy of DiffPose with finite differences instead of the deep-learning "differentiable" backbone. This script also connects the process of building synthetic data -> registaion -> evalutaion on the TopBrain 2025 dataset.

1. The Renderer (render_at_pose): This is the "forward model." It takes a 3D volume, applies 3D rotations and translations, blurs it to simulate camera effects, and then sums the voxels along one axis to create a 2D "pseudo-X-ray."

2. Similarity Metric (ncc): It uses Normalized Cross-Correlation to measure how well the rendered 2D image matches the real observed image.

3. Numerical Gradient Engine (finite_difference_gradient): Since the renderer is not "differentiable", the code calculates gradients by each of the 6 pose parameters (Rx, Ry, Rz, Tx, Ty, Tz) and measuring the change in NCC score.

4. Multiscale Strategy (multiscale_register): To prevent getting stuck in local errors, it runs the optimization in three stages: Coarse (Low resolution, high blur), Medium (Mid resolution, mid blur), Fine (High resolution, low blur).

5. Evaluation (compute_mtre): It calculates the Mean Target Registration Error (mTRE) in millimeters to see how far the final prediction is from the actual ground truth.

### hypothesis_test.ipynb
#### what it does?
- The notebook conducts statistical tests and generates visualizations for our three research questions.

#### how to run?
- run `scripts.py` to obtain data or download our output directly
  - output for naive pose: https://drive.google.com/drive/folders/1_plTyqcXWF07xpHGColF8AJ1Qup8V9fi?usp=share_link
  - output for random pose: https://drive.google.com/drive/folders/1GIiUNc07UaCogEUOgb1xYzRXoxkzXXgs?usp=share_link
- put data into a folder named `data` and run the notebook directly

### Registration_mTRE_multiscale.ipynb
#### what it does?
- The notebook show our data preprocessing process with a single patient's CTA. It visualizes the DSA we sythesized. 

#### how to run?
- download the dataset: https://topbrain2025.grand-challenge.org/data/
- store the dataset in your google drive folder `CSE291G`, inside the folder you should have two folders `labelsTr_topbrain_ct/topcow_ct_001_0000.nii.gz` and `imagesTr_topbrain_ct/topcow_ct_001.nii.gz`
- Then you can run the notebook to see our data exploration process and how we generated sythetic DSA images.
