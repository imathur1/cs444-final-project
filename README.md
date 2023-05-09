# cs444-final-project

Inspired by the AI Crowd Mono Depth Perception [challenge](https://www.aicrowd.com/challenges/scene-understanding-for-autonomous-drone-delivery-suadd-23/problems/mono-depth-perception), our project
analyzes the effects of targeted and untargeted adversarial attacks on depth estimation.

The `MiDaS` folder was taken from the MiDaS [repo](https://github.com/isl-org/MiDaS) and modified for adversarial attacks.
The `suadd23-monodepth-amazon` folder with evaluation tools was provided by the challenge.

We created `tutorial.ipynb` when first starting off, to get more familiar with how depth estimation works. It's based off this [tutorial](https://keras.io/examples/vision/depth_estimation/).

We used the Fast Gradient Sign Method (FGSM) with epsilons in [0, 0.05, 0.1, 0.15, 0.2, 0.25]. For targeted attacks we aimed to mislead
the model for specific masked objects, which were the objects provided by the semantic annotations. More info is found in the [paper](https://arxiv.org/pdf/2003.10315.pdf).
We also used 3 models, midas_v21_small_256, dpt_swin2_tiny_256, and dpt_swin2_large_384, provided by the MiDaS repo.

`visualizations.ipynb` contains the depth estimation for 3 pictures across all epsilons for all 3 models, for both untargeted and targeted attacks.
It also compares the models across various metrics, such as Mean SI Log, Mean RSME, Mean MAE, Mean Sq Rel, and Mean Abs Rel, which were provided by the AI crowd challenge.

We ran our code in a Google Cloud VM, so `fetch_data.py` was used to transfer some files locally for visualization.

# Run Adversarial Attacks
To run the adversarial attacks, you must first 
In the `MiDaS` folder 

# Evaluate Adversarial Attacks

# Visualize Adversarial Attacks
viz
