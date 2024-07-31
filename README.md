Generation of anatomy-realistic 4D infant brain atlases with tissue maps using generative adversarial networks
==============================================================================================================

This repository contains the code for the paper 
"Generation of anatomy-realistic 4D infant brain atlases with tissue maps using generative adversarial networks"
by Tang et al. (2024). *Note: this repository is still under construction.*

Requirements
------------
The code was originally developed and tested ona a Linux workstation with the following packages:
- Python 3.8.17
- Torch 2.1.0
- NumPy 1.24.4

Also, for preparation of the data, one should organize their data into something like the following structure:
```
- BCP_train
    - XXBCP######
        - T1w_rigid.npz
        - T1w_affine.npz
        - T1w_affine_seg.npz
    - ...
- BCP_test
    - XXBCP######
        - T1w_rigid.npz
        - T1w_affine.npz
        - T1w_affine_seg.npz
    - ...
```
where `T1w_rigid.npz` is the rigidly aligned T1w image, `T1w_affine.npz` is the affine aligned T1w image, and `T1w_affine_seg.npz` is the affine aligned segmentation map.
Note that all the saved arrays are expected to have channel dimension at the very front, i.e., `(C, H, W, D)`.
See `src/data.py` for more details.

Files
-----
- `src/data.py`: Contains utilities for dataset preparation, creating weighted sampler, and creating rough templates.
- `src/discriminator.py`: Contains augmentations for the discriminator.
- `src/film.py`: Contains the FiLM mechanism.
- `src/layers.py`: Contains the custom layers used in the generator and discriminator.
- `src/losses.py`: Contains the losses used in the training.
- `src/mean_stream.py`: Contains the `MeanStream` class for collecting running statistics (for loss calculation).
- `src/networks.py`: Contains the generator and discriminator networks.
- `src/trilinear.py`: Contains the trilinear interpolation function.

Acknowledgements
----------------
For the BCP dataset, please refer to the following paper:

Howell et al., 
“The UNC/UMN Baby Connectome Project (BCP): An Overview of the Study Design and Protocol Development,” 
NeuroImage, vol. 185, pp. 891–905, Jan. 2019.

The baseline model [Atlas-GAN](https://github.com/neel-dey/Atlas-GAN) was originally developed by Dey et al. (2021) and implemented in TensorFlow. 
Here, we re-implemented the model in PyTorch and made modifications to the model to fit our use case for infant brain atlases.
