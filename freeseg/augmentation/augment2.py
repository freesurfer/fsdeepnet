import os
import numpy as np
import numpy.random as npr
import math
import torch
import torch.nn as nn
from freeseg import voxynth
from freeseg.augmentation.augmentbase import AugmentBase
from freeseg.utils import save_framedimage, get_ras_axes, bbox, centroid

class Augment2(AugmentBase):
    def __init__(self, hyperparameters,
                 left_right_corresponding=None,
                 generation_labels=None,
                 output_dir=None,                 
                 device=None):
        valid_augmentations = ["biasfieldcorruption",
                               "intensityaugmentation",]
        self.valid_augmentations = self.valid_augmentations.extend(valid_augmentations)
        # remove duplicates
        self.valid_augmentations = list(set(self.valid_augmentations))

        self.hyperparameters = hyperparameters        
        self.left_right_corresponding = left_right_corresponding
        self.generation_labels = generation_labels
        self.output_dir = output_dir
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set up augmentations
        self.biasfieldcorruption = BiasFieldCorruption(self.hyperparameter, device=self.device)
        self.intensityaugmentation = IntensityAugmentation(self.hyperparameters, device=self.device)


# This is described in Hypothalamus paper (https://www.sciencedirect.com/science/article/pii/S1053811920307734)
#  "
#    The augmentation model also accounts for non-uniformities in the magnetic field commonly observed in MR scanners (Simmons et al., 1994).
#    Because this phenomenon translates into intensity inhomogeneities smoothly varying across MRI scans (Sled and Pike, 1998),
#    we model it with a multiplicative smooth field. As before, we sample a small low resolution field (e.g., of size 4 × 4 × 4),
#    and upscale it to image size with linear interpolation. Then, we take the voxel-wise exponential to ensure the non-negativity of this field.
#    Finally, we multiply the spatially deformed scan by the obtained bias field to corrupt its intensities (Fig. 1(c)).
#  "
class BiasFieldCorruption(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bias_field_std = hyperparameters.get("bias_field_max_magnitude", .7)  # SynthSeg
        self.bias_scale = hyperparameters.get("bias_field_scale", .025)
        self.prob = hyperparameters.get("bias_field_probability", 0.95)
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)

    def forward(self, image=None, label=None, prior=None, voxsize=None, aff=None):
        """
        Apply a smooth random bias field to the input tensor by applying the following steps:

        1) sample a value for the standard deviation of a centred normal distribution from U[0, bias_field_std)
        2) a small-size stationary velocity field (SVF) is sampled from this normal distribution
        3) the small SVF is then resized with trilinear interpolation to image size
        4) it is rescaled to positive values by taking the voxel-wise exponential
        5) it is multiplied to the input tensor.

        The input tensor is expected to have shape [C, H, W (,D)].

        The bias field is sampled and applied independently for each channel of the input tensor. 

        bias_field_std: if sampling = True,
                        max value to sample the standard deviation of a centred normal distribution from range [0, bias_field_std];
                        otherwise, standard deviation of a centred normal distribution
        bias_scale:     ratio between the shape of the input tensor and the shape of the sampled SVF.
        prob:           probability to apply this bias field corruption.
        sampling:       bool, optional
                        If True, sample the standard deviation of the Gaussian white noise from the range [0, bias_field_std);
                        otherwise, use bias_field_std as the standard deviation of a centred normal distribution
        """

        if (self.sampling and (not np.random.rand() < self.prob or self.bias_field_std <= 0)):
            return image
    
        num_channels = image.shape[0]
        ndims = image.ndim - 1
        image_shape = image.shape[1:]
    
        # sampling shapes, the bias field will be sampled and applied independently for each channel of the input tensor
        std_shape = [num_channels] + [1] * ndims   # [C, 1, 1(, 1)]
        small_bias_shape = [num_channels] + [math.ceil(image_shape[i] * self.bias_scale) for i in range(len(image_shape))]  # [C, h, w, (,d)]

        # sample small bias field (step 1 and 2)
        # stddev = U(0, bias_field_std) if sampling = True; otherwise stddev = bias_field_std
        stddev = self.bias_field_std * torch.rand(std_shape, device=image.device) if (self.sampling) else self.bias_field_std
        bias_field_tensor = stddev * torch.randn(small_bias_shape, device=image.device)   # N(0, stddev)
    
        # resize bias field and take exponential (step 3 and 4)
        mode = "trilinear" if (ndims == 3) else "bilinear"
        bias_field_tensor = torch.nn.functional.interpolate(bias_field_tensor.unsqueeze(0), image_shape, mode=mode)
        bias_field_tensor = bias_field_tensor.squeeze(0)  # remove the dummy batch dimension
        bias_field_tensor = torch.exp(bias_field_tensor)

        # element-wise multiplication (step 5)
        bf_augmented_image = torch.mul(bias_field_tensor, image)

        return bf_augmented_image, label, prior, None


# This is described in Hypothalamus paper (https://www.sciencedirect.com/science/article/pii/S1053811920307734)
#   "
#     In order to make the network robust against acquisition procedures,
#     we add further global intensity augmentation by shifting the brightness and contrast of the image with randomly sampled values (Fig. 1(d)).
#     The obtained scan is subsequently flipped along the right-left axis with a probability of 0.5 (Fig. 1(e)), and randomly cropped to a size of 160^3,
#     which is more than large enough to ensure that the hypothalamus is always present in the resulting scan.
#     Finally, intensities are rescaled between [0,1] with min-max normalisation.
#     Additional examples of augmented images are shown in the Supplementary materials (Fig. S1).
#   "
class IntensityAugmentation(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.noise_std = hyperparameters.get("added_noise_max_sigma", 1.0)  # default is 0 for SynthSeg, no white noise added
        self.normalize = hyperparameters.get("normalize", True)
        self.gamma_std = hyperparameters.get("gamma_scaling_max", 0.5)
        self.prob_noise = hyperparameters.get("added_noise_probability", 0.95)
        self.prob_gamma = hyperparameters.get("gamma_scaling_probability", 1)
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)

    def forward(self, image=None, label=None, prior=None, voxsize=None, aff=None):
        """
        Augment the intensities of the input tensor. All channels are augmented separately.

        The following steps are applied (all are optional):
        1) white noise corruption, with a randomly sampled std dev from U[0, noise_std)
        2) min-max normalisation
        3) gamma augmentation (i.e. voxel-wise exponentiation by a randomly sampled power from N(0, gamma_std))

        The input tensor is expected to have shape [C, H, W (,D)].

        The noise and gamma are sampled and applied independently for each channel of the input tensor.

        noise_std:  if sampling = True,
                    max value to sample the standard deviation of the Gaussian white noise from the range [0, noise_std];
                    otherwise, standard deviation of the Gaussian white noise.
                    Default is 0, where white noise corruption is skipped.
        normalize:  whether to apply min-max normalisation, to normalise between 0 and 1. Default is True.
        gamma_std:  standard deviation of the normal distribution from which we sample gamma.
                    Default is 0, where no gamma augmentation occurs.
        prob_noise: probability to apply noise injection
        prob_gamma: probability to apply gamma augmentation
        sampling:   bool, optional
                    If True, sample the standard deviation of the Gaussian white noise from the range [0, noise_std);
                    otherwise, use noise_std as the standard deviation of the Gaussian white noise
        """
    
        num_channels = image.shape[0]
        ndims = image.ndim - 1

        # noise and gamma are sampled and applied independently for each channel of the input tensor
        sample_shape = [num_channels] + [1] * ndims # [C, 1, 1 (,1)]
    
        # add noise with predefined probability
        if (self.noise_std > 0 and np.random.rand() < self.prob_noise):
            # noise_stddev = U(0, noise_std) if sampling = True; otherwise noise_stddev = noise_std
            noise_stddev = self.noise_std * torch.rand(sample_shape, device=image.device) if (self.sampling) else self.noise_std
            noise = noise_stddev * torch.randn(image.shape, device=image.device)       # N(0, noise_stddev)
            image += noise

        # normalize
        if (self.normalize):
            # simple min and max
            axis = tuple(dim for dim in range(1, ndims+1)) # axis=(H, W (,D))
            m = torch.amin(image, dim=axis) # [C, 1]
            M = torch.amax(image, dim=axis) # [C, 1]

            m = torch.reshape(m, sample_shape) # [C, 1, 1 (,1)]
            M = torch.reshape(M, sample_shape) # [C, 1, 1 (,1)]

            # normalize
            image = torch.clip(image, min=m, max=M)
            image = (image - m) / (M - m + torch.finfo(torch.float32).eps)

        # apply voxel-wise exponentiation with predefined probability
        if (self.gamma_std > 0 and np.random.rand() < self.prob_gamma):
            gamma = self.gamma_std * torch.randn(sample_shape, device=image.device)   # N(0, gamma_std)
            image = torch.pow(image, torch.exp(gamma))

        return image, label, prior, None
