import os
import logging
import numpy as np
import numpy.random as npr
import math
import torch
import torch.nn as nn
from freeseg import voxynth
from freeseg.augmentation.augmentbase import AugmentBase
from freeseg.utils import save_framedimage, get_ras_axes, bbox, centroid

class AugmentVoxynth(AugmentBase):
    def __init__(self, hyperparameters,
                 left_right_corresponding=None,
                 generation_labels=None,
                 output_dir=None,                 
                 device=None):
        super().__init__(hyperparameters,
                         left_right_corresponding=left_right_corresponding,
                         generation_labels=generation_labels,
                         output_dir=output_dir,                 
                         device=device)

        valid_augmentations = ["biasfieldcorruption",
                               "intensityaugmentation",
                               "biasfieldcorruptionandintensityaugmentation",]
        self.valid_augmentations.extend(valid_augmentations)
        # remove duplicates
        self.valid_augmentations = list(set(self.valid_augmentations))

        self.hyperparameters = hyperparameters        
        self.left_right_corresponding = left_right_corresponding
        self.generation_labels = generation_labels
        self.output_dir = output_dir
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = True if self.hyperparameters.get("verbose") else False
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentvoxynth.AugmentVoxynth' constructor")        

        # set up augmentations
        self.biasfieldcorruption = BiasFieldCorruption(self.hyperparameters, device=self.device)
        self.intensityaugmentation = IntensityAugmentation(self.hyperparameters, device=self.device)
        self.biasfieldcorruptionandintensityaugmentation = BiasFieldCorruptionAndIntensityAugmentation(self.hyperparameters, device=self.device)


class BiasFieldCorruption(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bias_field_probability = hyperparameters.get("bias_field_probability", 0.5)
        self.bias_field_max_magnitude = hyperparameters.get("bias_field_max_magnitude", 0.1)
        self.bias_field_smoothing_range = hyperparameters.get("bias_field_smoothing_range", None)
        self.bias_field_scale = hyperparameters.get("bias_field_scale", .025)
        self.bias_field_generation_method = hyperparameters.get("bias_field_generation_method", "blur")
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False

        assert (self.bias_field_generation_method == "blur" or self.bias_field_generation_method == "upsample"), \
            f"bias_field_generation_method {self.bias_field_generation_method} is not supported. The options are either 'blur' or 'upsample'"

        """
        assert (self.bias_field_smoothing_range is None), \
            f"'bias_field_smoothing_range' is deprecated, use 'bias_field_scale' instead"
        """
        if (self.verbose and self.bias_field_smoothing_range is not None):
            logging.debug(f"'augmentvoxynth.BiasFieldCorruption' respects bias_field_smoothing_range: {self.bias_field_smoothing_range}")
            return  # respect bias_field_smoothing_range if it is specified

        # calculate bias_field_smoothing_range from bias_field_scale
        if (self.bias_field_generation_method == "blur"):
            voxel_size = 1 / self.bias_field_scale
            fwhm = voxel_size
            gstd = fwhm/np.sqrt(np.log(256))  # natural logarithm
            self.bias_field_smoothing_range = [gstd, gstd]
        else:  # (self.bias_field_generation_method == "upsample")
            voxel_size = 1 / self.bias_field_scale
            self.bias_field_smoothing_range = [voxel_size, voxel_size]
        if (self.verbose):
            logging.debug(f"'augmentvoxynth.BiasFieldCorruption' calculated bias_field_smoothing_range: {self.bias_field_smoothing_range}, bias_field_generation_method: {self.bias_field_generation_method}")
            

        
    def forward(self, input, debugsaveprefix=None):
        """Applies bias field augmentation to the image volume."""
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentvoxynth.BiasFieldCorruption'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        bf_augmented_image = voxynth.augment.image_augment(
            image,
            voxsize=voxsize,
            bias_field_probability=self.bias_field_probability,
            bias_field_max_magnitude=self.bias_field_max_magnitude,
            bias_field_smoothing_range=self.bias_field_smoothing_range,
            bias_field_generation_method=self.bias_field_generation_method,
            sampling=self.sampling,
        )

        output = {
            'image': bf_augmented_image,
            'label': label,
            'prior': prior
                 }
        return output


class IntensityAugmentation(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.added_noise_probability = hyperparameters.get("added_noise_probability", 0.5)
        self.added_noise_max_sigma = hyperparameters.get("added_noise_max_sigma", 0.05)
        self.gamma_scaling_probability = hyperparameters.get("gamma_scaling_probability", 0.5)
        self.gamma_scaling_max = hyperparameters.get("gamma_scaling_max", 0.8)
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False
            
    def forward(self, input, debugsaveprefix=None):
        """Applies blurring and resampling to the image volume."""
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentvoxynth.IntensityAugmentation'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        blur_resampled_image = voxynth.augment.image_augment(
            image,
            added_noise_probability=self.added_noise_probability,
            added_noise_max_sigma=self.added_noise_max_sigma,
            gamma_scaling_probability=self.gamma_scaling_probability,
            gamma_scaling_max=self.gamma_scaling_max,
            sampling=self.sampling,
        )

        output = {
            'image': blur_resampled_image,
            'label': label,
            'prior': prior,
                 }
        return output


# biasfieldcorruption + intensityaugmentation in one voxynth.augment.image_augment() call
class BiasFieldCorruptionAndIntensityAugmentation(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # intensityaugmentation
        self.added_noise_probability = hyperparameters.get("added_noise_probability", 0.5)
        self.added_noise_max_sigma = hyperparameters.get("added_noise_max_sigma", 0.05)
        self.gamma_scaling_probability = hyperparameters.get("gamma_scaling_probability", 0.5)
        self.gamma_scaling_max = hyperparameters.get("gamma_scaling_max", 0.8)

        # biasfieldcorruption
        self.bias_field_probability = hyperparameters.get("bias_field_probability", 0.5)
        self.bias_field_max_magnitude = hyperparameters.get("bias_field_max_magnitude", 0.1)
        self.bias_field_smoothing_range = hyperparameters.get("bias_field_smoothing_range", None)
        self.bias_field_scale = hyperparameters.get("bias_field_scale", .025)
        self.bias_field_generation_method = hyperparameters.get("bias_field_generation_method", "blur")

        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False
        
        assert (self.bias_field_generation_method == "blur" or self.bias_field_generation_method == "upsample"), \
            f"bias_field_generation_method {self.bias_field_generation_method} is not supported. The options are either 'blur' or 'upsample'"

        """
        assert (self.bias_field_smoothing_range is None), \
            f"'bias_field_smoothing_range' is deprecated, use 'bias_field_scale' instead"
        """
        if (self.verbose and self.bias_field_smoothing_range is not None):
            logging.debug(f"'augmentvoxynth.BiasFieldCorruptionAndIntensityAugmentation' respects bias_field_smoothing_range: {self.bias_field_smoothing_range}")
            return  # respect bias_field_smoothing_range if it is specified

        # calculate bias_field_smoothing_range from bias_field_scale
        if (self.bias_field_generation_method == "blur"):
            voxel_size = 1 / self.bias_field_scale
            fwhm = voxel_size
            gstd = fwhm/np.sqrt(np.log(256))  # natural logarithm
            self.bias_field_smoothing_range = [gstd, gstd]
        else:  # (self.bias_field_generation_method == "upsample")
            voxel_size = 1 / self.bias_field_scale
            self.bias_field_smoothing_range = [voxel_size, voxel_size]           
        if (self.verbose):
            logging.debug(f"'augmentvoxynth.BiasFieldCorruptionAndIntensityAugmentation' calculated bias_field_smoothing_range: {self.bias_field_smoothing_range}, bias_field_generation_method: {self.bias_field_generation_method}")
                    
        
    def forward(self, input, debugsaveprefix=None):
        """Applies blurring and resampling to the image volume."""
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentvoxynth.BiasFieldCorruptionAndIntensityAugmentation'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        blur_resampled_image = voxynth.augment.image_augment(
            image,
            normalize=True,
            voxsize=voxsize,
            bias_field_probability=self.bias_field_probability,
            bias_field_max_magnitude=self.bias_field_max_magnitude,
            bias_field_smoothing_range=self.bias_field_smoothing_range,
            bias_field_generation_method=self.bias_field_generation_method,            
            added_noise_probability=self.added_noise_probability,
            added_noise_max_sigma=self.added_noise_max_sigma,
            gamma_scaling_probability=self.gamma_scaling_probability,
            gamma_scaling_max=self.gamma_scaling_max,
            sampling=self.sampling,
        )

        output = {
            'image': blur_resampled_image,
            'label': label,
            'prior': prior,
                 }
        return output
