import os
import logging
import numpy as np
import numpy.random as npr
import math
import torch
from fsdeepnet import voxynth
from fsdeepnet.augmentation.augmentbase import AugmentBase

# augmentation wrapper class derived from AugmentBase
class AugmentVoxynth(AugmentBase):
    def __init__(self,
                 hp,
                 transforms,
                 crop_size=None,
                 augmentation_dir=None,
                 device=None,
                 target_res=None,                 
                 num_channels=1,
                 sampling_hp=True,
                 verbose=False,
                 left_right_corresponding=None,
                 bbox_labels=None,
                 generation_labels=None,
                 generation_classes=None,
                 segmentation_labels=None,
                 **kwargs):
        super().__init__(hp, transforms, crop_size,
                         num_channels=num_channels,
                         left_right_corresponding=left_right_corresponding,
                         bbox_labels=bbox_labels,
                         generation_labels=generation_labels,
                         generation_classes=generation_classes,
                         segmentation_labels=segmentation_labels,
                         target_res=target_res,
                         augmentation_dir=augmentation_dir,                 
                         device=device,
                         sampling_hp=sampling_hp,
                         verbose=verbose,
                         **kwargs)

        valid_augmentations = ["biasfieldcorruption",
                               "intensityaugmentation",
                               "biasfieldcorruptionandintensityaugmentation",]

        # initialize required instance variables
        # 1. valid_augmentations: augmentations supported in the class, used to validate augmentations requested
        #    extend base class augmentations, remove duplicates
        self.valid_augmentations.extend(valid_augmentations)
        # remove duplicates
        self.valid_augmentations = list(set(self.valid_augmentations))

        if (device is None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (verbose):
            logging.debug(f"'fsdeepnet.augmentation.augmentvoxynth.AugmentVoxynth' constructor")        

        # 2. transforms: save the augmentations to be applied
        self.transforms = transforms

        # 3. individual augmentation instances: initiate augmentation instances that the wrapper class supports
        self.biasfieldcorruption = BiasFieldCorruption(hp=hp.get('biasfieldcorruption'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.intensityaugmentation = IntensityAugmentation(hp=hp.get('intensityaugmentation'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.biasfieldcorruptionandintensityaugmentation = BiasFieldCorruptionAndIntensityAugmentation(hp=hp.get('biasfieldcorruptionandintensityaugmentation'), device=device, sampling_hp=sampling_hp, verbose=verbose)


class BiasFieldCorruption(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.bias_field_probability = hp.get("bias_field_probability", 0.5)
        self.bias_field_max_magnitude = hp.get("bias_field_max_magnitude", 0.1)
        self.bias_field_smoothing_range = hp.get("bias_field_smoothing_range", None)
        self.bias_field_scale = hp.get("bias_field_scale", .025)
        self.bias_field_generation_method = hp.get("bias_field_generation_method", "blur")
        self.sampling = sampling_hp
        self.verbose = verbose

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
            

        
    def forward(self, input, **kwargs):
        """Applies bias field augmentation to the image volume."""
        if (self.verbose):
            logging.debug(f"'fsdeepnet.augmentation.augmentvoxynth.BiasFieldCorruption'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        geom = input.get("geom", None)

        # make it writeable or voxynth.augment.image_augment() will complain non-writable numpy array
        voxsize = np.copy(geom.voxsize)
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
            'prior': prior,
            'geom':  geom,            
                 }
        return output


class IntensityAugmentation(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.added_noise_probability = hp.get("added_noise_probability", 0.5)
        self.added_noise_max_sigma = hp.get("added_noise_max_sigma", 0.05)
        self.gamma_scaling_probability = hp.get("gamma_scaling_probability", 0.5)
        self.gamma_scaling_max = hp.get("gamma_scaling_max", 0.8)
        self.sampling = sampling_hp
        self.verbose = verbose
            
    def forward(self, input, **kwargs):
        """Applies blurring and resampling to the image volume."""
        if (self.verbose):
            logging.debug(f"'fsdeepnet.augmentation.augmentvoxynth.IntensityAugmentation'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
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
            'geom':  geom,            
                 }
        return output


# biasfieldcorruption + intensityaugmentation in one voxynth.augment.image_augment() call
class BiasFieldCorruptionAndIntensityAugmentation(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp

        # intensityaugmentation
        self.added_noise_probability = hp.get("added_noise_probability", 0.5)
        self.added_noise_max_sigma = hp.get("added_noise_max_sigma", 0.05)
        self.gamma_scaling_probability = hp.get("gamma_scaling_probability", 0.5)
        self.gamma_scaling_max = hp.get("gamma_scaling_max", 0.8)

        # biasfieldcorruption
        self.bias_field_probability = hp.get("bias_field_probability", 0.5)
        self.bias_field_max_magnitude = hp.get("bias_field_max_magnitude", 0.1)
        self.bias_field_smoothing_range = hp.get("bias_field_smoothing_range", None)
        self.bias_field_scale = hp.get("bias_field_scale", .025)
        self.bias_field_generation_method = hp.get("bias_field_generation_method", "blur")

        self.sampling = sampling_hp
        self.verbose = verbose
        
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
                    
        
    def forward(self, input, **kwargs):
        """Applies blurring and resampling to the image volume."""
        if (self.verbose):
            logging.debug(f"'fsdeepnet.augmentation.augmentvoxynth.BiasFieldCorruptionAndIntensityAugmentation'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        geom = input.get("geom", None)

        # make it writeable or voxynth.augment.image_augment() will complain non-writable numpy array
        voxsize = np.copy(geom.voxsize)
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
            'geom':  geom,
                 }
        return output
