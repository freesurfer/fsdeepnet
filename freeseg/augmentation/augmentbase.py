import logging
import numpy as np
import numpy.random as npr
import math
import torch
from freeseg import voxynth
from freeseg.utils import utility as utils
from freeseg.filter import Filter

class AugmentBase:
    def __init__(self, hp,
                 transforms,
                 crop_size,
                 num_channels=1,
                 left_right_corresponding=None,
                 bbox_labels=None,
                 generation_labels=None,
                 generation_classes=None,
                 segmentation_labels=None,
                 target_res=None,
                 output_dir=None,                 
                 device=None,
                 sampling_hp=True,
                 verbose=False):
        valid_augmentations_base = ["flip",
                                    "spatialdeformation",
                                    "randomcrop", "randomcentercrop", "centercrop", "centroidcrop",
                                    "biasfieldcorruption",
                                    "intensityaugmentation",
                                    "sampleconditionalgmm",
                                    "rescalevolume",
                                    "gaussianblur",
                                    "resamplevolume",
                                    "mimicresolution",
                                    "remaplabels",
                                   ]
        self.valid_augmentations = valid_augmentations_base.copy()

        self.output_dir = output_dir   # used in apply_augmentations()
        if (device is None):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if (verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.AugmentBase' constructor")

        self.transforms = transforms
        
        # set up augmentations
        self.flip = Flip(left_right_corresponding, hp=hp.get('flip'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.spatialdeformation = SpatialDeformation(hp=hp.get('spatialdeformation'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.randomcrop = RandomCrop(crop_size, bbox_labels=bbox_labels, hp=hp.get('randomcrop'), device=device, mode='random', sampling_hp=sampling_hp, verbose=verbose)
        self.randomcentercrop = RandomCrop(crop_size, bbox_labels=bbox_labels, hp=hp.get('randomcentercrop'), device=device, mode='center', sampling_hp=sampling_hp, verbose=verbose)
        self.centercrop = CenterCrop(crop_size, hp=hp.get('centercrop'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.centroidcrop = CentroidCrop(crop_size, hp=hp.get('centroidcrop'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.biasfieldcorruption = BiasFieldCorruption(hp=hp.get('biasfieldcorruption'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.intensityaugmentation = IntensityAugmentation(hp=hp.get('intensityaugmentation'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.sampleconditionalgmm = SampleConditionalGMM(generation_labels, generation_classes, hp=hp.get('sampleconditionalgmm'), num_channels=num_channels, device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.rescalevolume = RescaleVolume(hp=hp.get('rescalevolume'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.gaussianblur = GaussianBlur(hp=hp.get('gaussianblur'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.resamplevolume = ResampleVolume(target_res, hp=hp.get('resamplevolume'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.mimicresolution = MimicResolution(hp=hp.get('mimicresolution'), device=device, sampling_hp=sampling_hp, verbose=verbose)
        self.remaplabels = RemapLabels(generation_labels, dest_labels=segmentation_labels, hp=hp.get('remaplabels'), device=device, sampling_hp=sampling_hp, verbose=verbose)


class Flip(torch.nn.Module):
    def __init__(self, left_right_corresponding, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()

        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp        
        self.flip_prob = hp.get("flip_prob", 0.5)
        self.left_right_corresponding = left_right_corresponding
        self.verbose = verbose

    # ??? todo: flip priors ???        
    def forward(self, input, debugsaveprefix=None):
        """Applies a random left-right flip to image and label volumes."""
        """Swaps left-right labels on label volume."""

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)

        if (np.random.rand() >= self.flip_prob):
            # no flipping
            return dict(image=image, label=label)

        assert geom is not None, 'geom should not be None when applying flipping'
        assert self.left_right_corresponding is not None, 'left_right_corresponding should not be None when applying flipping'
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.Flip'")

        aff = geom.vox2world.matrix
        if (image is not None):
            ndims = len(image.shape[1:])
        else:
            ndims = len(label.shape[1:])
        
        # swap left-right labels
        n_left_right_labels = len(self.left_right_corresponding)
        left_right_corresponding = np.array(self.left_right_corresponding)
        left_labels  = left_right_corresponding[np.arange(start=0, stop=n_left_right_labels, step=2)]
        right_labels = left_right_corresponding[np.arange(start=1, stop=n_left_right_labels, step=2)]
        for idx in range(int(n_left_right_labels/2)):
            left_indices  = (label == left_labels[idx])
            right_indices = (label == right_labels[idx])
            label[left_indices]  = right_labels[idx]
            label[right_indices] = left_labels[idx]

        # find the left-right axis
        axis = utils.get_ras_axes(aff, ndims)[0]

        # image, label ([C, H, W(, D)]) have been reoriented to RAS
        flipped_image = None
        if (image is not None):
            flipped_image = image.flip([axis+1])
        flipped_label = label.flip([axis+1])
    
        output = {
            'image': flipped_image,
            'label': flipped_label,
                 }
        return output


class SpatialDeformation(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()

        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.affine_probability = hp.get("affine_probability", 1.0)
        self.max_translation = hp.get("max_translation", 5.0)
        self.max_rotation = hp.get("max_rotation", 5.0)
        self.max_shearing = hp.get("max_shearing", 0.015)        
        self.max_scaling = hp.get("max_scaling", 1.1)
        self.warp_probability = hp.get("warp_probability", 1.0)
        self.warp_integrations = hp.get("warp_integrations", 7)
        self.warp_generation_method = hp.get("warp_generation_method", "upsample")
        self.warp_smoothing_range = hp.get("warp_smoothing_range", [10, 20])
        self.warp_magnitude_range = hp.get("warp_magnitude_range", [1, 2])
        self.sampling = sampling_hp
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
        """Applies a random spatial transformation to image and label volumes."""

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.SpatialDeformation'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)

        """
        trf and aff_matrix are the same transform
        aff_matrix is None if there is non-linear component in trf
        """
        # voxsize is default to 1        
        trf, aff_matrix = voxynth.transform.random_transform(
            shape=label.shape[1:],
            device=self.device,
            affine_probability=self.affine_probability,
            max_translation=self.max_translation,
            max_rotation=self.max_rotation,
            max_shearing=self.max_shearing,        
            max_scaling=self.max_scaling,
            warp_probability=self.warp_probability,
            warp_integrations=self.warp_integrations,
            warp_smoothing_range=self.warp_smoothing_range,
            warp_magnitude_range=self.warp_magnitude_range,
            perlin_method=self.warp_generation_method,
            isdisp=True,  # the transformation is returned as displacement field
            sampling=self.sampling,
            return_aff=True
        )

        if (debugsaveprefix is not None):
            # trf is displacement in crs
            if (trf is not None):
                from surfa.transform import Warp
                trf_cpu = trf.cpu().detach().numpy().astype(np.float32)
                warp = Warp(trf_cpu, source=geom, target=geom, format=Warp.Format.disp_crs)
                warp.save(f"{debugsaveprefix}_warp_dispcrs.mgz")

            if (aff_matrix is not None):
                from surfa.transform import Affine
                aff_matrix_cpu = aff_matrix.cpu().detach().numpy().astype(np.float64)

                # aff_matrix is vox2vox mapping from target to source, which rotates around the image center
                # convert it to a standard-format affine that rotates around the corner (origin)
                center = np.eye(4)
                center[:3, -1] = -(np.asarray(label.shape[1:]) - 1)/2
                aff_matrix_cpu = np.linalg.inv(center) @ aff_matrix_cpu @ center

                affine = Affine(aff_matrix_cpu, source=geom, target=geom)
                affine.save(f"{debugsaveprefix}_vox2vox_trg2src.lta")

        transformed_image = None
        if (image is not None):
            transformed_image = voxynth.transform.spatial_transform(image, trf)
        transformed_label = voxynth.transform.spatial_transform(label, trf, method="nearest")

        transformed_priors = None
        if (prior is not None):
            transformed_priors = voxynth.transform.spatial_transform(prior, trf, method="nearest")

        output = {
            'image': transformed_image,
            'label': transformed_label,
            'prior': transformed_priors,
                 }
        return output


class RandomCrop(torch.nn.Module):
    def __init__(self, crop_size, bbox_labels=None, hp=None, device=None, mode='random', sampling_hp=True, verbose=False):
        super().__init__()
        self.mode = mode
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.crop_size = torch.tensor(crop_size, device=self.device)
        self.bbox_labels = bbox_labels
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
        """
        Randomly crop input tensors to a given shape. 
        The input tensors are non-batched, expected to have shape [C, H, W(, D)].

        Returns:
            cropped_image, cropped_label, cropped_prior, crop_idx
            TODO: raise exception if there is no crops found that fit the bounding box of all labels
                  handle batch > 1 ???
        """

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.RandomCrop'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
            
        # assuming image and label have the same dimensions
        if (image is not None):
            vol_shape = torch.tensor(image.shape[1:], device=image.device)
            device = image.device
        else:
            vol_shape = torch.tensor(label.shape[1:], device=label.device)
            device = label.device
        
        vol_ndims = len(vol_shape)

        bbox_upper = torch.zeros(vol_ndims, device=device, dtype=int)
        bbox_lower = vol_shape
        if (self.bbox_labels is not None):
            # calculate lower and upper bounds for the label bounding box
            bbox_lower, bbox_upper = utils.bbox(label, self.bbox_labels, verbose=self.verbose)
            if (self.verbose):
                logging.debug(f"crop around label bounding box {bbox_lower} - {bbox_upper}")
        
            # make sure crop_size > (bbox_upper - bbox_lower)
            """
            # ??? TODO ???
            if (torch.any(crop_size < (bbox_upper - bbox_lower))):
                raise exception
            """

        zero_tensor = torch.zeros(vol_ndims, device=device, dtype=int)
        if (self.mode == 'random'):
            if (self.bbox_labels is None):
                crop_min_val = zero_tensor
                crop_max_val = vol_shape - self.crop_size
            else:
                """
                |           |                    |            |
                -----------------------------------------------
                0         bound1               bound2      image size

                The [bbox_lower, bbox_upper] can be in any of these boundary.

                crop_min_val and crop_max_val need to be calculated accordingly
                to make sure after the random crop the bbox is inside the cropped 
                image of crop_size.

                Example of those boundary:
                    image size = [256 256 256]
                    crop_size  = [160 160 160]
                    bound1     = [ 96  96  96]
                    bound2     = [160 160 160]
                """
                bound1 = vol_shape - self.crop_size
                bound2 = self.crop_size

                # minimum to crop so that it will include bbox_upper
                # the value depends on if bbox_upper > bound2
                crop_min_val = torch.maximum(zero_tensor,  (bbox_upper - bound2))
                # maximum to crop so that it will include bbox_lower
                crop_max_val = torch.minimum(bbox_lower, bound1)

            # U[crop_min_val, crop_max_val)            
            start_coords = ((crop_min_val - crop_max_val) * torch.rand(vol_ndims, device=device) + crop_max_val).int()
            end_coords   = start_coords + self.crop_size
        elif (self.mode == 'center'):
            half_crop = (self.crop_size/2).int()
        
            if (self.bbox_labels is None):
                start_center = half_crop
                end_center = vol_shape - half_crop
            else:
                """
                |                     |       |              |       |                   |
                --------------------------------------------------------------------------
                0                  center1  bound1        bound2  center2           image size

                The [bbox_lower, bbox_upper] can be in any of these boundary.

                start_center and end_center need to be calculated accordingly
                to make sure after the random center crop the bbox is inside 
                the cropped image of crop_size.

                Example of those boundary:
                    image size = [256 256 256]
                    crop_size  = [160 160 160]
                    center1    = [ 80  80  80]
                    bound1     = [ 96  96  96]
                    bound2     = [160 160 160]
                    center2    = [176 176 176]
                """
            
                # initial values for start_center and end_center
                # where [bbox_lower, bbox_upper] is within [bound1, bound2]
                start_center = half_crop.clone()  # make a copy for start_center to be modified later
                end_center = vol_shape - half_crop

                bound1 = end_center - half_crop
                bound2 = start_center + half_crop
                if (torch.any(bbox_lower < bound1)):
                    # need to adjust end_center
                    distance = bound1 - bbox_lower
                    end_center -= torch.maximum(zero_tensor,  distance)    
                if (torch.any(bbox_upper > bound2)):
                    # need to adjust start center                
                    distance = bbox_upper - bound2
                    start_center += torch.maximum(zero_tensor,  distance)

            # U[start_center, end_center)
            center_point = ((start_center - end_center) * torch.rand(vol_ndims, device=device) + end_center).int()
            start_coords = torch.maximum(center_point-half_crop, zero_tensor)
            end_coords   = torch.minimum(center_point+half_crop, vol_shape)
                
        # Calculate the crop indices
        crop_idx = torch.concat((start_coords, end_coords)).int()
        if (self.verbose):
            dbg_msg = f"randomcrop({self.mode}) - {vol_shape.tolist()} => {self.crop_size.tolist()}, "
            if (self.bbox_labels is not None):
                dbg_msg += f"bbox: {bbox_lower.tolist()} - {bbox_upper.tolist()}, "
            if (self.mode == 'center'):
                dbg_msg += f"(start_center: {start_center.tolist()}, end_center: {end_center.tolist()}), center_point: {center_point.tolist()}, "
            else:
                dbg_msg += f"(crop_min_val: {crop_min_val.tolist()}, crop_max_val: {crop_max_val.tolist()}), "
            dbg_msg += f"start_coords: {start_coords.tolist()}, end_coords: {end_coords.tolist()}, "
            dbg_msg += f"crop indices: {crop_idx.tolist()}"
            logging.debug(dbg_msg)
            
        # check if bbox_lower/bbox_upper are inside start_coords/end_coords
        if (torch.any(bbox_lower < start_coords) or torch.any(bbox_upper > end_coords)):
            dbg_msg = f"***CROPPING WARNING*** randomcrop({self.mode}) - {vol_shape.tolist()} => {self.crop_size.tolist()}, "
            if (self.bbox_labels is not None):
                dbg_msg += f"bbox: {bbox_lower.tolist()} - {bbox_upper.tolist()}, "
            if (self.mode == 'center'):
                dbg_msg += f"(start_center: {start_center.tolist()}, end_center: {end_center.tolist()}), center_point: {center_point.tolist()}, "
            else:
                dbg_msg += f"(crop_min_val: {crop_min_val.tolist()}, crop_max_val: {crop_max_val.tolist()}), "            
            dbg_msg += f"start_coords: {start_coords.tolist()}, end_coords: {end_coords.tolist()}"
            dbg_msg += f"crop indices: {crop_idx.tolist()}"
            logging.debug(dbg_msg)
        
            """
            # ??? TODO ???
            raise exception
            """

        if (vol_ndims == 3):
            output = {
                'image': image[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (image is not None) else None,
                'label': label[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (label is not None) else None,
                'prior': prior[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (prior is not None) else None,
                'crop_idx': crop_idx,
                     }
        else:
            output = {
                'image': image[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (image is not None) else None,
                'label': label[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (label is not None) else None,
                'prior': prior[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (prior is not None) else None,
                'crop_idx': crop_idx,
                     }

        return output
    

class CentroidCrop(torch.nn.Module):
    def __init__(self, crop_size, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp
        self.max_offset = hp.get("max_offset", None)            
        self.crop_size = torch.tensor(crop_size, device=self.device)
        self.sampling = sampling_hp                    
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
        """Applies a crop centered around a specified point or the image center.

        Args:
            image (torch.Tensor): The 3D image to crop (C, H, W, D), it is non-batched.
            crop_size (tuple): The desired crop size, e.g., (160, 160, 160).
            center_point (tuple, optional): Coordinates of the center point for the crop 
                (x, y, z). If None, the image center is used. 

        Returns:
            torch.Tensor: The cropped image.
            numpy array:  The indices where the image is cropped.
        """    

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.CentroidCrop'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)            
        
        # input image is non-batched tensor
        if (image is not None):
            vol_shape = torch.tensor(image.shape[1:], device=image.device)
            device = image.device
        else:
            vol_shape = torch.tensor(label.shape[1:], device=label.device)
            device = label.device
        vol_ndims = len(vol_shape)
        
        if (not torch.any(vol_shape > self.crop_size)):
            return dict(image=image, label=label, prior=prior, crop_idx=None)

        crop_idx = None        
        center_point = None
        # calculate the center point to crop the image/label around    
        if (label is not None):
            center_point = utils.centroid(label.squeeze(0), verbose=self.verbose)
        else:
            center_point = (vol_shape/2).int()   #tuple(dim // 2 for dim in vol_shape)

        # off_center = U[-self.max_offset, +self.max_offset) if sampling = True; otherwise off_center = self.max_offset
        if (self.max_offset is not None):
            if (not self.sampling):
                off_center = self.max_offset
            else:
                # U[-self.max_offset, +self.max_offset)
                off_center = [(-self.max_offset[i]-self.max_offset[i]) * torch.rand(1, device=device) + self.max_offset[i] for i in range(vol_ndims)]
                off_center = [math.ceil(off_center[i]) for i in range(vol_ndims)]
            center_point = center_point - torch.tensor(off_center, device=device)

        zero_tensor = torch.zeros(vol_ndims, device=device, dtype=int)
        crop_half = (self.crop_size/2).int()
        if (label is not None):
            # adjust the calculated center so that croppred image will have crop_size
            if (torch.any(center_point < crop_half)):
                distance = crop_half - center_point
                center_point += torch.maximum(zero_tensor,  distance)    
            if (torch.any(center_point > (vol_shape - crop_half))):
                distance = center_point - (vol_shape - crop_half)
                center_point -= torch.maximum(zero_tensor,  distance)

        # Calculate the starting and ending indices for the crop region
        start_coords = torch.maximum(zero_tensor, center_point - crop_half)
        end_coords = torch.minimum(center_point + crop_half, vol_shape)
        crop_idx = torch.concat((start_coords, end_coords))
        if (self.verbose):
            logging.debug(f"adjusted crop center: {center_point.tolist()}, crop indices: {crop_idx.tolist()}")

        #ndims = len(label.shape[1:])
        if (vol_ndims == 3):
            output = {
                'image': image[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (image is not None) else None,
                'label': label[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (label is not None) else None,
                'prior': prior[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (prior is not None) else None,
                'crop_idx': crop_idx
                     }
        else:
            output = {
                'image': image[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (image is not None) else None,
                'label': label[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (label is not None) else None,
                'prior': prior[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (prior is not None) else None,
                'crop_idx': crop_idx
                     }

        return output


class CenterCrop(torch.nn.Module):
    def __init__(self, crop_size, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp
        self.max_offset = hp.get("max_offset", None)            
        self.crop_size = torch.tensor(crop_size, device=self.device)
        self.sampling = sampling_hp
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
        """Applies a crop centered around a specified point or the image center.

        Args:
            image (torch.Tensor): The 3D image to crop (C, H, W, D), it is non-batched.
            crop_size (tuple): The desired crop size, e.g., (160, 160, 160).
            center_point (tuple, optional): Coordinates of the center point for the crop 
                (x, y, z). If None, the image center is used. 

        Returns:
            torch.Tensor: The cropped image.
            numpy array:  The indices where the image is cropped.
        """    

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.CenterCrop'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        # input image is non-batched tensor
        if (image is not None):
            vol_shape = torch.tensor(image.shape[1:], device=image.device)
            #ndims = len(image.shape[1:])
            device = image.device
        else:
            vol_shape = torch.tensor(label.shape[1:], device=label.device)
            #ndims = len(label.shape[1:])
            device = label.device
        vol_ndims = len(vol_shape)

        if (not torch.any(vol_shape > self.crop_size)):
            return dict(image=image, label=label, prior=prior, crop_idx=None)

        crop_idx = None        
        zero_tensor = torch.zeros(vol_ndims, device=device, dtype=int)
        crop_half = (self.crop_size/2).int()
        center_point = (vol_shape/2).int()   #tuple(dim // 2 for dim in vol_shape)

        # off_center = U[-self.max_offset, self.max_offset) if sampling = True; otherwise off_center = self.max_offset
        if (self.max_offset is not None):
            if (not self.sampling):
                off_center = self.max_offset
            else:
                # U[-self.max_offset, +self.max_offset)
                off_center = [(-self.max_offset[i]-self.max_offset[i]) * torch.rand(1, device=device) + self.max_offset[i] for i in range(vol_ndims)]               
                off_center = [math.ceil(off_center[i]) for i in range(vol_ndims)]
            center_point = center_point - torch.tensor(off_center, device=device)

        # Calculate the starting and ending indices for the crop region
        start_coords = torch.maximum(zero_tensor, center_point - crop_half)
        end_coords = torch.minimum(center_point + crop_half, vol_shape)
        crop_idx = torch.concat((start_coords, end_coords))
        if (self.verbose):
            logging.debug(f"crop center: {center_point.tolist()}, crop indices: {crop_idx.tolist()}")

        if (vol_ndims == 3):
            output = {
                'image': image[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (image is not None) else None,
                'label': label[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (label is not None) else None,
                'prior': prior[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (prior is not None) else None,
                'crop_idx': crop_idx,
                     }
        else:
            output = {
                'image': image[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (image is not None) else None,
                'label': label[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (label is not None) else None,
                'prior': prior[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (prior is not None) else None,
                'crop_idx': crop_idx,
                     }

        return output


# This is described in Hypothalamus paper (https://www.sciencedirect.com/science/article/pii/S1053811920307734)
#   "
#     In order to make the network robust against acquisition procedures,
#     we add further global intensity augmentation by shifting the brightness and contrast of the image with randomly sampled values (Fig. 1(d)).
#     The obtained scan is subsequently flipped along the right-left axis with a probability of 0.5 (Fig. 1(e)), and randomly cropped to a size of 160^3,
#     which is more than large enough to ensure that the hypothalamus is always present in the resulting scan.
#     Finally, intensities are rescaled between [0,1] with min-max normalisation.
#     Additional examples of augmented images are shown in the Supplementary materials (Fig. S1).
#   "
class IntensityAugmentation(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp
        self.noise_std = hp.get("added_noise_max_sigma", 1.0)  # default is 0 for SynthSeg, no white noise added
        self.normalize = hp.get("normalize", True)
        self.gamma_std = hp.get("gamma_scaling_max", 0.5)
        self.prob_noise = hp.get("added_noise_probability", 0.95)
        self.prob_gamma = hp.get("gamma_scaling_probability", 1)
        self.sampling = sampling_hp
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
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

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.IntensityAugmentation'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        num_channels = image.shape[0]
        ndims = image.ndim - 1

        # noise and gamma are sampled and applied independently for each channel of the input tensor
        sample_shape = [num_channels] + [1] * ndims # [C, 1, 1 (,1)]
    
        # add noise with predefined probability
        if (self.noise_std > 0 and np.random.rand() < self.prob_noise):
            # noise_stddev = U(0, noise_std) if sampling = True; otherwise noise_stddev = noise_std
            noise_stddev = self.noise_std * torch.rand(sample_shape, device=image.device) if (self.sampling) else torch.full(sample_shape, self.noise_std, device=image.device)
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

        output = {
            'image': image,
            'label': label,
            'prior': prior,
                 }
        return output


# This is described in Hypothalamus paper (https://www.sciencedirect.com/science/article/pii/S1053811920307734)
#  "
#    The augmentation model also accounts for non-uniformities in the magnetic field commonly observed in MR scanners (Simmons et al., 1994).
#    Because this phenomenon translates into intensity inhomogeneities smoothly varying across MRI scans (Sled and Pike, 1998),
#    we model it with a multiplicative smooth field. As before, we sample a small low resolution field (e.g., of size 4 × 4 × 4),
#    and upscale it to image size with linear interpolation. Then, we take the voxel-wise exponential to ensure the non-negativity of this field.
#    Finally, we multiply the spatially deformed scan by the obtained bias field to corrupt its intensities (Fig. 1(c)).
#  "
class BiasFieldCorruption(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.bias_field_std = hp.get("bias_field_max_magnitude", .7)  # SynthSeg
        self.bias_scale = hp.get("bias_field_scale", .025)
        self.prob = hp.get("bias_field_probability", 0.95)
        self.sampling = sampling_hp
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
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

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        if (self.sampling and (not np.random.rand() < self.prob or self.bias_field_std <= 0)):
            if (self.verbose):
                logging.debug(f"'freeseg.augmentation.augmentbase.BiasFieldCorruption' - Skipped prob={self.prob}, bias_field_std={self.bias_field_std}")
            return dict(image=image, label=label, prior=prior, crop_idx=None)
    
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.BiasFieldCorruption'")

        num_channels = image.shape[0]
        ndims = image.ndim - 1
        image_shape = image.shape[1:]
    
        # sampling shapes, the bias field will be sampled and applied independently for each channel of the input tensor
        std_shape = [num_channels] + [1] * ndims   # [C, 1, 1(, 1)]
        small_bias_shape = [num_channels] + [math.ceil(image_shape[i] * self.bias_scale) for i in range(len(image_shape))]  # [C, h, w, (,d)]

        # sample small bias field (step 1 and 2)
        # stddev = U(0, bias_field_std) if sampling = True; otherwise stddev = bias_field_std
        stddev = self.bias_field_std * torch.rand(std_shape, device=image.device) if (self.sampling) else torch.full(std_shape, self.bias_field_std, device=image.device)
        bias_field_tensor = stddev * torch.randn(small_bias_shape, device=image.device)   # N(0, stddev)
    
        # resize bias field and take exponential (step 3 and 4)
        mode = "trilinear" if (ndims == 3) else "bilinear"
        bias_field_tensor = torch.nn.functional.interpolate(bias_field_tensor.unsqueeze(0), image_shape, mode=mode)
        bias_field_tensor = bias_field_tensor.squeeze(0)  # remove the dummy batch dimension
        bias_field_tensor = torch.exp(bias_field_tensor)

        # element-wise multiplication (step 5)
        bf_augmented_image = torch.mul(bias_field_tensor, image)

        output = {
            'image': bf_augmented_image,
            'label': label,
            'prior': prior,
                 }
        return output


# generate an initial synthetic scan G by sampling a GMM conditioned on L described in SynthSeg paper
# (https://www.sciencedirect.com/science/article/pii/S1361841523000506)
class SampleConditionalGMM(torch.nn.Module):
    def __init__(self, generation_labels, generation_classes, hp=None, num_channels=1, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.generation_labels  = generation_labels
        self.generation_classes = generation_classes,
        self.num_channels = num_channels  # dataset expected_num_channels
        self.prior_distribution = hp.get("prior_distribution", "uniform")  # 'normal'
        self.prior_mean = hp.get("prior_mean", [25, 225])
        self.prior_std = hp.get("prior_std", [5, 25])
        self.verbose = verbose

    def forward(self, input, debugsaveprefix=None):
        """
        Generate a synthetic image (num_channels) by sampling a Gaussian Mixture Model conditioned on a label map given as input.
        Each channel is sampled independently.

        GMM-sampling parameters:
          prior_distribution: type of distribution from which we sample the GMM parameters {'uniform', 'normal'}
          prior_mean: hyperparameters controlling the means of Gaussian distributions of the GMM
          prior_std:  hyperparameters controlling the standard deviations of Gaussian distributions of the GMM
 
        label: input tensors expected to have shape [1, H, W (,D)]
        sampled_image: output tensor [num_channels, H, W (,D)]
        """    

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.SampleConditionalGMM'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        assert (label is not None), 'label is needed for sampleConditionalGMM'
        assert (self.generation_labels is not None), 'generation_labels is needed for sampleConditionalGMM'
        assert (self.generation_classes is not None), 'generation_classes is needed for sampleConditionalGMM'        
        
        # sample means and stds of Gaussian distributions of the GMM
        n_classes = len(np.unique(self.generation_classes))
        prior_shape = n_classes
        if self.prior_distribution == 'uniform':
            means = np.random.uniform(low=self.prior_mean[0], high=self.prior_mean[1], size=prior_shape)
            stds  = np.random.uniform(low=self.prior_std[0], high=self.prior_std[1], size=prior_shape)
        elif self.prior_distribution == 'normal':
            means = np.random.normal(loc=self.prior_mean[0], scale=self.prior_mean[1], size=prior_shape)
            stds  = np.random.normal(loc=self.prior_std[0], scale=self.prior_std[1], size=prior_shape)
        else:
            raise ValueError("Prior distribution not supported, should be 'uniform' or 'normal'.")

        # reset all negative values to zero
        means[means < 0] = 0
        stds[stds < 0] = 0

        # the following is taken from SynthSeg.model_inputs.build_model_inputs()
        # https://github.com/BBillot/SynthSeg/blob/master/SynthSeg/model_inputs.py#L142C1-L149C1
        random_coef = npr.uniform()
        if random_coef > 0.95:   # reset the background to 0 in 5% of cases
            means[0] = 0
            stds[0] = 0
        elif random_coef > 0.7:  # reset the background to low Gaussian in 30% of cases
            means[0] = npr.uniform(0, 15)
            stds[0] = npr.uniform(0, 5)

        # get gaussian mean/std for each generation label
        means = means[self.generation_classes]
        stds  = stds[self.generation_classes]
        
        # generate synthetic image
        label = label.squeeze(0)   # remove the channel axis
        sampled_image = torch.zeros((self.num_channels, *label.shape), device=label.device)
        for idx_label in range(len(self.generation_labels)):
            label_indices = (label == self.generation_labels[idx_label])
            indices_count = label_indices.sum()

            # each channel is sampled independently
            for n_channel in range(self.num_channels):
                # N(means[labelid] + stds[labelid])
                gauss_samples = means[idx_label] + stds[idx_label] * torch.randn(indices_count, device=label.device)
                sampled_image[n_channel][label_indices] = gauss_samples
 
        output = {
            'image': sampled_image,
            'label': label.unsqueeze(0), # add back the channel axis
            'prior': prior,
                 }
        return output


class RescaleVolume(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.new_min = hp.get("new_min", 0.0)
        self.new_max = hp.get("new_max", 1.0)
        self.min_percentile = hp.get("min_percentile", 0.5)
        self.max_percentile = hp.get("max_percentile", 99.5)
        self.use_positive_only = hp.get("use_positive_only", False)
        self.sampling = sampling_hp
        self.verbose = verbose


    def forward(self, input, debugsaveprefix=None):
        """
        Applies intensity rescaling to the image volume. All channels are scaled separately.
        """
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.RescaleVolume'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)
        
        if (self.use_positive_only):
            image = image[image > 0]
            
        ndims = image.ndim - 1
        axis = tuple(dim for dim in range(1, ndims+1)) # axis=(H, W (,D))

        # m is reduced to [C, 1, 1 (,1)]
        if (self.min_percentile == 0):
            m = torch.amin(image, dim=axis, keepdim=True)
        else:
            q = torch.tensor(self.min_percentile/100).to(self.device)
            m = image
            for dim in (axis):
                m = torch.quantile(m, q, dim=dim, keepdim=True, interpolation='linear')

        # M is reduced to [C, 1, 1 (,1)]
        if (self.max_percentile == 100):
            M = torch.amax(image, dim=axis, keepdim=True)
        else:
            q = torch.tensor(self.max_percentile/100).to(self.device)
            M = image
            for dim in (axis):
                M = torch.quantile(M, q, dim=dim, keepdim=True, interpolation='linear')

        # normalize
        image = torch.clip(image, min=m, max=M)
        image = self.new_min + (image - m) / (M - m + torch.finfo(torch.float32).eps) * (self.new_max - self.new_min)
        
        output = {
            'image': image,
            'label': label,
            'prior': prior,
                 }
        return output


class GaussianBlur(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp        
        self.max_sigma = hp.get("gaussian_blur_max_sigma", 2)
        self.truncate = hp.get("gaussian_blur_truncate", 2.5)
        self.radius = hp.get("gaussian_blur_radius", None)
        self.sampling = sampling_hp
        self.verbose = verbose


    def forward(self, input, debugsaveprefix=None):
        """
        Applies gaussian smoothing to the image volume.

        image: torch.tensor
          non-batched tensor [C, H, W (,D)] 
        """
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.GaussianBlur'")

        assert (self.max_sigma is not None), \
            f"freeseg.augmentation.augmentbase.GaussianBlur(): need to specify gaussian_blur_sigma"

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)

        ndims = image.ndim - 1
        in_channels = image.shape[0]
        conv = getattr(torch.nn.functional, f'conv{ndims}d')
        sigma = np.random.uniform(0, self.max_sigma) if (self.sampling) else self.max_sigma
        if (np.isscalar(sigma)):
            sigma = [sigma] * ndims

        """
          gaussian_filter needs to have shape [out_channels, nfilters, H, W (,D)]

          padding='same' pads the input so the output has the shape as the input.
          this mode doesn’t support any stride values other than 1.
        
          each group will be convolved separately,
          the output is the concatenation of all the groups results along the channel axis
        """
        groups = in_channels
        out_channels = in_channels
        nfilters = int(out_channels/groups)
        gaussian_filter = Filter.gaussian_kernel(sigma, self.truncate, self.radius, self.device)
        gaussian_filter = gaussian_filter[None, None, :]  # add out_channels and nfilters dimension
        # repeat for each output channels
        gaussian_filter = gaussian_filter.repeat(out_channels, nfilters, *([1] * ndims))

        image_blurred = conv(image.unsqueeze(0), gaussian_filter, groups=groups, padding="same")

        # remove batch dimension before returning
        output = {
            'image': image_blurred.squeeze(0),
            'label': label,
            'prior': prior,
                 }
        return output


class ResampleVolume(torch.nn.Module):
    def __init__(self, target_res, hp=None, device=None, sampling_hp=True, verbose=False):
        """
        Resamples the volume to the given voxel size space
        """
        
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp
        self.target_res = target_res
        self.sampling = sampling_hp
        self.verbose = verbose


    def forward(self, input, debugsaveprefix=None):
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.ResampleVolume'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)

        ndims = image.ndim - 1
        image_shape = image.shape[1:]

        if (self.target_res is not None and np.isscalar(self.target_res)):        
            self.target_res = np.array([self.target_res] * ndims)

        if ((self.target_res is None) or \
            (not np.any((voxsize > self.target_res+0.05) | (voxsize < self.target_res-0.05)))):
            output = {
                'image': image,
                'label': label,
                'prior': prior,
                'geom': geom,
                     }
            return output
    
        factor = voxsize / self.target_res
        start = - (factor - 1) / (2 * factor)
        step = 1.0 / factor
        stop = start + step * np.ceil(image_shape * factor)

        xyzs = [torch.arange(start=start[d], end=stop[d], step=step[d], dtype=torch.float32, device=self.device) for d in range(ndims)]
        x, y, z = torch.meshgrid(*xyzs, indexing='ij')
        meshgrid = torch.stack((x, y, z), dim=-1)
                
        # scale meshgrid to range [-1, 1], which is expected by torch.nn.functional.grid_sample()
        for d in range(ndims):
           if image_shape[d] == 1:
               meshgrid[..., d] *= 0
           else:
               meshgrid[..., d] *= 2 / (image_shape[d] - 1)
               meshgrid[..., d] -= 1

        meshgrid = meshgrid.flip(-1)

        resampled_image = torch.nn.functional.grid_sample(image.unsqueeze(0), meshgrid.unsqueeze(0),
                                                          mode="bilinear", padding_mode='zeros', align_corners=True)
        if (label is not None):
            resampled_label = torch.nn.functional.grid_sample(label.float().unsqueeze(0), meshgrid.unsqueeze(0),
                                                              mode="nearest", padding_mode='zeros', align_corners=True)

        from surfa.transform import ImageGeometry
        new_geom = ImageGeometry(
            shape=resampled_image.shape[2:],
            voxsize=self.target_res,
            rotation=geom.rotation,
            center=geom.center)

        output = {
            'image': resampled_image.squeeze(0),
            'label': resampled_label.squeeze(0).int() if (label is not None) else label,
            'prior': prior,
            'geom':  new_geom,            
                 }
        return output


class MimicResolution(torch.nn.Module):
    def __init__(self, hp=None, device=None, sampling_hp=True, verbose=False):
        """
        Takes an image as input, and simulates data that has been acquired at low resolution.
        The output is obtained by resampling the input twice:
        - first at a resolution given as an input (i.e. the "acquisition" resolution),
        - then at the output resolution (specified output shape).
        """
        
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp
        self.max_subsample_res = hp.get("max_subsample_res", 0.0)
        self.sampling = sampling_hp
        self.verbose = verbose


    def forward(self, input, debugsaveprefix=None):
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.MimicResolution'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)

        ndims = image.ndim - 1
        image_shape = image.shape[1:]

        # sample the random resolution lower resolution from U(voxsize, max_subsample_res)
        subsample_res = np.random.uniform(voxsize, self.max_subsample_res) if (self.sampling) else [self.max_subsample_res] * ndims
        factor = tuple(voxsize / subsample_res)

        mode = "trilinear" if (ndims == 3) else "bilinear"
        
        # ??? todo: perform gaussian blur ???
        # ??? ... ???

        # downsample the image to subsample_res
        resampled_image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            scale_factor=factor,
            mode=mode,
            align_corners=True)
        
        # upsample it back to original res
        resampled_image = torch.nn.functional.interpolate(
            resampled_image,
            size=image_shape,
            mode=mode,
            align_corners=True)

        output = {
            'image': resampled_image.squeeze(0),
            'label': label,
            'prior': prior,
                 }
        return output


class RemapLabels(torch.nn.Module):
    def __init__(self, source_labels, dest_labels=None, hp=None, device=None, sampling_hp=True, verbose=False):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        hp = {} if (hp is None) else hp            
        self.source_labels  = source_labels
        self.dest_labels = dest_labels
        self.sampling = sampling_hp
        self.verbose = verbose
        if (self.dest_labels is None):
            self.source_labels = np.unique(self.source_labels)
            self.dest_labels = np.arange(len(self.source_labels), dtype='int32')

        self.mapping = None
        if (len(self.source_labels) == len(self.dest_labels) and np.all(self.source_labels == self.dest_labels)):
            logging.info(f"'freeseg.augmentation.augmentbase.RemapLabels': source_labels same as dest_labels")
            return

        # build the mapping from source labels to dest labels        
        self.mapping = {}        
        for src, dest in zip(self.source_labels, self.dest_labels):
            self.mapping.update({src:dest})
        

    def forward(self, input, debugsaveprefix=None):
        """
        Map source labels to dest labels
        """    

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.RemapLabels'")

        image = input.get("image", None)
        label = input.get("label", None)
        prior = input.get("prior", None)
        voxsize = input.get("voxsize", None)
        geom = input.get("geom", None)

        if (self.mapping is None):
            return dict(image=image, label=label, prior=prior)
        
        remapped_label = torch.zeros_like(label)
        for src, dest in self.mapping.items():
            remapped_label[label == src] = dest
 
        output = {
            'image': image,
            'label': remapped_label,
            'prior': prior,
                 }
        return output


def CropVolume(volume, crop_idx, verbose=False):
    """
    Crop volumes with given indices
    """    
    if (verbose):
        logging.debug(f"'freeseg.augmentation.augmentbase.Crop'")

    # input volume is non-batched tensor
    vol_shape = torch.tensor(volume.shape[1:], device=volume.device)
    device = volume.device
    vol_ndims = len(vol_shape)

    if (crop_idx is None):
        return volume

    if (vol_ndims == 3):
        volume = volume[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]]
    else:
        volume = volume[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]]

    return volume


def PadVolume(volume, padding_shape, padding_value=0):
    """
    Pad volume to a given shape

    volume: volume to be padded
    padding_shape: shape to pad volume to
    padding_value: (optional) value used for padding

    Returns:
        padded volume, pad_idx
    """

    new_volume = volume.clone()
    vol_shape = np.array(new_volume.shape[1:])
    ndims = len(vol_shape)

    # check if need to pad
    padding_shape = np.array(padding_shape)
    if (np.any(padding_shape > vol_shape)):
        # get padding margins
        min_margins = np.maximum(np.int32(np.floor((padding_shape - vol_shape)/2)), 0)
        max_margins = np.maximum(np.int32(np.ceil((padding_shape - vol_shape)/2)), 0)
        pad_idx = np.concatenate([min_margins, min_margins + vol_shape])

        # pad tuple specify pairs of padding for each dimension from the last to the first
        pad_margins = tuple(np.stack((np.flip(min_margins), np.flip(max_margins)), axis=1).flatten())

        # pad volume
        new_volume = torch.nn.functional.pad(volume, pad_margins, mode='constant', value=padding_value)
    else:
        pad_idx = np.concatenate([np.array([0] * ndims), vol_shape])

    return new_volume, pad_idx
