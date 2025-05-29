import os
import logging
import numpy as np
import numpy.random as npr
import math
import torch
import torch.nn as nn
from freeseg import voxynth
from freeseg.utils import save_framedimage, get_ras_axes, bbox, centroid

class AugmentBase:
    def __init__(self, hyperparameters,
                 left_right_corresponding=None,
                 generation_labels=None,
                 output_dir=None,                 
                 device=None):
        self.valid_augmentations_base = ["flip",
                                         "spatialdeformation",
                                         "randomcrop", "randomcentercrop", "centercrop", "centroidcrop",
                                         "biasfieldcorruption",
                                         "intensityaugmentation",
                                         "sampleconditionalgmm",
                                         "rescalevolume"]
        self.valid_augmentations = self.valid_augmentations_base.copy()

        self.hyperparameters = hyperparameters        
        self.left_right_corresponding = left_right_corresponding
        self.generation_labels = generation_labels
        self.output_dir = output_dir
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = True if self.hyperparameters.get("verbose") else False
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.AugmentBase' constructor")

        # set up augmentations
        self.flip = Flip(self.hyperparameters, device=self.device, left_right_corresponding=self.left_right_corresponding)
        self.spatialdeformation = SpatialDeformation(self.hyperparameters, device=self.device)
        self.randomcrop = RandomCrop(self.hyperparameters, device=self.device, mode='random')
        self.randomcentercrop = RandomCrop(self.hyperparameters, device=self.device, mode='center')
        self.centercrop = CenterCrop(self.hyperparameters, device=self.device)
        self.centroidcrop = CentroidCrop(self.hyperparameters, device=self.device)
        self.biasfieldcorruption = BiasFieldCorruption(self.hyperparameters, device=self.device)
        self.intensityaugmentation = IntensityAugmentation(self.hyperparameters, device=self.device)
        self.sampleconditionalgmm = SampleConditionalGMM(self.hyperparameters, self.generation_labels, device=self.device)
        self.rescalevolume = RescaleVolume(self.hyperparameters, device=self.device)


    def check_augmentations(self, augmentations_to_apply):
        """
        check if all requested augmentations are valid and any duplicated augmentations
        """

        for augmentation in (augmentations_to_apply):
            assert (augmentation in self.valid_augmentations), \
                f"Unknown augmentation '{augmentation}'. Supported augmentations {self.valid_augmentations}. "

        if ("flip" in augmentations_to_apply):
            assert self.left_right_corresponding is not None, "left_right_corresponding is required for augmentation 'flip'"
        if ("sampleConditionalgmm" in augmentations_to_apply):
            assert (self.generation_labels is not None), "generation_labels is required for augmentation 'sampleConditionalGMM'"
        if (("centroidcrop" in augmentations_to_apply) and ("centercrop" in augmentations_to_apply)):
            raise ValueError("Both 'centroidcrop' and 'centercrop' are selected. Choose one.")        
        if (("centroidcrop" in augmentations_to_apply) and ("randomcrop" in augmentations_to_apply)):
            raise ValueError("Both 'centroidcrop' and 'randomcrop' are selected. Choose one.")
        if (("centroidcrop" in augmentations_to_apply) and ("randomcentercrop" in augmentations_to_apply)):
            raise ValueError("Both 'centroidcrop' and 'randomcentercrop' are selected. Choose one.")
        if (("centercrop" in augmentations_to_apply) and ("randomcrop" in augmentations_to_apply)):
            raise ValueError("Both 'centercrop' and 'randomcrop' are selected. Choose one.")
        if (("centercrop" in augmentations_to_apply) and ("randomcentercrop" in augmentations_to_apply)):
            raise ValueError("Both 'centercrop' and 'randomcentercrop' are selected. Choose one.")
        if (("randomcrop" in augmentations_to_apply) and ("randomcentercrop" in augmentations_to_apply)):
            raise ValueError("Both 'randomcrop' and 'randomcentercrop' are selected. Choose one.")


    def apply_augmentations(self,
                            image_tensor,
                            label_tensor,
                            original_image,
                            original_label,
                            voxsize,
                            priors_tensor=None,
                            save_volumes=None,
                            augmentations_to_apply=None):
        debugsaveprefix = None
        if (save_volumes is not None and self.output_dir is not None):
            debugsaveprefix = os.path.join(self.output_dir, save_volumes)

        if (debugsaveprefix is not None):
            save_framedimage(
                image_tensor,
                f"{debugsaveprefix}_reoriented_image.mgz",
                original_framedimage=original_image,            
            )
            np.save(f"{debugsaveprefix}_reoriented_image.npy", image_tensor.cpu().numpy().astype(np.float32))
            save_framedimage(
                label_tensor,
                f"{debugsaveprefix}_reoriented_label.mgz",
                original_framedimage=original_label,            
            )
            np.save(f"{debugsaveprefix}_reoriented_label.npy", label_tensor.cpu().numpy().astype(np.float32))
            if (priors_tensor is not None):
                save_framedimage(
                    priors_tensor,
                    f"{debugsaveprefix}_reoriented_prior.mgz",
                    original_framedimage=original_image,
                    dtype=float
                )
                np.save(f"{debugsaveprefix}_reoriented_prior.npy", priors_tensor.cpu().numpy().astype(np.float32))

        for idx, augment_name in enumerate(augmentations_to_apply):
            augment = getattr(self, augment_name, None)
            if (augment is None):
                logging.warning(f"augmentation '{augment_name}' not support, skip")
                continue

            image_tensor, label_tensor, priors_tensor, _ = augment(image=image_tensor, label=label_tensor, prior=priors_tensor, voxsize=voxsize, geom=original_image.geom, debugsaveprefix=debugsaveprefix)

            # save augmented volumes
            if (debugsaveprefix is not None):
                save_framedimage(
                    image_tensor,
                    f"{debugsaveprefix}_{augment_name}_{idx}_image.mgz",
                    original_framedimage=original_image,            
                )
                np.save(f"{debugsaveprefix}_{augment_name}_{idx}_image.npy", image_tensor.cpu().numpy().astype(np.float32))
                save_framedimage(
                    label_tensor,
                    f"{debugsaveprefix}_{augment_name}_{idx}_label.mgz",
                    original_framedimage=original_label,            
                )
                np.save(f"{debugsaveprefix}_{augment_name}_{idx}_label.npy", label_tensor.cpu().numpy().astype(np.float32))
                if (priors_tensor is not None):
                    save_framedimage(
                        priors_tensor,
                        f"{debugsaveprefix}_{augment_name}_{idx}_prior.mgz",
                        original_framedimage=original_image,
                        dtype=float
                    )
                    np.save(f"{debugsaveprefix}_{augment_name}_{idx}_prior.npy", priors_tensor.cpu().numpy().astype(np.float32))

        return image_tensor, label_tensor, priors_tensor


class Flip(nn.Module):
    def __init__(self, hyperparameters, device=None, left_right_corresponding=None):
        super().__init__()

        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.left_right_corresponding = left_right_corresponding
        self.flip_prob = hyperparameters.get("flip_prob", 0.5)
        self.verbose = True if hyperparameters.get("verbose") else False

    # ??? todo: flip priors ???        
    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
        """Applies a random left-right flip to image and label volumes."""
        """Swaps left-right labels on label volume."""
        if (np.random.rand() >= self.flip_prob):
            # no flipping
            return image, label

        assert geom is not None, 'geom should not be None when applying flipping'
        assert self.left_right_corresponding is not None, 'left_right_corresponding should not be None when applying flipping'
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.Flip'")

        aff = geom.vox2world.matrix
        ndims = len(image.shape[1:])
        
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
        axis = get_ras_axes(aff, ndims)[0]

        # image, label have shape [B, R, A, S]
        flipped_image = image.flip([axis+1])
        flipped_label = label.flip([axis+1])
    
        return flipped_image, flipped_label


class SpatialDeformation(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()

        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.affine_probability = hyperparameters.get("affine_probability", 1.0)
        self.max_translation = hyperparameters.get("max_translation", 5.0)
        self.max_rotation = hyperparameters.get("max_rotation", 5.0)
        self.max_shearing = hyperparameters.get("max_shearing", 0.015)        
        self.max_scaling = hyperparameters.get("max_scaling", 1.1)
        self.warp_probability = hyperparameters.get("warp_probability", 1.0)
        self.warp_integrations = hyperparameters.get("warp_integrations", 7)
        self.warp_generation_method = hyperparameters.get("warp_generation_method", "upsample")
        self.warp_smoothing_range = hyperparameters.get("warp_smoothing_range", [10, 20])
        self.warp_magnitude_range = hyperparameters.get("warp_magnitude_range", [1, 2])
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False

    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
        """Applies a random spatial transformation to image and label volumes."""

        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.SpatialDeformation'")

        """
        trf and aff_matrix are the same transform
        aff_matrix is None if there is non-linear component in trf
        """
        # voxsize is default to 1        
        trf, aff_matrix = voxynth.transform.random_transform(
            shape=image.shape[1:],
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
                center[:3, -1] = -(np.asarray(image.shape[1:]) - 1)/2
                aff_matrix_cpu = np.linalg.inv(center) @ aff_matrix_cpu @ center

                affine = Affine(aff_matrix_cpu, source=geom, target=geom)
                affine.save(f"{debugsaveprefix}_vox2vox_trg2src.lta")

        transformed_image = voxynth.transform.spatial_transform(image, trf)
        transformed_label = voxynth.transform.spatial_transform(label, trf, method="nearest")

        transformed_priors = None
        if (prior is not None):
            transformed_priors = voxynth.transform.spatial_transform(prior, trf, method="nearest")

        return transformed_image, transformed_label, transformed_priors, None


class RandomCrop(nn.Module):
    def __init__(self, hyperparameters, device=None, mode='random'):
        super().__init__()
        self.mode = mode
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        crop_size = hyperparameters.get("crop_size", None)
        self.crop_size = torch.tensor(crop_size, device=self.device)
        self.bbox_labels = hyperparameters.get("bbox_labels", None)
        self.verbose = True if hyperparameters.get("verbose") else False

    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
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
        
        # assuming image and label have the same dimensions
        image_shape = torch.tensor(image.shape[1:], device=image.device)
        image_ndims = len(image_shape)

        #crop_size = torch.tensor(crop_size, device=image.device)
    
        bbox_upper = torch.zeros(image_ndims, device=image.device, dtype=int)
        bbox_lower = image_shape
        if (self.bbox_labels is not None):
            # calculate lower and upper bounds for the label bounding box
            bbox_lower, bbox_upper = bbox(label, self.bbox_labels, verbose=self.verbose)
            if (self.verbose):
                logging.debug(f"crop around label bounding box {bbox_lower} - {bbox_upper}")
        
            # make sure crop_size > (bbox_upper - bbox_lower)
            """
            # ??? TODO ???
            if (torch.any(crop_size < (bbox_upper - bbox_lower))):
                raise exception
            """

        zero_tensor = torch.zeros(image_ndims, device=image.device, dtype=int)
        if (self.mode == 'random'):
            if (self.bbox_labels is None):
                crop_min_val = zero_tensor
                crop_max_val = image_shape - self.crop_size
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
                bound1 = image_shape - self.crop_size
                bound2 = self.crop_size

                # minimum to crop so that it will include bbox_upper
                # the value depends on if bbox_upper > bound2
                crop_min_val = torch.maximum(zero_tensor,  (bbox_upper - bound2))
                # maximum to crop so that it will include bbox_lower
                crop_max_val = torch.minimum(bbox_lower, bound1)

            # U[crop_min_val, crop_max_val)            
            start_coords = ((crop_min_val - crop_max_val) * torch.rand(image_ndims, device=image.device) + crop_max_val).int()
            end_coords   = start_coords + self.crop_size
        elif (self.mode == 'center'):
            half_crop = (self.crop_size/2).int()
        
            if (self.bbox_labels is None):
                start_center = half_crop
                end_center = image_shape - half_crop
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
                end_center = image_shape - half_crop

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
            center_point = ((start_center - end_center) * torch.rand(image_ndims, device=image.device) + end_center).int()
            start_coords = torch.maximum(center_point-half_crop, zero_tensor)
            end_coords   = torch.minimum(center_point+half_crop, image_shape)
                
        # Calculate the crop indices
        crop_idx = torch.concat((start_coords, end_coords)).int()
        if (self.verbose):
            dbg_msg = f"randomcrop({self.mode}) - {image_shape.tolist()} => {self.crop_size.tolist()}, "
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
            dbg_msg = f"***CROPPING WARNING*** randomcrop({self.mode}) - {image_shape.tolist()} => {self.crop_size.tolist()}, "
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

        if (image_ndims == 3):
            return image[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]], \
                   label[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (label is not None) else None, \
                   prior[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (prior is not None) else None, \
                   crop_idx
        else:
            return image[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]], \
                   label[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (label is not None) else None, \
                   prior[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (prior is not None) else None, \
                   crop_idx
    

class CentroidCrop(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        crop_size = hyperparameters.get("crop_size", None)
        self.crop_size = torch.tensor(crop_size, device=self.device)
        self.verbose = True if hyperparameters.get("verbose") else False

    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
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
        
        # input image is non-batched tensor
        image_shape = torch.tensor(image.shape[1:], device=image.device)
        #crop_size = torch.tensor(crop_size, device=image.device)

        crop_idx = None
        if (not torch.any(image_shape > self.crop_size)):
            return image, label, prior, crop_idx

        center_point = None
        # calculate the center point to crop the image/label around    
        if (label is not None):
            center_point = centroid(label.squeeze(0), verbose=self.verbose)

        zero_tensor = torch.zeros(image.ndim-1, device=image.device, dtype=int)
        crop_half = (self.crop_size/2).int()
        if (center_point is None):
            center_point = (image_shape/2).int()   #tuple(dim // 2 for dim in image_shape)
        else:
            # adjust the calculated center so that croppred image will have crop_size
            if (torch.any(center_point < crop_half)):
                distance = crop_half - center_point
                center_point += torch.maximum(zero_tensor,  distance)    
            if (torch.any(center_point > (image_shape - crop_half))):
                distance = center_point - (image_shape - crop_half)
                center_point -= torch.maximum(zero_tensor,  distance)

        # Calculate the starting and ending indices for the crop region
        start_coords = torch.maximum(zero_tensor, center_point - crop_half)
        end_coords = torch.minimum(center_point + crop_half, image_shape)
        crop_idx = torch.concat((start_coords, end_coords))
        if (self.verbose):
            logging.debug(f"adjusted crop center: {center_point.tolist()}, crop indices: {crop_idx.tolist()}")

        ndims = len(image.shape[1:])
        if (ndims == 3):
            return image[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]], \
                   label[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (label is not None) else None, \
                   prior[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (prior is not None) else None, \
                   crop_idx
        else:
            return image[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]], \
                   label[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (label is not None) else None, \
                   prior[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (prior is not None) else None, \
                   crop_idx


class CenterCrop(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        crop_size = hyperparameters.get("crop_size", None)
        self.crop_size = torch.tensor(crop_size, device=self.device)
        self.verbose = True if hyperparameters.get("verbose") else False

    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
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
        
        # input image is non-batched tensor
        image_shape = torch.tensor(image.shape[1:], device=image.device)
        #crop_size = torch.tensor(crop_size, device=image.device)

        crop_idx = None
        if (not torch.any(image_shape > self.crop_size)):
            return image, label, prior, crop_idx

        zero_tensor = torch.zeros(image.ndim-1, device=image.device, dtype=int)
        crop_half = (self.crop_size/2).int()
        center_point = (image_shape/2).int()   #tuple(dim // 2 for dim in image_shape)

        # Calculate the starting and ending indices for the crop region
        start_coords = torch.maximum(zero_tensor, center_point - crop_half)
        end_coords = torch.minimum(center_point + crop_half, image_shape)
        crop_idx = torch.concat((start_coords, end_coords))
        if (self.verbose):
            logging.debug(f"crop center: {center_point.tolist()}, crop indices: {crop_idx.tolist()}")

        ndims = len(image.shape[1:])
        if (ndims == 3):
            return image[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]], \
                   label[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (label is not None) else None, \
                   prior[:, crop_idx[0]:crop_idx[3], crop_idx[1]:crop_idx[4], crop_idx[2]:crop_idx[5]] if (prior is not None) else None, \
                   crop_idx
        else:
            return image[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]], \
                   label[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (label is not None) else None, \
                   prior[:, crop_idx[0]:crop_idx[2], crop_idx[1]:crop_idx[3]] if (prior is not None) else None, \
                   crop_idx


class IntensityAugmentation(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        #self.smoothing_probability = hyperparameters.get("smoothing_probability", 0.5)
        #self.smoothing_max_sigma = hyperparameters.get("smoothing_max_sigma", 2.0)
        self.added_noise_probability = hyperparameters.get("added_noise_probability", 0.5)
        self.added_noise_max_sigma = hyperparameters.get("added_noise_max_sigma", 0.05)
        self.gamma_scaling_probability = hyperparameters.get("gamma_scaling_probability", 0.5)
        self.gamma_scaling_max = hyperparameters.get("gamma_scaling_max", 0.8)
        #self.resized_probability = hyperparameters.get("resized_probability", 0)
        #self.resized_one_axis_probability = hyperparameters.get("resized_one_axis_probability", 0)
        #self.resized_max_voxsize = hyperparameters.get("resized_max_voxsize", 2)
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False
            
    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
        """Applies blurring and resampling to the image volume."""
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.IntensityAugmentation'")
        
        blur_resampled_image = voxynth.augment.image_augment(
            image,
            added_noise_probability=self.added_noise_probability,
            added_noise_max_sigma=self.added_noise_max_sigma,
            gamma_scaling_probability=self.gamma_scaling_probability,
            gamma_scaling_max=self.gamma_scaling_max,
            sampling=self.sampling,
        )
        return blur_resampled_image, label, prior, None


class BiasFieldCorruption(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.bias_field_probability = hyperparameters.get("bias_field_probability", 0.5)
        self.bias_field_max_magnitude = hyperparameters.get("bias_field_max_magnitude", 0.1)
        self.bias_field_smoothing_range = hyperparameters.get("bias_field_smoothing_range", [1, 2])
        self.bias_field_generation_method = hyperparameters.get("bias_field_generation_method", "blur")
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False
            
        
    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
        """Applies bias field augmentation to the image volume."""
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.BiasFieldCorruption'")
        
        bf_augmented_image = voxynth.augment.image_augment(
            image,
            voxsize=voxsize,
            bias_field_probability=self.bias_field_probability,
            bias_field_max_magnitude=self.bias_field_max_magnitude,
            bias_field_smoothing_range=self.bias_field_smoothing_range,
            bias_field_generation_method=self.bias_field_generation_method,
            sampling=self.sampling,
        )
        return bf_augmented_image, label, prior, None


# generate an initial synthetic scan G by sampling a GMM conditioned on L described in SynthSeg paper
# (https://www.sciencedirect.com/science/article/pii/S1361841523000506)
class SampleConditionalGMM(nn.Module):
    def __init__(self, hyperparameters, generation_labels, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generation_labels = generation_labels
        self.num_channels = hyperparameters.get("num_channels", 1)  # dataset expected_num_channels
        self.prior_distribution = hyperparameters.get("prior_distribution", "uniform")  # 'normal'
        self.prior_mean = hyperparameters.get("prior_mean", [25, 225])
        self.prior_std = hyperparameters.get("prior_std", [5, 25])
        self.verbose = True if hyperparameters.get("verbose") else False

    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
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
        
        assert (self.generation_labels is not None), 'generation_labels is needed for sampleConditionalGMM'
        
        # sample means and stds of Gaussian distributions of the GMM
        num_classes = len(self.generation_labels)
        prior_shape = (self.num_channels, num_classes)
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

        # generate synthetic image
        label = label.squeeze(0)   # remove the channel axis
        sampled_image = torch.zeros((self.num_channels, *label.shape), device=label.device)
        for labelid in range(num_classes):
            label_indices = (label == self.generation_labels[labelid])
            indices_count = label_indices.sum()

            # each channel is sampled independently
            for n_channel in range(self.num_channels):
                gauss_samples = means[n_channel, labelid] + stds[n_channel, labelid] * torch.randn(indices_count, device=label.device)   # N(means[n_channel, labelid] + stds[n_channel, labelid])
                sampled_image[n_channel][label_indices] = gauss_samples
 
        return sampled_image, label, prior, None


class RescaleVolume(nn.Module):
    def __init__(self, hyperparameters, device=None):
        super().__init__()
        self.device = device
        if (self.device is None):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.new_min = hyperparameters.get("new_min", 0.0)
        self.new_max = hyperparameters.get("new_max", 1.0)
        self.min_percentile = hyperparameters.get("min_percentile", 0.5)
        self.max_percentile = hyperparameters.get("max_percentile", 99.5)
        self.use_positive_only = hyperparameters.get("use_positive_only", False)
        self.sampling = hyperparameters.get("sampling_hyperparameters", True)
        self.verbose = True if hyperparameters.get("verbose") else False


    def forward(self, image=None, label=None, prior=None, voxsize=None, geom=None, debugsaveprefix=None):
        """
        Applies intensity rescaling to the image volume. All channels are scales separately.
        """
        if (self.verbose):
            logging.debug(f"'freeseg.augmentation.augmentbase.RescaleVolume'")

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
        
        return image, label, prior, None
