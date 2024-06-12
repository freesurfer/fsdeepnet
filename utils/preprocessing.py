import os
import numpy as np
import math
import torch
from voxynth import voxynth
from utils.data_utils import save_volume, get_ras_axes, bbox


def apply_flipping(image, label, aff, left_right_corresponding, flip_prob=0.5):
    """Applies a random left-right flip to image and label volumes."""
    """Swaps left-right labels on label volume."""
    if (np.random.rand() < flip_prob):
        assert aff is not None, 'aff should not be None when applying flipping'
        assert left_right_corresponding is not None, 'left_right_corresponding should not be None when applying flipping'

        ndims = len(image.shape[1:])
        
        # swap left-right labels
        n_left_right_labels = len(left_right_corresponding)
        left_right_corresponding = np.array(left_right_corresponding)
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
    else:
        # no flipping
        return image, label


def apply_spatial_transform(image, label, voxsize,
                            affine_probability=1.0,
                            max_translation=5.0,
                            max_rotation=5.0,
                            max_scaling=1.1,
                            warp_probability=1.0,
                            warp_integrations=5,
                            warp_smoothing_range=[10, 20],
                            warp_magnitude_range=[1, 2]):
    """Applies a random spatial transformation to image and label volumes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    label = label.to(device)

    trf = voxynth.transform.random_transform(
        shape=image.shape[1:],
        device=device,
        affine_probability=affine_probability,
        max_translation=max_translation,
        max_rotation=max_rotation,
        max_scaling=max_scaling,
        warp_probability=warp_probability,
        warp_integrations=warp_integrations,
        warp_smoothing_range=warp_smoothing_range,
        warp_magnitude_range=warp_magnitude_range,
    )

    transformed_image = voxynth.transform.spatial_transform(image, trf)
    transformed_label = voxynth.transform.spatial_transform(
        label, trf, method="nearest"
    )
    return transformed_image, transformed_label


def apply_randomcrop(image, label, crop_size, mode='random', bbox_labels=None, debug=False):
    """
    Randomly crop input tensors to a given shape. The input tensors are expected to have shape [batch H W D].

    Returns:
        cropped_image, cropped_label
        TODO: raise exception if there is no crops found that fit the bounding box of all labels
    """

    # assuming image and label have the same dimensions
    image_shape = image.shape[1:]
    image_ndims = len(image_shape)

    bbox_upper = np.array([0] * image_ndims)
    bbox_lower = np.array([image_shape[0]] * image_ndims)
    if (bbox_labels is not None):
        # calculate lower and upper bounds for the label bounding box
        bbox_lower, bbox_upper = bbox(label, bbox_labels)
        
        # make sure crop_size > (bbox_upper - bbox_lower)
        """
        # ??? TODO ???
        if (np.any(crop_size < (bbox_upper - bbox_lower))):
            raise exception
        """

    if (mode == 'random'):
        if (bbox_labels is None):
            crop_min_val = 0
            crop_max_val = np.array(image_shape) - np.array(crop_size)
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
            bound1 = np.array(image_shape) - np.array(crop_size)
            bound2 = np.array(crop_size)

            # minimum to crop so that it will include bbox_upper
            # the value depends on if bbox_upper > bound2
            crop_min_val = np.maximum(0,  (bbox_upper - bound2))
            # maximum to crop so that it will include bbox_lower
            crop_max_val = np.minimum(bbox_lower, bound1)
        
        start_coords = np.random.uniform(low=crop_min_val, high=crop_max_val).astype(int)
        end_coords   = start_coords + np.array(crop_size)
    elif (mode == 'center'):
        half_crop = np.array(crop_size)/2
        
        if (bbox_labels is None):
            start_center = half_crop
            end_center = np.array(image_shape) - half_crop
        else:
            """
            |                     |       |              |       |        |
            ---------------------------------------------------------------
            0                  center1  bound1        bound2  center2  image size

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
            start_center = half_crop.copy()
            end_center = np.array(image_shape) - half_crop

            bound1 = end_center - half_crop
            bound2 = start_center + half_crop
            if (np.any(bbox_lower < bound1)):
                # need to adjust end_center
                distance = bound1 - bbox_lower
                end_center -= np.maximum(0,  distance)    
            if (np.any(bbox_upper > bound2)):
                # need to adjust start center                
                distance = bbox_upper - bound2
                start_center += np.maximum(0,  distance)
            
        center_point = np.random.uniform(low=start_center, high=end_center).astype(int)
        start_coords = np.maximum(np.array(center_point)-half_crop, 0).astype(int)
        end_coords   = np.minimum(np.array(center_point)+half_crop, np.array(image_shape)).astype(int)
                
    if (debug):
        dbg_msg = f"apply_randomcrop({mode}) - image_shape: {np.array(image_shape)}, crop_size: {crop_size}, "
        if (bbox_labels is not None):
            dbg_msg += f"bbox: {bbox_lower} - {bbox_upper}, "
        if (mode == 'center'):
            dbg_msg += f"(start_center: {start_center}, end_center: {end_center}), center_point: {center_point}, "
        else:
            dbg_msg += f"(crop_min_val: {crop_min_val}, crop_max_val: {crop_max_val}), "
        dbg_msg += f"start_coords: {start_coords}, end_coords: {end_coords}"
        print(dbg_msg)
            
    # check if bbox_lower/bbox_upper are inside start_coords/end_coords
    if (np.any(bbox_lower < start_coords) or np.any(bbox_upper > end_coords)):
        dbg_msg = f"CROPPING ERROR apply_randomcrop({mode}) - image_shape: {np.array(image_shape)}, crop_size: {crop_size}, "
        if (bbox_labels is not None):
            dbg_msg += f"bbox: {bbox_lower} - {bbox_upper}, "
        if (mode == 'center'):
            dbg_msg += f"(start_center: {start_center}, end_center: {end_center}), center_point: {center_point}, "
        else:
            dbg_msg += f"(crop_min_val: {crop_min_val}, crop_max_val: {crop_max_val}), "            
        dbg_msg += f"start_coords: {start_coords}, end_coords: {end_coords}"
        print(dbg_msg)
        
        """
        # ??? TODO ???
        raise exception
        """
        
    slicing = [slice(None)] + [slice(start, end) for start, end in zip(start_coords, end_coords)]
    cropped_image = image[slicing]
    cropped_label = label[slicing]
        
    return cropped_image, cropped_label


def apply_cropping(image, label, crop_size=None):
    """Applies center cropping to image and label volumes."""
    cropped_image = voxynth.augment.apply_center_crop(image, crop_size)
    cropped_label = voxynth.augment.apply_center_crop(label, crop_size)
    return cropped_image, cropped_label


def apply_blur_resample(image, voxsize,
                        smoothing_probability=0.5,
                        smoothing_max_sigma=2.0,
                        added_noise_probability=0.5,
                        added_noise_max_sigma=0.05,
                        gamma_scaling_probability=0.5,
                        gamma_scaling_max=0.8,
                        resized_probability=0,
                        resized_one_axis_probability=0,
                        resized_max_voxsize=2):
    """Applies blurring and resampling to the image volume."""
    image_cpu = image.cpu()
    blur_resampled_image = voxynth.augment.image_augment(
        image_cpu,
        normalize=True,
        smoothing_probability=smoothing_probability,
        smoothing_max_sigma=smoothing_max_sigma,
        added_noise_probability=added_noise_probability,
        added_noise_max_sigma=added_noise_max_sigma,
        gamma_scaling_probability=gamma_scaling_probability,
        gamma_scaling_max=gamma_scaling_max,
        resized_probability=resized_probability,
        resized_one_axis_probability=resized_one_axis_probability,
        resized_max_voxsize=resized_max_voxsize,
    )
    return blur_resampled_image


def apply_bias_field(image, voxsize,
                     bias_field_probability=0.5,
                     bias_field_max_magnitude=0.1,
                     bias_field_smoothing_range=[1, 2]):
    """Applies bias field augmentation to the image volume."""
    bf_augmented_image = voxynth.augment.image_augment(
        image,
        voxsize=voxsize,
        bias_field_probability=bias_field_probability,
        bias_field_max_magnitude=bias_field_max_magnitude,
        bias_field_smoothing_range=bias_field_smoothing_range,
    )
    return bf_augmented_image


def apply_augmentations(
    image_tensor,
    label_tensor,
    original_image,
    original_label,
    augment_para,
    voxsize,
    output_dir=None,
    save_volumes=None,
    augmentations_to_apply=None,
    left_right_corresponding=None
):
    """
    Apply data augmentations to the image and label tensors and optionally save intermediate results.

    Args:
        image_tensor (torch.Tensor): PyTorch tensor representing the image volume.
        label_tensor (torch.Tensor): PyTorch tensor representing the label volume.
        original_image (surfa.Volume): Original loaded image volume.
        original_label (surfa.Volume): Original loaded label volume.
        augment_para (dict): preprocessing (augmentation) parameters
        voxsize (tuple): Voxel size of the volumes.
        output_dir (str, optional): Directory to save the intermediate results. If None, volumes are not saved.
        save_volumes (str, optional): prefix for augmented volumes. Defaults to None.
        augmentations_to_apply (list, optional): A list of augmentation names to apply.
                                                If None, all available augmentations are applied.
        left_right_corresponding (list, optional): left-right label correspondings. This is needed for label flipping.

    Returns:
        tuple: Augmented image and label tensors.
    """

    if augmentations_to_apply is None:
        augmentations_to_apply = [
            "flipping",
            "spatial_transform",
            "cropping",
            "blur_resample",
            "bias_field",
        ]

    crop_size = augment_para.get("crop_size", None)
            
    if "flipping" in augmentations_to_apply:
        flip_prob = augment_para.get("flip_prob")
        aff = original_image.geom.vox2world.matrix

        image_tensor, label_tensor = apply_flipping(image_tensor, label_tensor, aff, left_right_corresponding, flip_prob)
        if save_volumes is not None and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, save_volumes + "_flipped_image.mgz"),
            )
            save_volume(
                label_tensor,
                original_label,
                os.path.join(output_dir, save_volumes + "_flipped_label.mgz"),
            )

    # ??? spatial_transform always happens for hypothalamus
    if "spatial_transform" in augmentations_to_apply:
        image_tensor, label_tensor = apply_spatial_transform(
            image_tensor, label_tensor, voxsize,
            affine_probability=augment_para.get("affine_probability", 1.0),
            max_translation=augment_para.get("max_translation", 5.0),
            max_rotation=augment_para.get("max_rotation", 5.0),
            max_scaling=augment_para.get("max_scaling", 1.1),
            warp_probability=augment_para.get("warp_probability", 1.0),
            warp_integrations=augment_para.get("warp_integrations", 5),
            warp_smoothing_range=augment_para.get("warp_smoothing_range", [10, 20]),
            warp_magnitude_range=augment_para.get("warp_magnitude_range", [1, 2])
        )
        if save_volumes is not None and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, save_volumes + "_transformed_image.mgz"),
            )
            save_volume(
                label_tensor,
                original_label,
                os.path.join(output_dir, save_volumes + "_transformed_label.mgz"),
            )

    # ??? we are now supporting cropping, randomcrop, randomcrop_center. check to allow only one type of cropping ???
    if "cropping" in augmentations_to_apply:
        if crop_size is not None:
            image_tensor, label_tensor = apply_cropping(image_tensor, label_tensor, crop_size)
            if save_volumes is not None and output_dir is not None:
                save_volume(
                    image_tensor,
                    original_image,
                    os.path.join(output_dir, save_volumes + "_cropped_image.mgz"),
                )
                save_volume(
                    label_tensor,
                    original_label,
                    os.path.join(output_dir, save_volumes + "_cropped_label.mgz"),
                )
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")

    if "randomcrop" in augmentations_to_apply:
        if crop_size is not None:
            bbox_labels = augment_para.get("bbox_labels", None)
            debug = True if augment_para.get("debug") else False
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='random', bbox_labels=bbox_labels, debug=debug)
            if save_volumes is not None and output_dir is not None:
                save_volume(
                    image_tensor,
                    original_image,
                    os.path.join(output_dir, save_volumes + "_randomcropped_image.mgz"),
                )
                save_volume(
                    label_tensor,
                    original_label,
                    os.path.join(output_dir, save_volumes + "_randomcropped_label.mgz"),
                )
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")

    if "randomcrop_center" in augmentations_to_apply:
        if crop_size is not None:
            bbox_labels = augment_para.get("bbox_labels", None)
            debug = True if augment_para.get("debug") else False
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='center', bbox_labels=bbox_labels, debug=debug)
            if save_volumes is not None and output_dir is not None:
                save_volume(
                    image_tensor,
                    original_image,
                    os.path.join(output_dir, save_volumes + "_randomcropped_center_image.mgz"),
                )
                save_volume(
                    label_tensor,
                    original_label,
                    os.path.join(output_dir, save_volumes + "_randomcropped_center_label.mgz"),
                )
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")
        
    if "blur_resample" in augmentations_to_apply:
        image_tensor = apply_blur_resample(
            image_tensor, voxsize,
            smoothing_probability=augment_para.get("smoothing_probability", 0.5),
            smoothing_max_sigma=augment_para.get("smoothing_max_sigma", 2.0),
            added_noise_probability=augment_para.get("added_noise_probability", 0.5),
            added_noise_max_sigma=augment_para.get("added_noise_max_sigma", 0.05),
            gamma_scaling_probability=augment_para.get("gamma_scaling_probability", 0.5),
            gamma_scaling_max=augment_para.get("gamma_scaling_max", 0.8),
            resized_probability=augment_para.get("resized_probability", 0),
            resized_one_axis_probability=augment_para.get("resized_one_axis_probability", 0),
            resized_max_voxsize=augment_para.get("resized_max_voxsize", 2)
        )
        if save_volumes is not None and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, save_volumes + "_blur_resampled_image.mgz"),
            )

    if "bias_field" in augmentations_to_apply:
        image_tensor = apply_bias_field(
            image_tensor, voxsize,
            bias_field_probability=augment_para.get("bias_field_probability", 0.5),
            bias_field_max_magnitude=augment_para.get("bias_field_max_magnitude", 0.1),
            bias_field_smoothing_range=augment_para.get("bias_field_smoothing_range", [1, 2])
        )
        if save_volumes is not None and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, save_volumes + "_bias_field_augmented_image.mgz"),
            )

    # ??? global intensity augmentation ???

    return image_tensor, label_tensor
