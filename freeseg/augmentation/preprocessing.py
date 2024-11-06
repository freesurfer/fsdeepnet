import os
import numpy as np
import math
import torch
from freeseg import voxynth
from freeseg.utils import save_framedimage, get_ras_axes, bbox, centroid


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
                            warp_integrations=7,
                            warp_smoothing_range=[10, 20],
                            warp_magnitude_range=[1, 2],
                            shearing_bounds=0.015,
                            device=None):
    """Applies a random spatial transformation to image and label volumes."""
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #print(f"apply_spatial_transform() - image.get_device() = {image.get_device()}, label.get_device() = {label.get_device()}, device = {device}")

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
        shearing_bounds=shearing_bounds,
    )

    transformed_image = voxynth.transform.spatial_transform(image, trf)
    transformed_label = voxynth.transform.spatial_transform(label, trf, method="nearest")

    return transformed_image, transformed_label


def apply_randomcrop(image, label, crop_size, mode='random', bbox_labels=None, debug=False):
    """
    Randomly crop input tensors to a given shape. 
    The input tensors are expected to have shape [batch H W D].

    Returns:
        cropped_image, cropped_label
        TODO: raise exception if there is no crops found that fit the bounding box of all labels
              handle batch > 1 ???
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
            start_center = half_crop.copy()  # make a copy for start_center to be modified later
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


def apply_centercrop(image, crop_size, center_point=None, debug=False):
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

    # input image is non-batched tensor
    image_shape = image.shape[1:]

    crop_half = (np.array(crop_size)/2).astype(int)

    if (center_point is None):
        center_point = tuple(dim // 2 for dim in image_shape)
    else:
        # adjust the calculated center so that croppred image will have crop_size
        if (np.any(center_point < crop_half)):
            distance = crop_half - center_point
            center_point += np.maximum(0,  distance)    
        if (np.any(center_point > (image_shape - crop_half))):
            distance = center_point - (image_shape - crop_half)
            center_point -= np.maximum(0,  distance)

    # Calculate the starting and ending indices for the crop region
    start_coords = tuple(max(0, center - half) for center, half in zip(center_point, crop_half))
    end_coords = tuple(min(center + half, dim) for center, half, dim in zip(center_point, crop_half, image_shape))
    crop_idx = np.concatenate([np.array(start_coords), np.array(end_coords)])
    if (debug):
        print(f"adjusted crop center: {center_point}, crop indices: {crop_idx}")
 
    # Create slicing expression for efficient cropping
    slicing = [slice(None)] + [slice(start, end) for start, end in zip(start_coords, end_coords)]

    return image[slicing], crop_idx


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
    blur_resampled_image = voxynth.augment.image_augment(
        image,
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


def apply_sampleConditionalGMM(label_map, expected_classes, prior_mean=[25, 225], prior_std=[5, 25], distribution='uniform'):
    """
    The label_map tensors are expected to have shape [batch H W D].
    gaussian mean and std are sampled independently for each batch
    """    

    assert expected_classes is not None, 'expected_classes is needed for sampleConditionalGMM'

    batchsize = label_map.shape[0]
    n_classes = len(expected_classes)

    prior_shape = [batchsize, n_classes]
    if distribution == 'uniform':
        means = np.random.uniform(low=prior_mean[0], high=prior_mean[1], size=prior_shape)
        stds  = np.random.uniform(low=prior_std[0], high=prior_std[1], size=prior_shape)
    elif distribution == 'normal':
        means = np.random.normal(loc=prior_mean[0], scale=prior_mean[1], size=prior_shape)
        stds  = np.random.normal(loc=prior_std[0], scale=prior_std[1], size=prior_shape)
    else:
        raise ValueError("Distribution not supported, should be 'uniform' or 'normal'.")    

    sampled_image = np.zeros(label_map.shape)
    for labelid in range(n_classes):
        label_indices = (label_map == expected_classes[labelid])
        indices_count = label_indices.sum(axis=[1,2,3])
        gauss_samples = np.random.normal(loc=means[:, labelid], scale=stds[:, labelid], size=(batchsize, indices_count))
        for batch, indices in enumerate(label_indices):
            sampled_image[batch][indices] = gauss_samples[batch]
        
    # convert it to tensor        
    sampled_image_tensor = torch.from_numpy(sampled_image).to(label_map.device)
 
    return sampled_image_tensor


def apply_biasFieldCorruption(image, bias_field_std=.5, bias_scale=.025, prob=0.95):
    """
    The input tensors are expected to have shape [batch H W D].
    bias field is sampled independently for each batch. 
    """
    
    if (not np.random.rand() < prob):
        return image

    batchsize = image.shape[0]
    ndims = len(image.shape) - 1
    image_shape = image.shape[1:]
    
    # sampling shapes
    std_shape = [batchsize] + [1] * ndims
    small_bias_shape = [batchsize] + [math.ceil(image_shape[i] * bias_scale) for i in range(len(image_shape))]

    # sample small bias field
    stddev = np.random.uniform(low=0, high=bias_field_std, size=std_shape)
    bias_field = np.random.normal(loc=0, scale=stddev, size=small_bias_shape) 

    bias_field_tensor = torch.from_numpy(bias_field)
    bias_field_tensor = bias_field_tensor.to(image.device, dtype=image.dtype)
        
    # resize bias field and take exponential
    bias_field_tensor = torch.nn.functional.interpolate(bias_field_tensor.unsqueeze(1), image_shape, mode='trilinear')
    bias_field_tensor = bias_field_tensor.squeeze(1)  # remove the dummy channel dimension
    bias_field_tensor = torch.exp(bias_field_tensor)

    # elementwise multiplication
    bf_augmented_image = torch.mul(bias_field_tensor, image)

    return bf_augmented_image


def apply_intensityAugmentation(image, noise_std=0, clip_values=[0, 300], normalise=True, gamma_std=0, contrast_inversion=False,
                                prob_noise=0.95, prob_gamma=1):
    """
    The input tensors are expected to have shape [batch H W D].
    noise and gamma are sampled independently for each batch.
    """
    
    image_cpu = image.cpu().detach().numpy()
    
    batchsize = image_cpu.shape[0]
    ndims = len(image_cpu.shape) - 1

    sample_shape = None
    if (noise_std > 0 or gamma_std > 0 or contrast_inversion):
        sample_shape = [batchsize] + [1] * ndims
    
    # add noise with predefined probability
    if (noise_std > 0 and np.random.rand() < prob_noise):
        noise_stddev = np.random.uniform(low=0, high=noise_std, size=sample_shape)
        noise = np.random.normal(loc=0, scale=noise_stddev, size=image_cpu.shape)
        image_cpu += noise

    # clip image_cpus to given values
    if (clip_values is not None):
        image_cpu = np.clip(image_cpu, a_min=clip_values[0], a_max=clip_values[1])

    # normalise
    if (normalise):
        # simple min and max
        axis = tuple(dim for dim in range(image_cpu.ndim))
        m = np.min(image_cpu, axis=axis)
        M = np.max(image_cpu, axis=axis)
        # normalise
        image_cpu = np.clip(image_cpu, a_min=m, a_max=M)
        image_cpu = (image_cpu - m) / (M - m + np.finfo(float).eps)

    # apply voxel-wise exponentiation with predefined probability
    if (gamma_std > 0 and np.random.rand() < prob_gamma):
        gamma = np.random.normal(loc=0, scale=gamma_std, size=sample_shape)
        image_cpu = np.power(image_cpu, np.exp(gamma))

    # apply random contrast inversion
    if (contrast_inversion):
        image_cpu = 1 - image_cpu

    return torch.from_numpy(image_cpu).to(image.device)


def apply_augmentations(
    image_tensor,
    label_tensor,
    original_image,
    original_label,
    expected_classes,
    augment_para,
    voxsize,
    output_dir=None,
    save_volumes=None,
    augmentations_to_apply=None,
    left_right_corresponding=None,
    device=None    
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

    debug = True if augment_para.get("debug") else False
    if save_volumes is not None and output_dir is not None:
        save_framedimage(
            image_tensor,
            os.path.join(output_dir, save_volumes + "_reoriented_image.mgz"),
            original_framedimage=original_image,            
        )
        save_framedimage(
            label_tensor,
            os.path.join(output_dir, save_volumes + "_reoriented_label.mgz"),
            original_framedimage=original_label,            
        )
            
    crop_size = augment_para.get("crop_size", None)
            
    if "flipping" in augmentations_to_apply:
        flip_prob = augment_para.get("flip_prob")
        aff = original_image.geom.vox2world.matrix

        image_tensor, label_tensor = apply_flipping(image_tensor, label_tensor, aff, left_right_corresponding, flip_prob)
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_flipped_image.mgz"),
                original_framedimage=original_image,                
            )
            save_framedimage(
                label_tensor,
                os.path.join(output_dir, save_volumes + "_flipped_label.mgz"),
                original_framedimage=original_label,                
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
            warp_integrations=augment_para.get("warp_integrations", 7),
            warp_smoothing_range=augment_para.get("warp_smoothing_range", [10, 20]),
            warp_magnitude_range=augment_para.get("warp_magnitude_range", [1, 2]),
            shearing_bounds=augment_para.get("shearing_bounds", 0.015),
            device=device
        )
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_transformed_image.mgz"),
                original_framedimage=original_image,                
            )
            save_framedimage(
                label_tensor,
                os.path.join(output_dir, save_volumes + "_transformed_label.mgz"),
                original_framedimage=original_label,                
            )

    # ??? we are now supporting cropping, randomcrop, randomcrop_center. check to allow only one type of cropping ???
    if "cropping" in augmentations_to_apply:
        # check if the original image already has crop_size
        if (crop_size is not None):
            # image_tensor/label_tensor is non-batched
            image_shape = image_tensor.shape[1:]
            if (np.any(np.array(image_shape) > np.array(crop_size))):
                # calculate the center point to crop the image/label around
                center_point = centroid(label_tensor.cpu().squeeze(0).detach().numpy(), debug=debug)
            
                image_tensor, _ = apply_centercrop(image_tensor, crop_size, center_point=center_point, debug=debug)
                label_tensor, _ = apply_centercrop(label_tensor, crop_size, center_point=center_point, debug=debug)
                if save_volumes is not None and output_dir is not None:
                    save_framedimage(
                        image_tensor,
                        os.path.join(output_dir, save_volumes + "_centercropped_image.mgz"),
                        original_framedimage=original_image,                        
                    )
                    save_framedimage(
                        label_tensor,
                        os.path.join(output_dir, save_volumes + "_centercropped_label.mgz"),
                        original_framedimage=original_label,                        
                    )
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")

    if "randomcrop" in augmentations_to_apply:
        if crop_size is not None:
            bbox_labels = augment_para.get("bbox_labels", None)
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='random', bbox_labels=bbox_labels, debug=debug)
            if save_volumes is not None and output_dir is not None:
                save_framedimage(
                    image_tensor,
                    os.path.join(output_dir, save_volumes + "_randomcropped_image.mgz"),
                    original_framedimage=original_image,                    
                )
                save_framedimage(
                    label_tensor,
                    os.path.join(output_dir, save_volumes + "_randomcropped_label.mgz"),
                    original_framedimage=original_label,                    
                )
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")

    if "randomcrop_center" in augmentations_to_apply:
        if crop_size is not None:
            bbox_labels = augment_para.get("bbox_labels", None)
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='center', bbox_labels=bbox_labels, debug=debug)
            if save_volumes is not None and output_dir is not None:
                save_framedimage(
                    image_tensor,
                    os.path.join(output_dir, save_volumes + "_randomcropped_center_image.mgz"),
                    original_framedimage=original_image,                    
                )
                save_framedimage(
                    label_tensor,
                    os.path.join(output_dir, save_volumes + "_randomcropped_center_label.mgz"),
                    original_framedimage=original_label,                    
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
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_blur_resampled_image.mgz"),
                original_framedimage=original_image,                
            )

    # ??? only allow one "bias_field" or "biasFieldCorruption" ???
    if "bias_field" in augmentations_to_apply:
        image_tensor = apply_bias_field(
            image_tensor, voxsize,
            bias_field_probability=augment_para.get("bias_field_probability", 0.5),
            bias_field_max_magnitude=augment_para.get("bias_field_max_magnitude", 0.1),
            bias_field_smoothing_range=augment_para.get("bias_field_smoothing_range", [1, 2])
        )
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_bias_field_augmented_image.mgz"),
                original_framedimage=original_image,                
            )

    if "biasFieldCorruption" in augmentations_to_apply:
        image_tensor = apply_biasFieldCorruption(
            image_tensor
        )
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_biasFieldCorruption_image.mgz"),
                original_framedimage=original_image,                
            )
            
    if "sampleConditionalGMM" in augmentations_to_apply:
        image_tensor = apply_sampleConditionalGMM(
            label_tensor, expected_classes
        )
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_sampleConditionalGMM_image.mgz"),
                original_framedimage=original_image,                
            )

    if "intensityAugmentation" in augmentations_to_apply:
        image_tensor = apply_intensityAugmentation(
            image_tensor,
        )
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_intensityAugmentation_image.mgz"),
                original_framedimage=original_image,                
            )


    return image_tensor, label_tensor
