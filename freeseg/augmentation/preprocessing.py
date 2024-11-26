import os
import numpy as np
import numpy.random as npr
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
                            priors=None,
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

    transformed_priors = None
    if (priors is not None):
        transformed_priors = voxynth.transform.spatial_transform(priors, trf, method="nearest")

    return transformed_image, transformed_label, transformed_priors


def apply_randomcrop(image, label, crop_size, mode='random', bbox_labels=None, verbose=False):
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
                
    if (verbose):
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


def apply_centercrop(image, crop_size, center_point=None, verbose=False):
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
    if (verbose):
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


# generate an initial synthetic scan G by sampling a GMM conditioned on L described in SynthSeg paper
# (https://www.sciencedirect.com/science/article/pii/S1361841523000506)
def apply_sampleConditionalGMM(label_map, generation_labels, prior_mean=[25, 225], prior_std=[5, 25], prior_distribution='uniform', num_channels=1):
    """
    Generate a synthetic image (num_channels) by sampling a Gaussian Mixture Model conditioned on a label map given as input.
    Each channel is sampled independently.

    GMM-sampling parameters:
      prior_distribution: type of distribution from which we sample the GMM parameters {'uniform', 'normal'}
      prior_mean: hyperparameters controlling the means of Gaussian distributions of the GMM
      prior_std:  hyperparameters controlling the standard deviations of Gaussian distributions of the GMM
 
    label_map: input tensors expected to have shape [1, H, W (,D)]
    sampled_image: output tensor [num_channels, H, W (,D)]
    """    

    assert (generation_labels is not None), 'generation_labels is needed for sampleConditionalGMM'

    # sample means and stds of Gaussian distributions of the GMM
    num_classes = len(generation_labels)
    prior_shape = (num_channels, num_classes)
    if prior_distribution == 'uniform':
        means = np.random.uniform(low=prior_mean[0], high=prior_mean[1], size=prior_shape)
        stds  = np.random.uniform(low=prior_std[0], high=prior_std[1], size=prior_shape)
    elif prior_distribution == 'normal':
        means = np.random.normal(loc=prior_mean[0], scale=prior_mean[1], size=prior_shape)
        stds  = np.random.normal(loc=prior_std[0], scale=prior_std[1], size=prior_shape)
    else:
        raise ValueError("Prior distribution not supported, should be 'uniform' or 'normal'.")

    # reset all negative values to zero
    means[means < 0] = 0
    stds[stds < 0] = 0

    # the following is taken from SynthSeg.model_inputs.build_model_inputs()
    # https://github.com/BBillot/SynthSeg/blob/master/SynthSeg/model_inputs.py#L142C1-L149C1
    random_coef = npr.uniform()
    if random_coef > 0.95:  # reset the background to 0 in 5% of cases
        means[0] = 0
        stds[0] = 0
    elif random_coef > 0.7:  # reset the background to low Gaussian in 25% of cases
        means[0] = npr.uniform(0, 15)
        stds[0] = npr.uniform(0, 5)

    # generate synthetic image
    label_map = label_map.squeeze(0)   # remove the channel axis
    sampled_image = torch.zeros((num_channels, *label_map.shape)).to(label_map.device)
    for labelid in range(num_classes):
        label_indices = (label_map == generation_labels[labelid])
        indices_count = label_indices.sum()

        # each channel is sampled independently
        for n_channel in range(num_channels):
            gauss_samples = np.random.normal(loc=means[n_channel, labelid], scale=stds[n_channel, labelid], size=indices_count)
            gauss_samples = torch.from_numpy(gauss_samples).to(sampled_image.dtype).to(label_map.device)
            sampled_image[n_channel][label_indices] = gauss_samples
 
    return sampled_image


# This is descibed in Hypothalamus paper (https://www.sciencedirect.com/science/article/pii/S1053811920307734)
#  "
#    The augmentation model also accounts for non-uniformities in the magnetic field commonly observed in MR scanners (Simmons et al., 1994).
#    Because this phenomenon translates into intensity inhomogeneities smoothly varying across MRI scans (Sled and Pike, 1998),
#    we model it with a multiplicative smooth field. As before, we sample a small low resolution field (e.g., of size 4 × 4 × 4),
#    and upscale it to image size with linear interpolation. Then, we take the voxel-wise exponential to ensure the non-negativity of this field.
#    Finally, we multiply the spatially deformed scan by the obtained bias field to corrupt its intensities (Fig. 1(c)).
#  "
def apply_biasFieldCorruption(image, bias_field_std=.5, bias_scale=.025, prob=0.95):
    """
    Apply a smooth random bias field to the input tensor by applying the following steps:

    1) sample a value for the standard deviation of a centred normal distribution
    2) a small-size SVF is sampled from this normal distribution
    3) the small SVF is then resized with trilinear interpolation to image size
    4) it is rescaled to positive values by taking the voxel-wise exponential
    5) it is multiplied to the input tensor.

    The input tensor is expected to have shape [C, H, W (,D)].

    The bias field is sampled and applied independently for each channel of the input tensor. 

    bias_field_std: max value to sample the standard deviation of a centred normal distribution from range [0, bias_field_std]
    bias_scale:     ratio between the shape of the input tensor and the shape of the sampled SVF.
    prob:           probability to apply this bias field corruption.
    """

    if (not np.random.rand() < prob or bias_field_std <= 0):
        return image
    
    num_channels = image.shape[0]
    ndims = image.ndim - 1
    image_shape = image.shape[1:]
    
    # sampling shapes, the bias field will be sampled and applied independently for each channel of the input tensor
    std_shape = [num_channels] + [1] * ndims   # [C, 1, 1, 1]
    small_bias_shape = [num_channels] + [math.ceil(image_shape[i] * bias_scale) for i in range(len(image_shape))]  # [C, h, w, (,d)]

    # sample small bias field
    stddev = np.random.uniform(low=0, high=bias_field_std, size=std_shape)
    bias_field = np.random.normal(loc=0, scale=stddev, size=small_bias_shape) 
    
    # resize bias field and take exponential    
    bias_field_tensor = torch.from_numpy(bias_field).to(image.device, dtype=image.dtype)
    bias_field_tensor = torch.nn.functional.interpolate(bias_field_tensor.unsqueeze(0), image_shape, mode='trilinear')
    bias_field_tensor = bias_field_tensor.squeeze(0)  # remove the dummy batch dimension
    bias_field_tensor = torch.exp(bias_field_tensor)

    # element-wise multiplication
    bf_augmented_image = torch.mul(bias_field_tensor, image)

    return bf_augmented_image


# This is descibed in Hypothalamus paper (https://www.sciencedirect.com/science/article/pii/S1053811920307734)
#   "
#     In order to make the network robust against acquisition procedures,
#     we add further global intensity augmentation by shifting the brightness and contrast of the image with randomly sampled values (Fig. 1(d)).
#     The obtained scan is subsequently flipped along the right-left axis with a probability of 0.5 (Fig. 1(e)), and randomly cropped to a size of 160^3,
#     which is more than large enough to ensure that the hypothalamus is always present in the resulting scan.
#     Finally, intensities are rescaled between [0,1] with min-max normalisation.
#     Additional examples of augmented images are shown in the Supplementary materials (Fig. S1).
#   "
def apply_intensityAugmentation(image, noise_std=0, normalise=True, gamma_std=0, prob_noise=0.95, prob_gamma=1):
    """
    Augment the intensities of the input tensor. All channels are augmented separately.

    The following steps are applied (all are optional):
    1) white noise corruption, with a randomly sampled std dev.
    2) min-max normalisation
    3) gamma augmentation (i.e. voxel-wise exponentiation by a randomly sampled power)

    The input tensor is expected to have shape [C, H, W (,D)].

    The noise and gamma are sampled and applied independently for each channel of the input tensor.

    noise_std:  max value to sample the standard deviation of the Gaussian white noise from the range [0, noise_std].
                Default is 0, where white noise corruption is skipped.
    normalise:  whether to apply min-max normalisation, to normalise between 0 and 1. Default is True.
    gamma_std:  standard deviation of the normal distribution from which we sample gamma.
                Default is 0, where no gamma augmentation occurs.
    prob_noise: probability to apply noise injection
    prob_gamma: probability to apply gamma augmentation
    """
    
    num_channels = image.shape[0]
    ndims = image.ndim - 1

    image_cpu = image.cpu().detach().numpy()    

    sample_shape = None
    # noise and gamma are sampled and applied independently for each channel of the input tensor
    if (noise_std > 0 or gamma_std > 0):
        sample_shape = [num_channels] + [1] * ndims
    
    # add noise with predefined probability
    if (noise_std > 0 and np.random.rand() < prob_noise):
        noise_stddev = np.random.uniform(low=0, high=noise_std, size=sample_shape)
        noise = np.random.normal(loc=0, scale=noise_stddev, size=image_cpu.shape)
        image_cpu += noise

    # normalise
    if (normalise):
        # simple min and max
        axis = tuple(dim for dim in range(1, ndims+1)) # axis=(H, W (,D))
        m = np.min(image_cpu, axis=axis) # [C, 1]
        M = np.max(image_cpu, axis=axis) # [C, 1]
        
        m = np.expand_dims(m, axis=axis) # [C, 1, 1 (,1)]
        M = np.expand_dims(M, axis=axis) # [C, 1, 1 (,1)]

        # normalise
        image_cpu = np.clip(image_cpu, a_min=m, a_max=M)
        image_cpu = (image_cpu - m) / (M - m + np.finfo(float).eps)

    # apply voxel-wise exponentiation with predefined probability
    if (gamma_std > 0 and np.random.rand() < prob_gamma):
        gamma = np.random.normal(loc=0, scale=gamma_std, size=sample_shape)
        image_cpu = np.power(image_cpu, np.exp(gamma))

    return torch.from_numpy(image_cpu).to(image.device)


def apply_augmentations(
    image_tensor,
    label_tensor,
    original_image,
    original_label,
    generation_labels,
    augment_para,
    voxsize,
    priors_tensor=None,
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

    verbose = True if augment_para.get("verbose") else False
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
        save_framedimage(
            priors_tensor,
            os.path.join(output_dir, save_volumes + "_reoriented_prior.mgz"),
            original_framedimage=original_image,
            dtype=float
        )        
            
    crop_size = augment_para.get("crop_size", None)
            
    if "flipping" in augmentations_to_apply:
        flip_prob = augment_para.get("flip_prob")
        aff = original_image.geom.vox2world.matrix

        # ??? todo: need to flip priors too ???
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
        image_tensor, label_tensor, priors_tensor = apply_spatial_transform(
            image_tensor, label_tensor, voxsize,
            priors=priors_tensor,
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
            save_framedimage(
                priors_tensor,
                os.path.join(output_dir, save_volumes + "_transformed_prior.mgz"),
                original_framedimage=original_image,
                dtype=float
            )            

    # ??? we are now supporting cropping, randomcrop, randomcrop_center. check to allow only one type of cropping ???
    if "cropping" in augmentations_to_apply:
        # check if the original image already has crop_size
        if (crop_size is not None):
            # image_tensor/label_tensor is non-batched
            image_shape = image_tensor.shape[1:]
            if (np.any(np.array(image_shape) > np.array(crop_size))):
                # calculate the center point to crop the image/label around
                center_point = centroid(label_tensor.cpu().squeeze(0).detach().numpy(), verbose=verbose)
            
                image_tensor, _ = apply_centercrop(image_tensor, crop_size, center_point=center_point, verbose=verbose)
                label_tensor, _ = apply_centercrop(label_tensor, crop_size, center_point=center_point, verbose=verbose)
                if (priors_tensor is not None):
                    priors_tensor, _ = apply_centercrop(priors_tensor, crop_size, center_point=center_point, verbose=verbose)

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
                    save_framedimage(
                        priors_tensor,
                        os.path.join(output_dir, save_volumes + "_centercropped_prior.mgz"),
                        original_framedimage=original_image,
                        dtype=float
                    )                    
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")

    # ??? todo: apply_randomcrop() needs to handle priors ???
    if "randomcrop" in augmentations_to_apply:
        if crop_size is not None:
            bbox_labels = augment_para.get("bbox_labels", None)
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='random', bbox_labels=bbox_labels, verbose=verbose)
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

    # ??? todo: apply_randomcrop() needs to handle priors ???
    if "randomcrop_center" in augmentations_to_apply:
        if crop_size is not None:
            bbox_labels = augment_para.get("bbox_labels", None)
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='center', bbox_labels=bbox_labels, verbose=verbose)
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

    if "sampleConditionalGMM" in augmentations_to_apply:
        num_channels = augment_para.get("num_channels", 1)  # dataset expected_num_channels
        prior_distribution = augment_para.get("prior_distribution", "uniform")  # 'normal'
        prior_mean = augment_para.get("prior_mean", [25, 225])
        prior_std = augment_para.get("prior_std", [5, 25])
        image_tensor = apply_sampleConditionalGMM(label_tensor, generation_labels,
                                                  prior_mean=prior_mean, prior_std=prior_std, prior_distribution=prior_distribution, num_channels=num_channels)
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + f"_sampleConditionalGMM_{prior_distribution}_image.mgz"),
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
        bias_field_std = augment_para.get("bias_field_max_magnitude", .7)  # SynthSeg
        bias_scale = augment_para.get("bias_field_scale", .025)
        prob = augment_para.get("bias_field_probability", 0.95)
        image_tensor = apply_biasFieldCorruption(image_tensor, bias_field_std=bias_field_std, bias_scale=bias_scale, prob=prob)
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_biasFieldCorruption_image.mgz"),
                original_framedimage=original_image,                
            )
            
    if "intensityAugmentation" in augmentations_to_apply:
        ia_noise_std = augment_para.get("ia_noise_std", 1.0)  # default is 0 for SynthSeg, no white noise added
        ia_normalise = augment_para.get("ia_normalise", True)
        ia_gamma_std = augment_para.get("ia_gamma_std", 0.5)
        ia_prob_noise = augment_para.get("ia_prob_noise", 0.95)
        ia_prob_gamma = augment_para.get("ia_prob_gamma", 1)
        image_tensor = apply_intensityAugmentation(image_tensor, noise_std=ia_noise_std, normalise=ia_normalise, gamma_std=ia_gamma_std,
                                                   prob_noise=ia_prob_noise, prob_gamma=ia_prob_gamma)
        if save_volumes is not None and output_dir is not None:
            save_framedimage(
                image_tensor,
                os.path.join(output_dir, save_volumes + "_intensityAugmentation_image.mgz"),
                original_framedimage=original_image,                
            )


    return image_tensor, label_tensor, priors_tensor
