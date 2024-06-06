import os
import numpy as np
import torch
from voxynth import voxynth
from utils.data_utils import save_volume, get_ras_axes


def apply_flipping(image, label, aff, left_right_corresponding, flip_prob=0.5):
    """Applies a random left-right flip to image and label volumes."""
    """Swaps left-right labels on label volume."""

    if (voxynth.utility.chance(flip_prob)):
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


def apply_randomcrop(image, label, crop_size=None, mode='random'):
    """Randomly crop input tensors to a given shape. The input tensors are expected to have shape [batch H W D]."""

    # assuming image and label have the same dimensions
    input_shape = image.shape[1:]

    if (mode == 'random'):
        crop_max_val = np.array(input_shape) - np.array(crop_size)            
        start_coords = np.random.uniform(low=0, high=crop_max_val).astype(int)
        end_coords = start_coords + np.array(crop_size)
    elif (mode == 'center'):
        start_center = np.array(crop_size)/2
        end_center = np.array(input_shape) - np.array(crop_size)/2
        center_point = np.random.uniform(low=start_center, high=end_center).astype(int)

        start_coords = np.maximum(np.array(center_point)-np.array(crop_size)/2, 0).astype(int)
        end_coords = np.minimum(np.array(center_point)+np.array(crop_size)/2, np.array(input_shape)).astype(int)

    #print(f"apply_randomcrop({mode}) - input_shape: {input_shape}, crop_size: {crop_size}, start_coords: {start_coords}, end_coords: {end_coords}")

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
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='random')
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
            image_tensor, label_tensor = apply_randomcrop(image_tensor, label_tensor, crop_size, mode='center')
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
