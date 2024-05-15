import os
import torch
from voxynth import voxynth
from utils.data_utils import save_volume


def apply_flipping(image, label, axis=1):
    """Applies a random left-right flip to image and label volumes."""
    flipped_image, flipped_label = voxynth.transform.random_flip(
        axis, image, label, prob=1.0
    )
    return flipped_image, flipped_label


def apply_spatial_transform(image, label, voxsize):
    """Applies a random spatial transformation to image and label volumes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    label = label.to(device)

    trf = voxynth.transform.random_transform(
        shape=image.shape[1:],
        device=device,
        affine_probability=1.0,
        max_translation=5.0,
        max_rotation=5.0,
        max_scaling=1.1,
        warp_probability=1.0,
        warp_integrations=5,
        warp_smoothing_range=[10, 20],
        warp_magnitude_range=[1, 2],
    )

    transformed_image = voxynth.transform.spatial_transform(image, trf)
    transformed_label = voxynth.transform.spatial_transform(
        label, trf, method="nearest"
    )
    return transformed_image, transformed_label


def apply_cropping(image, label, crop_size=None):
    """Applies center cropping to image and label volumes."""
    cropped_image = voxynth.augment.apply_center_crop(image, crop_size)
    cropped_label = voxynth.augment.apply_center_crop(label, crop_size)
    return cropped_image, cropped_label


def apply_blur_resample(image, voxsize):
    """Applies blurring and resampling to the image volume."""
    image_cpu = image.cpu()
    blur_resampled_image = voxynth.augment.image_augment(
        image_cpu,
        normalize=True,
        smoothing_probability=0.5,
        smoothing_max_sigma=2.0,
        added_noise_probability=0.5,
        added_noise_max_sigma=0.05,
        gamma_scaling_probability=0.5,
        gamma_scaling_max=0.8,
        resized_probability=0,
        resized_one_axis_probability=0,
        resized_max_voxsize=2,
    )
    return blur_resampled_image


def apply_bias_field(image, voxsize):
    """Applies bias field augmentation to the image volume."""
    bf_augmented_image = voxynth.augment.image_augment(
        image,
        voxsize=voxsize,
        bias_field_probability=0.5,
        bias_field_max_magnitude=0.1,
        bias_field_smoothing_range=[1, 2],
    )
    return bf_augmented_image


def apply_augmentations(
    image_tensor,
    label_tensor,
    original_image,
    original_label,
    voxsize,
    crop_size=None,
    output_dir=None,
    save_volumes=False,
    augmentations_to_apply=None,
):
    """
    Apply data augmentations to the image and label tensors and optionally save intermediate results.

    Args:
        image_tensor (torch.Tensor): PyTorch tensor representing the image volume.
        label_tensor (torch.Tensor): PyTorch tensor representing the label volume.
        original_image (surfa.Volume): Original loaded image volume.
        original_label (surfa.Volume): Original loaded label volume.
        voxsize (tuple): Voxel size of the volumes.
        output_dir (str, optional): Directory to save the intermediate results. If None, volumes are not saved.
        save_volumes (bool, optional): Whether to save augmented volumes. Defaults to False.
        augmentations_to_apply (list, optional): A list of augmentation names to apply.
                                                If None, all available augmentations are applied.

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

    if "flipping" in augmentations_to_apply:
        image_tensor, label_tensor = apply_flipping(image_tensor, label_tensor)
        if save_volumes and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, "flipped_image.mgz"),
            )
            save_volume(
                label_tensor,
                original_label,
                os.path.join(output_dir, "flipped_label.mgz"),
            )

    if "spatial_transform" in augmentations_to_apply:
        image_tensor, label_tensor = apply_spatial_transform(
            image_tensor, label_tensor, voxsize
        )
        if save_volumes and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, "transformed_image.mgz"),
            )
            save_volume(
                label_tensor,
                original_label,
                os.path.join(output_dir, "transformed_label.mgz"),
            )

    if "cropping" in augmentations_to_apply:
        if crop_size is not None:
            image_tensor, label_tensor = apply_cropping(image_tensor, label_tensor, crop_size)
            if save_volumes and output_dir is not None:
                save_volume(
                    image_tensor,
                    original_image,
                    os.path.join(output_dir, "cropped_image.mgz"),
                )
                save_volume(
                    label_tensor,
                    original_label,
                    os.path.join(output_dir, "cropped_label.mgz"),
                )
        else:
            raise ValueError("Crop size must be provided when using the 'cropping' augmentation.")

    if "blur_resample" in augmentations_to_apply:
        image_tensor = apply_blur_resample(image_tensor, voxsize)
        if save_volumes and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, "blur_resampled_image.mgz"),
            )

    if "bias_field" in augmentations_to_apply:
        image_tensor = apply_bias_field(image_tensor, voxsize)
        if save_volumes and output_dir is not None:
            save_volume(
                image_tensor,
                original_image,
                os.path.join(output_dir, "bias_field_augmented_image.mgz"),
            )

    return image_tensor, label_tensor
