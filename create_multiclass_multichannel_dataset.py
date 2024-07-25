import argparse
import os
import numpy as np
import nibabel as nib


def create_labeled_spheres(
    vol_shape, num_channels, center_sphere, symmetric_spheres, noise_mean, noise_std, channel_noise
):
    vol = np.zeros(vol_shape, dtype=np.uint8)
    label_vol = np.zeros(vol_shape, dtype=np.uint8)

    center, radius, label = center_sphere
    xx, yy, zz = np.mgrid[: vol_shape[0], : vol_shape[1], : vol_shape[2]]
    sphere_mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 + (zz - center[2]) ** 2 <= radius**2
    vol[sphere_mask] = label
    label_vol[sphere_mask] = label

    offset, y, z, radius, label1, label2 = symmetric_spheres
    center1 = [vol_shape[0] // 2 - offset, y, z]
    center2 = [vol_shape[0] // 2 + offset, y, z]

    xx, yy, zz = np.mgrid[: vol_shape[0], : vol_shape[1], : vol_shape[2]]
    sphere_mask1 = (xx - center1[0]) ** 2 + (yy - center1[1]) ** 2 + (
        zz - center1[2]
    ) ** 2 <= radius**2
    vol[sphere_mask1] = label1
    label_vol[sphere_mask1] = label1

    sphere_mask2 = (xx - center2[0]) ** 2 + (yy - center2[1]) ** 2 + (
        zz - center2[2]
    ) ** 2 <= radius**2
    vol[sphere_mask2] = label2
    label_vol[sphere_mask2] = label2

    vol_4d = np.repeat(vol[..., np.newaxis], num_channels, axis=-1)

    for i, noise_std in enumerate(channel_noise):
        noise = np.random.normal(noise_mean, noise_std, vol_shape)
        vol_4d[..., i] = vol_4d[..., i] + noise.astype(np.uint8)

    return vol_4d, label_vol


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset of 3D volumes with labeled spheres and noise."
    )
    parser.add_argument(
        "--output-dir", type=str, default="dataset", help="Output directory for the dataset"
    )
    parser.add_argument(
        "--dataset-size",
        type=int,
        default=10,
        help="Number of samples in the dataset (default: 10)",
    )
    parser.add_argument(
        "--vol-shape",
        type=int,
        nargs=3,
        default=[64, 64, 64],
        help="Shape of the 3D volume (default: 64 64 64)",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels in the output volume (default: 1)",
    )
    parser.add_argument(
        "--center-sphere", type=str, required=True, help="Center sphere parameters: radius,label"
    )
    parser.add_argument(
        "--symmetric-spheres",
        type=str,
        required=True,
        help="Symmetric sphere parameters: offset,y,z,radius,label1,label2",
    )
    parser.add_argument(
        "--noise-mean", type=float, default=0.0, help="Mean of the Gaussian noise (default: 0.0)"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        nargs="+",
        required=True,
        help="Standard deviation of the Gaussian noise for each channel (specify multiple values for different noise levels per channel)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.noise_std) != args.channels:
        raise ValueError(
            f"Number of noise STD values ({len(args.noise_std)}) does not match the number of channels ({args.channels})"
        )

    center_radius, center_label = map(int, args.center_sphere.split(","))
    center_sphere = (
        (args.vol_shape[0] // 2, args.vol_shape[1] // 2, args.vol_shape[2] // 2),
        center_radius,
        center_label,
    )

    offset, y, z, radius, label1, label2 = map(int, args.symmetric_spheres.split(","))
    symmetric_spheres = (offset, y, z, radius, label1, label2)

    os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "labels"), exist_ok=True)

    for i in range(args.dataset_size):
        vol_4d, label_vol = create_labeled_spheres(
            args.vol_shape,
            args.channels,
            center_sphere,
            symmetric_spheres,
            args.noise_mean,
            args.noise_std,
            args.noise_std,
        )

        image_filename = os.path.join(args.output_dir, "images", f"image{i:03d}.nii.gz")
        label_filename = os.path.join(args.output_dir, "labels", f"label{i:03d}.nii.gz")

        nib.save(nib.Nifti1Image(vol_4d, affine=np.eye(4)), image_filename)
        nib.save(nib.Nifti1Image(label_vol, affine=np.eye(4)), label_filename)

    print(
        f"Dataset generated successfully. Images and labels saved in '{args.output_dir}' directory."
    )


if __name__ == "__main__":
    main()
