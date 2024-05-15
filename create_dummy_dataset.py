import os
import random
import subprocess
import argparse


# Function to run the mri_volsynth command with specified parameters
def run_mri_volsynth(template, output, gstd, verbose):
    command = [
        "mri_volsynth",
        "--template",
        template,
        "--offset",
        "--o",
        output,
        "--gstd",
        str(gstd),
    ]

    # Set output and error behavior based on verbosity
    if verbose:
        subprocess.run(command, check=True)
    else:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate dummy image-label pairs for segmentation."
    )
    parser.add_argument(
        "--num_images", type=int, default=30, help="Number of images to generate (default: 30)"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="data/dummy",
        help="Output folder path (default: data/dummy)",
    )
    parser.add_argument(
        "--template_file",
        type=str,
        default="data/dummy/sphere_template.mgz",
        help="Template file path (default: data/dummy/sphere_template.mgz)",
    )
    parser.add_argument(
        "--gstd_mean", type=float, default=0, help="Mean for the Gaussian distribution (default: 0)"
    )
    parser.add_argument(
        "--gstd_stddev",
        type=float,
        default=0.5,
        help="Standard deviation for the Gaussian distribution (default: 0.5)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output (default: off)"
    )
    args = parser.parse_args()

    # Extract command line arguments
    num_images = args.num_images
    output_folder = args.output_folder
    template_file = args.template_file
    gstd_mean = args.gstd_mean
    gstd_stddev = args.gstd_stddev
    verbose = args.verbose

    # Ensure output folders exist
    output_images = os.path.join(output_folder, "images")
    output_labels = os.path.join(output_folder, "labels")
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)

    # Generate images and corresponding labels
    for i in range(1, num_images + 1):
        # Generate a random Gaussian standard deviation
        gstd = abs(random.gauss(gstd_mean, gstd_stddev))

        # Format image names with leading zeros
        image_num = f"{i:03d}"
        image_name = f"image{image_num}.mgz"

        # Generate file paths
        image_path = os.path.join(output_images, image_name)
        label_path = os.path.join(output_labels, image_name)

        # Run mri_volsynth with the current parameters
        run_mri_volsynth(template_file, image_path, gstd, verbose)

        # Copy the label file from the template
        if verbose:
            subprocess.run(["cp", template_file, label_path], check=True)
        else:
            subprocess.run(
                ["cp", template_file, label_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

    print(f"Generated {num_images} image-label pairs in {output_folder}.")


if __name__ == "__main__":
    main()

# image copies in bash
# for i in {02..30}; do cp image01.mgz image$i.mgz; done
# for i in $(seq -w 2 30); do cp image001.mgz image$(printf "%03d" $i).mgz; done



# binarize all files in a folder
# for i in $(seq -w 1 30); do mri_binarize --i image${i}.mgz --min 0.5 --o image${i}.mgz; done
