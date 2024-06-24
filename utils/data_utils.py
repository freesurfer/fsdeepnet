import os
import numpy as np
import surfa as sf
import torch
import yaml
import json
import logging
from omegaconf import DictConfig

log = logging.getLogger(__name__)

def load_volume(file_path, orientation=None):
    """
    Load a volume from a file and convert it to a PyTorch tensor.
    The loaded volume data is re-oriented to conform to a specific slice orientation.
    
    Args:
        file_path (str): Path to the volume file.
    
    Returns:
        tuple: A tuple containing the loaded volume and its PyTorch tensor representation.
    """
    volume = sf.load_volume(file_path)
    if (orientation is not None):
        volume = volume.reorient(orientation, copy=True)
    
    volume_data_native = volume.framed_data.astype(volume.dtype.newbyteorder('='))
    volume_data_writable = np.copy(volume_data_native)  # Create a writable copy of the array
    volume_tensor = torch.from_numpy(volume_data_writable).movedim(-1, 0)
    return volume, volume_tensor

def save_volume(volume_tensor, original_volume, output_file):
    """
    Save the augmented volume to a file.
    
    Args:
        volume_tensor (torch.Tensor): Augmented volume tensor.
        original_volume (surfa.Volume): Original loaded volume.
        output_file (str): Path to the output file.
    """
    tensor_cpu = volume_tensor.cpu().squeeze(0)
    np_vol = tensor_cpu.detach().numpy().astype(original_volume.dtype)
    surfa_vol = original_volume.new(np_vol)
    surfa_vol.save(output_file)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_label_mapping(labels, cfg: DictConfig):
    """Create a label mapping and save it to a JSON file."""
    unique_labels = torch.unique(labels)
    mapping = {label.item(): i for i, label in enumerate(unique_labels)}

    # Create output directory if it doesn't exist
    # if not os.path.exists(cfg.output_dir):
    #     os.makedirs(cfg.output_dir)

    filepath = os.path.join(cfg.output_dir, "label_mapping.json")
    log.info(f"Saving label mapping to {filepath}")
    with open(filepath, 'w') as f:
        json.dump(mapping, f, indent=4)

    return mapping

def remap_labels(labels, mapping):
    remapped_labels = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        remapped_labels[labels == old_label] = new_label
    return remapped_labels

def load_labels_color_table(file_path):
    labels_color_table = {}
    with open(file_path, 'r') as file:
        for line in file:
            label, color = line.strip().split(':')
            labels_color_table[int(label)] = tuple(map(int, color.split(',')))
    return labels_color_table

def embed_colors(predicted_labels, labels_color_table):
    colored_labels = np.zeros(predicted_labels.shape + (3,), dtype=np.uint8)
    for label, color in labels_color_table.items():
        mask = predicted_labels == label
        colored_labels[mask] = color
    return colored_labels

def onehot(labels, num_classes, device=None):
    """
    One-hot encode a tensor of integer labels.

    Args:
        labels (torch.Tensor): A tensor of integer labels.
        num_classes (int): The number of classes.
        device (torch.device, optional): The desired device (CPU or GPU). 
                                         If None, defaults to the device of the 'labels' tensor.

    Returns:
        torch.Tensor: A one-hot encoded tensor. 
    """
    if device is None:
        device = labels.device 
    onehot_labels = torch.eye(num_classes, device=device)[labels.long().squeeze(1)]
    return onehot_labels.permute(0, 4, 1, 2, 3)


def bbox(image, labels):
    """
    calculate label bounding box in the image

    Args:
        image (torch.Tensor):
        labels (list):

    Returns:
        lowerbound (1d numpy array), upperbound (1d numpy array)
    """

    image_cpu = image.cpu().squeeze(0)
    np_vol = image_cpu.detach().numpy()
    
    # binarize the image with labels given
    mask = np.zeros(np_vol.shape).astype(int)
    for label in (labels):
        label_indices = (np_vol == label)
        mask[label_indices] = 1

    # calculate bounding box coordinates
    lowerbound = np.zeros(np_vol.ndim).astype(int)
    upperbound = np.zeros(np_vol.ndim).astype(int)
    coords = np.where(mask == 1)
    for dim, coord in enumerate(coords):
        lowerbound[dim] = np.min(coord)
        upperbound[dim] = np.max(coord)

    # leave some rooms
    lowerbound -= 1
    upperbound += 1
    
    return lowerbound, upperbound
    



    
# ================================================================================================
#                                        Lab2Im Utilities
# ================================================================================================
def get_ras_axes(aff, n_dims=3):
    """This function finds the RAS axes corresponding to each dimension of a volume, based on its affine matrix.
    :param aff: affine matrix Can be a 2d numpy array of size n_dims*n_dims, n_dims+1*n_dims+1, or n_dims*n_dims+1.
    :param n_dims: number of dimensions (excluding channels) of the volume corresponding to the provided affine matrix.
    :return: two numpy 1d arrays of length n_dims, one with the axes corresponding to RAS orientations,
    and one with their corresponding direction.
    """
    aff_inverted = np.linalg.inv(aff)
    img_ras_axes = np.argmax(np.absolute(aff_inverted[0:n_dims, 0:n_dims]), axis=0)
    for i in range(n_dims):
        if i not in img_ras_axes:
            unique, counts = np.unique(img_ras_axes, return_counts=True)
            incorrect_value = unique[np.argmax(counts)]
            img_ras_axes[np.where(img_ras_axes == incorrect_value)[0][-1]] = i

    return img_ras_axes
