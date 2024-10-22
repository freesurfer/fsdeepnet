import numpy as np
import surfa as sf
import torch
import yaml

def load_volume(file_path, orientation=None, device=None):
    """
    Load a volume from a file and convert it to a PyTorch tensor.
    The loaded volume data is re-oriented to conform to a specific slice orientation.
    
    Args:
        file_path (str): Path to the volume file.
    
    Returns:
        tuple: A tuple containing the loaded volume and its PyTorch tensor representation.
               tensor returned is non-batched [C, H, W (,D)]
    """
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    volume = sf.load_volume(file_path)
    orig_orientation = sf.transform.orientation.rotation_matrix_to_orientation(volume.geom.vox2world.matrix)
    if (orientation is not None):
        volume = volume.reorient(orientation)

    # handle both 2D and 3D data
    volume_data_native = volume.framed_data.squeeze().astype(volume.dtype.newbyteorder('='))
    if (volume_data_native.ndim < 4):
        volume_data_native = np.expand_dims(volume_data_native, axis=-1)
    #volume_data_writable = np.copy(volume_data_native)  # Create a writable copy of the array
    #volume_tensor = torch.from_numpy(volume_data_writable).movedim(-1, 0)
    volume_tensor = torch.from_numpy(volume_data_native).movedim(-1, 0).to(device)
    
    return volume, volume_tensor, orig_orientation

def save_volume(volume_tensor, original_volume, output_file, orientation=None, labels=None, reshape=False):
    """
    Save the augmented volume to a file.
    
    Args:
        volume_tensor (torch.Tensor): Augmented volume tensor, non-batched [C, H, W (,D)]
        original_volume (surfa.Volume): Original loaded volume.
        output_file (str): Path to the output file.
    """
    # the input tensor is non-batched [C, H, W, (D)], move C to the last axis, C >= 1
    tensor_cpu = volume_tensor.cpu().movedim(0, -1).squeeze()
    np_vol = tensor_cpu.detach().numpy().astype(original_volume.dtype)
    surfa_vol = original_volume.new(np_vol)
    
    if (reshape):
        surfa_vol = surfa_vol.reshape(original_volume.shape)
    if (orientation is not None):
        surfa_vol = surfa_vol.reorient(orientation)
    if (labels is not None):
        surfa_vol.labels = labels

    surfa_vol.save(output_file)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def remap_labels(labels, mapping):
    remapped_labels = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        if (old_label != new_label):
            remapped_labels[labels == old_label] = new_label
    return remapped_labels

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
    if (onehot_labels.ndim == 5):  # 3D
        return onehot_labels.permute(0, 4, 1, 2, 3)
    elif (onehot_labels.ndim == 4): # 2D
        return onehot_labels.permute(0, 3, 1, 2)
    else:
        raise ValueError("Onehot encoded label is expected to be 4 or 5 dimensions")


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
    

def centroid(label):
    """
    calculate centroid for given label image

    Args:
        label (numpy.narray): input label image array

    Returns:
        tuple: Coordinates of the center point for given label image
    """

    # binarize the label image
    mask = np.zeros(label.shape).astype(int)
    mask[label > 0] = 1

    # calculate bounding box coordinates
    lowerbound = np.zeros(label.ndim).astype(int)
    upperbound = np.zeros(label.ndim).astype(int)
    if (np.any(mask == 1)):  # check if any of the labels exist
        coords = np.where(mask == 1)
        for dim, coord in enumerate(coords):
            lowerbound[dim] = np.min(coord)
            upperbound[dim] = np.max(coord)

    centroid = lowerbound + (upperbound - lowerbound)/2
    
    return centroid.astype(int)    
    #return tuple(centroid.astype(int))


def DataGenerator(dataloader, device=None):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while (True):
        for n_batch, (dataset_idx, images, labels) in enumerate(dataloader):
            images, labels = images.to(device).float(), labels.to(device)
            
            # extracts the single value from the dataset_idx tensor
            # returns it as a Python scalar
            yield n_batch, images, labels, dataset_idx.item()


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
