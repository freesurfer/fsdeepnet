import os
import importlib
import logging
import subprocess
import random
import platform
import numpy as np
import surfa as sf
import torch
import yaml


def load_framedimage(file_path, orientation=None, device=None, ndims=3):
    """
    Load a framedimage from a file and convert it to a PyTorch tensor.
    The loaded framedimage 3D data is re-oriented to conform to a specific slice orientation.
    surfa.image.framed.reorient() is not yet implemented for 2D data.

    use surfa.load_volume() to load both 3D and 2D images so we can handle 2D multi-channel data
    non-batched tensor is returned [C, H, W (,D)] (ndims = tensor.ndim - 1)
    
    Args:
        file_path (str): Path to the framedimage file.
    
    Returns:
        tuple: A tuple containing the loaded framedimage (surfa.Volume)
               and its PyTorch tensor representation.

               tensor returned is non-batched [C, H, W (,D)]
    """
    assert (ndims == 3 or ndims == 2), "data needs to be either 3D or 2D"

    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    framedimage = sf.load_volume(file_path)

    orig_geom = framedimage.geom.copy()   # save a copy of original image geom before it is reoriented
    # surfa.image.framed.reorient() is not yet implemented for 2D data
    if (ndims == 3 and orientation is not None):
        framedimage = framedimage.reorient(orientation, copy=False, inplace=True)

    # framedimage.framed_data has shape [H, W, D, C] (3D) or [H, W, C, 1] (2D)
    framedimage_data_native = framedimage.framed_data.astype(framedimage.dtype.newbyteorder('='))
    if (ndims == 2):
        # remove last axis, so ndims = tensor.ndim - 1
        framedimage_data_native = framedimage_data_native.squeeze(-1)
    framedimage_tensor = torch.from_numpy(framedimage_data_native).movedim(-1, 0).to(device)
    
    return framedimage, framedimage_tensor, orig_geom


def save_framedimage(framedimage_tensor, output_file, original_framedimage=None, geom=None, orientation=None,
                     labels=None, onehotencoded=False, dtype=None, resample=False, method='nearest', target_im_geom=None):
    """
    Save the augmented framedimage to a file.
    input tensor is non-batched [C, H, W (,D)] (ndims = tensor.ndim - 1)
    
    Args:
        framedimage_tensor (torch.Tensor): Augmented framedimage tensor, non-batched [C, H, W(, D)]
        original_framedimage: Original loaded framedimage (surfa.Volume).
        output_file (str): Path to the output file.
        geom: The surfa.ImageGeometry for the input framedimage_tensor
        resample: Whether to resample to original_framedimage space
        method: resampling method if resample=True
        target_im_geom: If resample=False, it is the surfa.ImageGeometry for the saved image
    """
    # the input tensor is non-batched [C, H, W(, D)], move C to the last axis, C >= 1
    tensor_cpu = framedimage_tensor.cpu().movedim(0, -1)
    ndims = framedimage_tensor.ndim - 1  # get the surfa.FramedArray.basedim
    np_image = tensor_cpu.detach().numpy()  # [H, W, (D,) C]

    dtype = original_framedimage.dtype if ((dtype is None) and (original_framedimage is not None)) else dtype
    if (dtype is not None):
        np_image = np_image.astype(dtype)        

    """
    output input tensor as surfa.Slice only if ndim=2 and onehotencoded=True
    this will output posteriors and onehot encoded label as 4D volume [H, W, D, nlabels], D=1 for 2D
    other image/label is output as [H, W, (D,) C]
    """
    if (original_framedimage is not None):
        geom = geom if (geom is not None) else original_framedimage.geom
        if (ndims == 2 and onehotencoded):
            surfa_image = sf.Slice(np_image.squeeze(), geometry=geom, labels=labels, metadata=original_framedimage.metadata)
        else:
            surfa_image = sf.Volume(np_image.squeeze(), geometry=geom, labels=labels, metadata=original_framedimage.metadata)

        # resample to original input image space
        if (resample):
            surfa_image = surfa_image.resample_like(original_framedimage.geom, method=method)

        # surfa.image.framed.reorient() is not yet implemented for 2D data
        # the 'orientation' should be the original input image orientation, so reorienting should not be needed if resample=True
        if (ndims == 3 and orientation is not None):
            surfa_image = surfa_image.reorient(orientation, copy=False, inplace=True)
    else:
        orientation = "RAS" if (orientation is None) else orientation
        rotation_matrix = sf.transform.orientation.orientation_to_rotation_matrix(orientation)
        geom = geom if (geom is not None) else sf.transform.geometry.ImageGeometry(shape=np_image.shape[:-1], voxsize=1, rotation=rotation_matrix)
        if (ndims == 2 and onehotencoded):
            surfa_image = sf.Slice(np_image.squeeze(), labels=labels, geometry=geom)
        else:
            surfa_image = sf.Volume(np_image.squeeze(), labels=labels, geometry=geom)            

    # if we are not resampling back to original image space,
    # simply put image in target_im_geom space if it is provided
    if ((target_im_geom is not None) and (not resample)):
        surfa_image = surfa_image.new(surfa_image.data, geometry=target_im_geom)
    
    surfa_image.save(output_file)


def remap_labels(labels, mapping, return_counts=False):
    if (return_counts):
        vox_counts = []

    remapped_labels = torch.zeros_like(labels)
    for old_label, new_label in mapping.items():
        label_mask = labels == old_label
        remapped_labels[label_mask] = new_label
        if (return_counts and new_label != 0):  # exclude background
            if (label_mask.is_cuda):
                label_mask = label_mask.cpu()
            vox_counts.append(np.count_nonzero(label_mask))

    if (return_counts):
        return remapped_labels, vox_counts
    else:
        return remapped_labels


def onehot(labels, num_classes, device=None):
    """
    One-hot encode a tensor of integer labels.

    Args:
        labels (torch.Tensor): A tensor of integer labels [N, (1,) H, W(, D)]
        num_classes (int): The number of classes.
        device (torch.device, optional): The desired device (CPU or GPU). 
                                         If None, defaults to the device of the 'labels' tensor.

    Returns:
        torch.Tensor: A one-hot encoded tensor [N, num_classes, H, W(, D)]
    """
    if device is None:
        device = labels.device 
    onehot_labels = torch.eye(num_classes, device=device)[labels.squeeze(1)]
    if (onehot_labels.ndim == 5):  # 3D
        return onehot_labels.permute(0, 4, 1, 2, 3)
    elif (onehot_labels.ndim == 4): # 2D
        return onehot_labels.permute(0, 3, 1, 2)
    else:
        raise ValueError("Onehot encoded label is expected to be 4 or 5 dimensions")


def bbox(image, labels, verbose=False):
    """
    calculate label bounding box in the image

    Args:
        image (torch.Tensor):
        labels (list):

    Returns:
        lowerbound (1d numpy array), upperbound (1d numpy array)
    """

    image = image.squeeze(0)
    
    # binarize the image with labels given
    mask = torch.zeros(image.shape, dtype=int, device=image.device)
    for label in (labels):
        label_indices = (image == label)
        mask[label_indices] = 1

    # calculate bounding box coordinates
    lowerbound = torch.zeros(image.ndim, dtype=int, device=image.device)
    upperbound = torch.zeros(image.ndim, dtype=int, device=image.device)
    coords = torch.where(mask == 1)
    for dim, coord in enumerate(coords):
        lowerbound[dim] = torch.min(coord)
        upperbound[dim] = torch.max(coord)

    # leave some rooms
    lowerbound = lowerbound - 1
    upperbound = upperbound + 1
    if (verbose):
        logging.debug(f"bbox(): {lowerbound.tolist()} - {upperbound.tolist()}")
        
    return lowerbound, upperbound
    

def centroid(label, verbose=False):
    """
    calculate centroid for given label image

    Args:
        label (numpy.narray): input label image array

    Returns:
        tuple: Coordinates of the center point for given label image
    """

    # binarize the label image
    #mask = torch.zeros(label.shape, dtype=int, device=label.device)
    #mask[label > 0] = 1

    # calculate bounding box coordinates
    lowerbound = torch.zeros(label.ndim, dtype=int, device=label.device)
    upperbound = torch.zeros(label.ndim, dtype=int, device=label.device)
    if (torch.any(label > 0)):  # check if any of the labels exist
        coords = torch.where(label > 0)
        for dim, coord in enumerate(coords):
            lowerbound[dim] = torch.min(coord)
            upperbound[dim] = torch.max(coord)

    centroid = lowerbound + (upperbound - lowerbound)/2
    if (verbose):
        logging.debug(f"label centroid: {lowerbound.tolist()} - {upperbound.tolist()}, centroid: {centroid.tolist()}")    
    
    return centroid.int()


# yield sampled data in this order: n_batch, images, labels, [priors,] dataset_indices
def DataGenerator(dataloader, device=None, return_priors=True, **kwargs):
    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while (True):
        for n_batch, batched_sample in enumerate(dataloader):
            # images/labels are batched, dataset_indices is list of index to dataset entry
            dataset_indices = batched_sample.pop(0)
            sample = [n_batch] + batched_sample  # n_batch, images, labels, [priors,]
            
            if (return_priors and len(sample) == 3):
                # torch.utils.data.DataLoader can't return NoneType, make an empty tensor with 0 elements
                sample.append(torch.empty(0, *sample[2].shape[1:], device=device))
            elif (not return_priors and len(sample) > 3):
                # remove tensors except image and onehot encoded label
                del(sample[3:])

            # insert dataset_indices at last position
            sample.append(dataset_indices)

            yield sample


# https://pytorch.org/docs/stable/notes/randomness.html
# ??? todo: for multi-process dataloader, use worker_init_fn() and generator to preserve reproducibility
def set_deterministic_training(seed=42):
    logging.info("set deterministic training")
    logging.info("\tSet Random Seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if (torch.cuda.is_available()):
        logging.info("\tControl CUDA Randomness")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html
    logging.info("\tUse Deterministic Algorithms")
    # operations that do not have a deterministic implementation will throw a warning instead of an error
    torch.use_deterministic_algorithms(True, warn_only=True)
    # fill the uninitialized memory with a known value
    # torch.utils.deterministic.fill_uninitialized_memory = True  # default if torch.use_deterministic_algorithms(True) is set
    # for CUDA version >= 10.2, see https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"  # or CUBLAS_WORKSPACE_CONFIG=:16:8


def print_vm_peak():
    """
    Return the VM peak of the running process. This is only available on linux platforms.
    """
    if platform.system() != 'Linux':
        return None

    procstat = os.path.join('/proc', str(os.getpid()), 'status')
    fp = open(procstat, 'r')
    lines = fp.readlines()
    for line in lines:
        strs = line.split()
        if (strs[0] == "VmPeak:"):
            vmpeak = f"{strs[1]} {strs[2]}"
            return vmpeak
            """
            logging.info(f"VmPeak: {strs[1]} {strs[2]}")
            break
            """


def gpu_report(gpu_index):
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True)
    result = result.stdout.splitlines()[gpu_index]
    index, name, utilization, mem_used, mem_total, temp = result.split(",")
    logging.info(f"GPU {index}: {name}" + f"  - Utilization: {utilization}" + f"  - Memory Usage: {mem_used} / {mem_total}" + f"  - Temperature: {temp}")


def config_logger(logfile=None, mode='a', level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s"):
    """
    Setup python logging module.
    - Log messages to sys.stderr only if there is no logfile specified.
    - append to exisiting logfile by default, 'mode' is as described in https://docs.python.org/3/library/functions.html#filemodes
    """
    if (logfile is not None):
        logdir = os.path.dirname(logfile)
        if (logdir):
            os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(filename=logfile, filemode=mode, level=level, format=format)

    if (logfile is None):
        # define a Handler which writes messages to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(level)
        # set a format which is simpler for console use
        formatter = logging.Formatter(consolefmt)
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)


def get_class(qualified_class_name):
    """
    retrieve python class from given module
    qualified_class_name is expected to be a python fully qualified class name
    """
    module = '.'.join(qualified_class_name.split('.')[:-1])
    assert (module is not None), f"No python module found"
    
    class_name = qualified_class_name.split('.')[-1]

    py_module = importlib.import_module(module)
    py_class = getattr(py_module, class_name, None)
    assert (py_class is not None), f"Couldn't get attr '{class_name}' from {py_module}"

    return py_class


def remove_duplicates(inlist, lowercase=True):
    if (inlist is None):
        return None

    # remove duplicates but keep the order
    outlist = []
    for t in inlist:
        t_lower = t.lower()
        if t_lower not in outlist:
            outlist.append(t_lower)

    return outlist


def unique_unsorted(arr):
    # get unique elements and their first appearance indices
    unique_elements, first_indices = np.unique(arr, return_index=True)

    # sort the indices to preserve original order
    sorted_indices = np.sort(first_indices)

    # get unique elements in original order
    unique_in_order = unique_elements[sorted_indices]

    return unique_in_order


def write_csv(fcsv, data, header=[]):
    import csv

    # the first row is the header information
    with open(fcsv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(header)
        writer.writerows(data)


def write_volume_stats(fstats, vox_counts, volumes, labels, etiv=None):
    with open(fstats, 'w') as fp:
        fp.write('# Volumetric Stats\n')
        if (etiv is not None):
            fp.write('# Measure EstimatedTotalIntraCranialVol, eTIV, Estimated ' + \
                       f'Total Intracranial Volume, {etiv:.6f}, mm^3\n')
        fp.write(f'# NRows {len(labels)}\n')
        fp.write('# NTableCols 5\n')
        fp.write('# ColHeaders  Index  SegId  NVoxels      Volume_mm3      StructName\n')
        for j, (id, name) in enumerate(labels):
            for i in range(len(volumes)):
                fp.write("            %4d %6d    %7d %17.4f     %s\n" % (j+1, id, vox_counts[i][j], volumes[i][j], name))


def mask_volume(volume, mask):
    """
    Mask a volume with the given mask. The volume and mask are numpy array with the same shapes.
    The regions outside the mask are marked as 0.
    """

    assert (mask.shape == volume.shape), f"the 'volume' and 'mask' need to have same shape"

    new_volume = volume.squeeze(0)

    # replace values outside of mask by padding_char
    mask_to_apply = mask.squeeze(0) > 0
    masking_value = 0
    new_volume[np.logical_not(mask_to_apply)] = masking_value

    return torch.tensor(new_volume[None, ...])


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


def find_closest_number_divisible_by_m(n, m, answer_type='lower'):
    """Return the closest integer to n that is divisible by m. answer_type can either be 'closer', 'lower' (only returns
    values lower than n), or 'higher' (only returns values higher than m)."""
    if n % m == 0:
        return n
    else:
        q = int(n / m)
        lower = q * m
        higher = (q + 1) * m
        if answer_type == 'lower':
            return lower
        elif answer_type == 'higher':
            return higher
        elif answer_type == 'closer':
            return lower if (n - lower) < (higher - n) else higher
        else:
            raise Exception('answer_type should be lower, higher, or closer, had : %s' % answer_type)


def get_largest_connected_component(mask, structure=None):
    """Function to get the largest connected component for a given input.
    :param mask: a 2d or 3d label map of boolean type.
    :param structure: numpy array defining the connectivity.
    """
    from scipy.ndimage import label as scipy_label
    
    components, n_components = scipy_label(mask, structure)
    return components == np.argmax(np.bincount(components.flat)[1:]) + 1 if n_components > 0 else mask.copy()
