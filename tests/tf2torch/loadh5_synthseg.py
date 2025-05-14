import sys
import h5py
import numpy as np
import torch

from freeseg.config import Config
from freeseg.checkpoint import Checkpoint

"""
Usage: fspython loadh5_synthseg.py <synthseg h5 model file> <freeseg config file for synthseg> <converted freeseg pytorch pth model file>
Example:
  loadh5_synthseg.py /autofs/cluster/scratch_wednesday/yh887/FS8.0.0-copy/models/synthseg_2.0.h5 
                     /autofs/cluster/scratch_wednesday/yh887/training.freeseg/pitgland/configs/config_synthseg.yaml 
                     /autofs/cluster/scratch_wednesday/yh887/training.freeseg/synthseg/synthseg-converted.pth
"""
model_file   = sys.argv[1]
config_file  = sys.argv[2]
model_saveas = sys.argv[3]

keys_model_state_dict = {
"unet_conv_downarm_0_0/unet_conv_downarm_0_0/kernel:0"  : "encoder.0.0.convs.0.0.weight", 
"unet_conv_downarm_0_0/unet_conv_downarm_0_0/bias:0"    : "encoder.0.0.convs.0.0.bias", 
"unet_conv_downarm_0_1/unet_conv_downarm_0_1/kernel:0"  : "encoder.0.0.convs.1.0.weight", 
"unet_conv_downarm_0_1/unet_conv_downarm_0_1/bias:0"    : "encoder.0.0.convs.1.0.bias", 
"unet_bn_down_0/unet_bn_down_0/gamma:0"                 : "encoder.0.1.weight", 
"unet_bn_down_0/unet_bn_down_0/beta:0"                  : "encoder.0.1.bias",
"unet_bn_down_0/unet_bn_down_0/moving_mean:0"           : "encoder.0.1.running_mean",
"unet_bn_down_0/unet_bn_down_0/moving_variance:0"       : "encoder.0.1.running_var",

"unet_conv_downarm_1_0/unet_conv_downarm_1_0/kernel:0"  : "encoder.1.0.convs.0.0.weight", 
"unet_conv_downarm_1_0/unet_conv_downarm_1_0/bias:0"    : "encoder.1.0.convs.0.0.bias", 
"unet_conv_downarm_1_1/unet_conv_downarm_1_1/kernel:0"  : "encoder.1.0.convs.1.0.weight", 
"unet_conv_downarm_1_1/unet_conv_downarm_1_1/bias:0"    : "encoder.1.0.convs.1.0.bias", 
"unet_bn_down_1/unet_bn_down_1/gamma:0"                 : "encoder.1.1.weight", 
"unet_bn_down_1/unet_bn_down_1/beta:0"                  : "encoder.1.1.bias",
"unet_bn_down_1/unet_bn_down_1/moving_mean:0"           : "encoder.1.1.running_mean",
"unet_bn_down_1/unet_bn_down_1/moving_variance:0"       : "encoder.1.1.running_var",


"unet_conv_downarm_2_0/unet_conv_downarm_2_0/kernel:0"  : "encoder.2.0.convs.0.0.weight", 
"unet_conv_downarm_2_0/unet_conv_downarm_2_0/bias:0"    : "encoder.2.0.convs.0.0.bias", 
"unet_conv_downarm_2_1/unet_conv_downarm_2_1/kernel:0"  : "encoder.2.0.convs.1.0.weight", 
"unet_conv_downarm_2_1/unet_conv_downarm_2_1/bias:0"    : "encoder.2.0.convs.1.0.bias", 
"unet_bn_down_2/unet_bn_down_2/gamma:0"                 : "encoder.2.1.weight", 
"unet_bn_down_2/unet_bn_down_2/beta:0"                  : "encoder.2.1.bias",
"unet_bn_down_2/unet_bn_down_2/moving_mean:0"           : "encoder.2.1.running_mean",
"unet_bn_down_2/unet_bn_down_2/moving_variance:0"       : "encoder.2.1.running_var",


"unet_conv_downarm_3_0/unet_conv_downarm_3_0/kernel:0"  : "encoder.3.0.convs.0.0.weight", 
"unet_conv_downarm_3_0/unet_conv_downarm_3_0/bias:0"    : "encoder.3.0.convs.0.0.bias", 
"unet_conv_downarm_3_1/unet_conv_downarm_3_1/kernel:0"  : "encoder.3.0.convs.1.0.weight", 
"unet_conv_downarm_3_1/unet_conv_downarm_3_1/bias:0"    : "encoder.3.0.convs.1.0.bias", 
"unet_bn_down_3/unet_bn_down_3/gamma:0"                 : "encoder.3.1.weight", 
"unet_bn_down_3/unet_bn_down_3/beta:0"                  : "encoder.3.1.bias",
"unet_bn_down_3/unet_bn_down_3/moving_mean:0"           : "encoder.3.1.running_mean",
"unet_bn_down_3/unet_bn_down_3/moving_variance:0"       : "encoder.3.1.running_var",


"unet_conv_downarm_4_0/unet_conv_downarm_4_0/kernel:0"  : "bottleneck.0.convs.0.0.weight", 
"unet_conv_downarm_4_0/unet_conv_downarm_4_0/bias:0"    : "bottleneck.0.convs.0.0.bias", 
"unet_conv_downarm_4_1/unet_conv_downarm_4_1/kernel:0"  : "bottleneck.0.convs.1.0.weight", 
"unet_conv_downarm_4_1/unet_conv_downarm_4_1/bias:0"    : "bottleneck.0.convs.1.0.bias", 
"unet_bn_down_4/unet_bn_down_4/gamma:0"                 : "bottleneck.1.weight", 
"unet_bn_down_4/unet_bn_down_4/beta:0"                  : "bottleneck.1.bias",
"unet_bn_down_4/unet_bn_down_4/moving_mean:0"           : "bottleneck.1.running_mean",
"unet_bn_down_4/unet_bn_down_4/moving_variance:0"       : "bottleneck.1.running_var",


"unet_conv_uparm_5_0/unet_conv_uparm_5_0/kernel:0"      : "decoder.0.1.convs.0.0.weight", 
"unet_conv_uparm_5_0/unet_conv_uparm_5_0/bias:0"        : "decoder.0.1.convs.0.0.bias", 
"unet_conv_uparm_5_1/unet_conv_uparm_5_1/kernel:0"      : "decoder.0.1.convs.1.0.weight", 
"unet_conv_uparm_5_1/unet_conv_uparm_5_1/bias:0"        : "decoder.0.1.convs.1.0.bias", 
"unet_bn_up_0/unet_bn_up_0/gamma:0"                     : "decoder.0.2.weight", 
"unet_bn_up_0/unet_bn_up_0/beta:0"                      : "decoder.0.2.bias",
"unet_bn_up_0/unet_bn_up_0/moving_mean:0"               : "decoder.0.2.running_mean",
"unet_bn_up_0/unet_bn_up_0/moving_variance:0"           : "decoder.0.2.running_var",  

"unet_conv_uparm_6_0/unet_conv_uparm_6_0/kernel:0"      : "decoder.1.1.convs.0.0.weight", 
"unet_conv_uparm_6_0/unet_conv_uparm_6_0/bias:0"        : "decoder.1.1.convs.0.0.bias", 
"unet_conv_uparm_6_1/unet_conv_uparm_6_1/kernel:0"      : "decoder.1.1.convs.1.0.weight", 
"unet_conv_uparm_6_1/unet_conv_uparm_6_1/bias:0"        : "decoder.1.1.convs.1.0.bias", 
"unet_bn_up_1/unet_bn_up_1/gamma:0"                     : "decoder.1.2.weight", 
"unet_bn_up_1/unet_bn_up_1/beta:0"                      : "decoder.1.2.bias",
"unet_bn_up_1/unet_bn_up_1/moving_mean:0"               : "decoder.1.2.running_mean",
"unet_bn_up_1/unet_bn_up_1/moving_variance:0"           : "decoder.1.2.running_var",

"unet_conv_uparm_7_0/unet_conv_uparm_7_0/kernel:0"      : "decoder.2.1.convs.0.0.weight", 
"unet_conv_uparm_7_0/unet_conv_uparm_7_0/bias:0"        : "decoder.2.1.convs.0.0.bias", 
"unet_conv_uparm_7_1/unet_conv_uparm_7_1/kernel:0"      : "decoder.2.1.convs.1.0.weight", 
"unet_conv_uparm_7_1/unet_conv_uparm_7_1/bias:0"        : "decoder.2.1.convs.1.0.bias", 
"unet_bn_up_2/unet_bn_up_2/gamma:0"                     : "decoder.2.2.weight", 
"unet_bn_up_2/unet_bn_up_2/beta:0"                      : "decoder.2.2.bias",
"unet_bn_up_2/unet_bn_up_2/moving_mean:0"               : "decoder.2.2.running_mean",
"unet_bn_up_2/unet_bn_up_2/moving_variance:0"           : "decoder.2.2.running_var",    

"unet_conv_uparm_8_0/unet_conv_uparm_8_0/kernel:0"      : "decoder.3.1.convs.0.0.weight", 
"unet_conv_uparm_8_0/unet_conv_uparm_8_0/bias:0"        : "decoder.3.1.convs.0.0.bias", 
"unet_conv_uparm_8_1/unet_conv_uparm_8_1/kernel:0"      : "decoder.3.1.convs.1.0.weight", 
"unet_conv_uparm_8_1/unet_conv_uparm_8_1/bias:0"        : "decoder.3.1.convs.1.0.bias", 
"unet_bn_up_3/unet_bn_up_3/gamma:0"                     : "decoder.3.2.weight", 
"unet_bn_up_3/unet_bn_up_3/beta:0"                      : "decoder.3.2.bias",
"unet_bn_up_3/unet_bn_up_3/moving_mean:0"               : "decoder.3.2.running_mean",
"unet_bn_up_3/unet_bn_up_3/moving_variance:0"           : "decoder.3.2.running_var",

"unet_likelihood/unet_likelihood/kernel:0"              : "classifier.weight", 
"unet_likelihood/unet_likelihood/bias:0"                : "classifier.bias"
}

config = Config.load(config_file)

# create training/validation dataset with the desired augmentations specified
labels_segmentation = sorted(config["dataset"]["expected_classes"])
label_mapping = {label:i for i, label in enumerate(labels_segmentation)}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}
#config["dataset"]["label_mapping"] = label_mapping
train_dataset_dict = {
    "batch_size": config["training"]["batch_size"],
    "segmentation_labels" : labels_segmentation,
    "label_mapping": label_mapping,
    "inverse_label_mapping": inverse_label_mapping,
    "crop_size"    : config["preprocessing"]["crop_size"],    
    "num_channels" : 1,
}
model_arch_dict = config["model"]
model_arch_dict["num_channels"] = config["dataset"]["expected_num_channels"]
model_arch_dict["nb_labels"] = len(config["dataset"]["expected_classes"])
model_arch_dict["bn_track_running_stats"] = True
#model_arch_dict["add_priors"] = train_dataset_dict.get("priors", False)

checkpoint = Checkpoint(model_arch_dict=model_arch_dict, train_dataset_dict=train_dataset_dict, label_lookup=None) 
model_state_dict = {}

def preview_hdf5(name, obj):
    if (isinstance(obj, h5py.Dataset)):
        dict_key = keys_model_state_dict.get(name)
        if (dict_key is None):
            print(f"Dataset: {name} {obj.shape} - skip")
        else:
            print(f"Dataset: {name} {obj.shape} {dict_key}")

            numpy_array = np.zeros(obj.shape, dtype=obj.dtype)
            obj.read_direct(numpy_array)
            torch_tensor = torch.tensor(numpy_array)
            if (torch_tensor.ndim == 5):
                torch_tensor = torch_tensor.permute(4, 3, 0, 1, 2)
            model_state_dict[dict_key] = torch_tensor


with h5py.File(model_file, 'r') as f:
    f.visititems(preview_hdf5)

# model_dict
model_dict = {"model_state_dict" : model_state_dict}
checkpoint.save(model_saveas, model_dict)

print(f"Synthseg model saved as {model_saveas}")

def preview_hdf5_0(name, obj):
    if (isinstance(obj, h5py.Group)):
        print(f"Group:   {obj.name}")
    elif (isinstance(obj, h5py.Dataset)):
        print(f"Dataset: {name} {obj.shape} {keys_model_state_dict.get(name)}")


