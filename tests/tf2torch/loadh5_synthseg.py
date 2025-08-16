import sys
import logging
import argparse
import h5py
import numpy as np
import torch

from freeseg.config import Config
from freeseg.checkpoint import Checkpoint
from freeseg.training import Training

"""
Usage: fspython loadh5_synthseg.py 
         --tf_model <synthseg h5 model file>
         --config <freeseg config file for synthseg>
         --torch_model_saveas <converted freeseg pytorch pth model file>
Example:
  loadh5_synthseg.py --tf_model /usr/local/freesurfer/8.0.0/models/synthseg_2.0.h5
                     --config   tests/tf2torch/synthseg_config.yaml 
                     --torch_model_saveas synthseg_2.0.pth

Notes: The following ML tools use the same UNet, these tensorflow models can be converted
       to pytorch models with similar weight/bias mapping shown in this script.
       Check each network configuration, adjust 'keys_model_state_dict' accordingly.

              mri_easyreg/mri_easyreg
              mri_synthseg/mri_synthseg
              mri_synthsr/mri_synthsr
              mri_segment_thalamic_nuclei_dti_cnn/mri_segment_thalamic_nuclei_dti_cnn
              mri_sclimbic_seg/mri_sclimbic_seg
              mri_segment_hypothalamic_subunits/mri_segment_hypothalamic_subunits
              mri_synthsr/mri_synthsr_hyperfine
              recon_all_clinical/mri_synth_surf
              mri_claustrum_seg/mri_claustrum_seg
"""
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

# Configure logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Set the log level (e.g., DEBUG, INFO, WARNING, ERROR)
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Print to terminal
    ],
)


def main():
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


    args = argument_parse()

    tf_model_file = args.tf_model
    torch_model_saveas = args.torch_model_saveas
    
    config = Config.process(args, logger=logging, require_train_outfolder=False, require_dataset_list=False)
    config, _, _, model, _, _ = Training.setup(config, preload_dataset=False, create_train_dataset=False, create_loader=False, create_model=True)

    model_arch_dict = model.arch_dict
    logging.info(f"{model_arch_dict.get('name')}:")    
    for k in model_arch_dict.keys():
        logging.info(f"    {k}: {model_arch_dict[k]}")
    logging.info("")
    
    train_dataset_dict = config["dataset"]
    label_lookup = train_dataset_dict.get("label_lookup", None)
    checkpoint = Checkpoint(model_arch_dict=model_arch_dict, train_dataset_dict=train_dataset_dict, label_lookup=None) 

    model_state_dict = {}
    with h5py.File(tf_model_file, 'r') as f:
        f.visititems(preview_hdf5)

    # model_dict
    model_dict = {"model_state_dict" : model_state_dict}
    checkpoint.save(torch_model_saveas, model_dict)

    print(f"Synthseg pytorch model saved as {torch_model_saveas}")


def argument_parse():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()

    # input/outputs
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--tf_model", type=str, help="Path to tensorflow model file")
    parser.add_argument("--torch_model_saveas", type=str, help="Path to converted pytorch checkpoint")

    # parse commandline
    args = parser.parse_args()

    return args


# execute script
if __name__ == '__main__':
    main()


def preview_hdf5_0(name, obj):
    if (isinstance(obj, h5py.Group)):
        print(f"Group:   {obj.name}")
    elif (isinstance(obj, h5py.Dataset)):
        print(f"Dataset: {name} {obj.shape} {keys_model_state_dict.get(name)}")


