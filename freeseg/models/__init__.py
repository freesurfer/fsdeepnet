from .unet import UNet

import logging

def model_print(model, logger=logging):
    logger.info(model)


# model_summary_torchinfo(model, (1, 1, 160, 160, 160))
def model_summary_torchinfo(model, input_size):
    from torchinfo import summary
    summary(model, input_size=input_size)


def model_summary(model, input_size, logger=logging, device=None, debug=False):
    import torch
    import torch.nn as nn
    
    def register_hook(module, name=''):
        def forward_pre_hook(module, input):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(forward_hook_pre_summary)
            m_key = f"{name}.{class_name}-{module_idx+1}"
            forward_hook_pre_summary[m_key] = {
                "input_shape": list(input[0].size()),
            }
            
        def forward_hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(forward_hook_summary)
            m_key = f"{name}.{class_name}-{module_idx+1}"

            param_sizes = []
            for nm, param in module.named_parameters():
                if (param.requires_grad):
                    datasize = param.data.size()
                    datasize_list = list(datasize)
                    param_sizes.append(f"{nm:s}:{str(datasize_list):s}")

            forward_hook_summary.append({
                "name": m_key,
                "input_shape":  list(input[0].size()),
                "output_shape": list(output.size()),
                "nb_params":    sum(p.numel() for p in module.parameters()),
                "param_sizes":  ", ".join(param_sizes)
            })

            
        if isinstance(module, (list, tuple)):
            it = iter(module)
            # iterate through the list two elements at a time
            for nm, mod in zip(it, it):
                name += nm if len(name) == 0 else "." + nm
                register_hook(mod, name)
        else:
            chld_iter = module.named_children()
            chld_count = len(list(chld_iter))
            if (chld_count == 0):
                #forward_pre_hooks.append(module.register_forward_pre_hook(forward_pre_hook))
                forward_hooks.append(module.register_forward_hook(forward_hook))                
                if (debug):
                    class_name = str(module.__class__).split(".")[-1].split("'")[0]
                    logging.info(f"\t{name:25s} {str(module):80s} class_name={class_name}")
            else:
                for idx, chld in enumerate(module.named_children()):
                    register_hook(chld, name)


    forward_hook_pre_summary, forward_hook_summary = {}, []
    forward_pre_hooks, forward_hooks = [], []  # save the hook handlers registered

    if (debug):
        logger.info("<<< REGISTER FORWARD HOOKS >>>")
        logger.info("------------------------------------------------------------------------------------------------------------------------------------------")
    register_hook(model)
    if (debug):
        logger.info("------------------------------------------------------------------------------------------------------------------------------------------")    

    if (device is None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    with torch.no_grad():
        model(torch.zeros(1, *input_size).to(device))

    # remove the hooks registered
    for h in forward_hooks:
        h.remove()
    for h in forward_pre_hooks:
        h.remove()

    """
    # forward pre-hook summary
    logger.info("<<< PRE-FORWARD HOOK SUMMARY >>>")
    logger.info("--------------------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>25}".format("Layer (type)", "Input Shape")
    logger.info(line_new)
    logger.info("============================================================================================")
    for layer in forward_hook_pre_summary:
        line_new = "{:>20}  {:>25}".format(
            layer,
            str(forward_hook_pre_summary[layer]["input_shape"]),
        )
        logger.info(line_new)
    logger.info("============================================================================================")
    logger.info("--------------------------------------------------------------------------------------------")
    """

    # forward hook summary
    logger.info("<<< FORWARD HOOK SUMMARY >>>")
    logger.info("-----------------------------------------------------------------------------------------------------------------------------------------")
    line_new = "{:<35}  {:>25}  {:>25}  {:>10}  {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #", "Param Size")
    logger.info(line_new)
    logger.info("=========================================================================================================================================")
    total_params = 0
    for forward_hook in forward_hook_summary:
        layer = forward_hook["name"]
        line_new = "{:<35}  {:>25}  {:>25}  {:>10}  {:<15}".format(
            layer,
            str(forward_hook["input_shape"]),
            str(forward_hook["output_shape"]),
            "{0:,}".format(forward_hook["nb_params"]),
            forward_hook["param_sizes"]
        )
        total_params += forward_hook["nb_params"]
        logger.info(line_new)
    logger.info("=========================================================================================================================================")
    logger.info(f"Total params: {total_params:,}")
    logger.info("-----------------------------------------------------------------------------------------------------------------------------------------")


def model_parameters(model, logger=logging):
    logger.info("<<< NETWORK PARAMETERS >>>")
    for name, param in model.named_parameters():
        trainable = False
        if param.requires_grad:
            trainable = True
        datasize = param.data.size()
        datasize_list = list(datasize)                
        logger.info(f"\t{name:30s}: {str(datasize_list):20s}, {param.numel():10,d}  trainable={trainable}")
    total_params = sum(param.numel() for param in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
