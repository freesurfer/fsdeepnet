import logging
import torch
import torch.nn as nn

from fsdeepnet.apps.pglands_seg.models.unet3d import UNet3D

__unetver__ = 1

class UNet(UNet3D):

    # constructor
    def __init__(self, model_arch_dict=None, **kwargs):
        self.arch_dict = {}
        
        # set network defaults
        self._setdefault_arch_dict()
        # update network parameters with user input
        self._update_arch_dict(model_arch_dict)

        super().__init__(**self.arch_dict)


    # forward method
    def forward(self, x, **kwargs):
        x = super().forward(x)
        return x


    # set network defaults
    def _setdefault_arch_dict(self):
        self.arch_dict["__unetver__"] = __unetver__

        self.arch_dict["in_channels"] = 1                  # no. of input channels
        self.arch_dict["conv_sz"] = 3                      # conv window size
        self.arch_dict["pool_sz"] = 2                      # pooling window size
        self.arch_dict["n_convs_per_block"] = 2            # no. of layers per block
        self.arch_dict["nb_levels"] = 4                    # no. of enc/dec levels
        self.arch_dict["n_starting_features"] = 24         # no. of starting features
        self.arch_dict["normalization_type"] = 'Instance'  # normalization func.
        self.arch_dict["activation_function"] = 'ELU'      # activation func.
        self.arch_dict["pooling_type"] = 'MaxPool'         # pooling func.
        self.arch_dict["residuals"] = False                # flag for residual conns.
        self.arch_dict["skip"] = True                      # flag for skip conns.
        self.arch_dict["X"] = 3                            # no. of spatial dims.


    # update network parameters with user input
    def _update_arch_dict(self, model_arch_dict):
        # `ndims` -> `X`
        if ("ndims" in model_arch_dict):
            self.arch_dict["X"] = model_arch_dict["ndims"]
            del(model_arch_dict["ndims"])

        # update self.arch_dict
        for k in (model_arch_dict.keys()):
            self.arch_dict[k] = model_arch_dict[k]

        """
        # verify network output channels
        assert ("out_channels" in self.arch_dict), \
            "Use model configurable `out_channels` to specify network output channels"
        """


        
