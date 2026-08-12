import logging
import torch
import torch.nn as nn

from fsdeepnet.apps.pglands.models.unet3d import UNet3D

__unetver__ = 1

class UNet(UNet3D):

    # constructor
    def __init__(self, model_arch_dict):
        self._model_arch_dict = {}
        
        # set network defaults
        self._setdefault_arch_dict()
        # update network parameters with user input
        self._update_arch_dict(model_arch_dict)

        super().__init__(**self._model_arch_dict)


    # forward method
    def forward(self, x, **kwargs):
        x = super().forward(x)
        return x


    # set network defaults
    def _setdefault_arch_dict(self):
        self._model_arch_dict["__unetver__"] = __unetver__

        self._model_arch_dict["in_channels"] = 1                  # no. of input channels
        self._model_arch_dict["conv_sz"] = 3                      # conv window size
        self._model_arch_dict["pool_sz"] = 2                      # pooling window size
        self._model_arch_dict["n_convs_per_block"] = 2            # no. of layers per block
        self._model_arch_dict["nb_levels"] = 4                    # no. of enc/dec levels
        self._model_arch_dict["n_starting_features"] = 24         # no. of starting features
        self._model_arch_dict["normalization_type"] = 'Instance'  # normalization func.
        self._model_arch_dict["activation_function"] = 'ELU'      # activation func.
        self._model_arch_dict["pooling_type"] = 'MaxPool'         # pooling func.
        self._model_arch_dict["residuals"] = False                # flag for residual conns.
        self._model_arch_dict["skip"] = True                      # flag for skip conns.
        self._model_arch_dict["X"] = 3                            # no. of spatial dims.        


    # update network parameters with user input
    def _update_arch_dict(self, model_arch_dict):
        # `ndims` -> `X`
        if ("ndims" in model_arch_dict):
            self._model_arch_dict["X"] = model_arch_dict["ndims"]
            del(model_arch_dict["ndims"])

        # update self._model_arch_dict
        for k in (model_arch_dict.keys()):
            self._model_arch_dict[k] = model_arch_dict[k]

        """
        # verify network output channels
        assert ("out_channels" in self._model_arch_dict), \
            "Use model configurable `out_channels` to specify network output channels"
        """


    @property
    def arch_dict(self):
        return self._model_arch_dict

        
