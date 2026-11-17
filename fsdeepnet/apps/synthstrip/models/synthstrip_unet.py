from fsdeepnet.apps.synthstrip.models.stripmodel import StripModel

__unetver__ = 1

class UNet(StripModel):

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
   
        self.arch_dict["ndims"] = 3
        self.arch_dict["nb_features"] = 16
        self.arch_dict["nb_levels"] = 7
        self.arch_dict["feat_mult"] = 2
        self.arch_dict["max_features"] = 64
        self.arch_dict["nb_conv_per_level"] = 2
        self.arch_dict["max_pool"] = 2
        self.arch_dict["return_mask"] = False


    # update network parameters with user input
    def _update_arch_dict(self, model_arch_dict):
        # update self.arch_dict
        for k in (model_arch_dict.keys()):
            self.arch_dict[k] = model_arch_dict[k]



