import logging
import torch
import torch.nn as nn

__unetver__ = 1

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ndims=3,
        conv_size=3,
        nb_conv_per_level=1,
        use_residuals=False,
        activation="elu",
        weight_init="xavier_uniform"
    ):
        super().__init__()

        # support both 3D and 2D
        convL = getattr(nn, "Conv%dd" % ndims)
        if convL is None:
            raise ValueError(f"Unsupported number of dimensions for the Unet: {ndims}")

        if activation.lower() == "elu":
            activation = nn.ELU
        elif activation.lower() == "relu":
            activation = nn.ReLU
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        # convolution + activation layers
        self.convs = nn.ModuleList()
        for conv in range(nb_conv_per_level):
            in_channels_conv = in_channels if (conv == 0) else out_channels
            
            module_list = nn.ModuleList()
            # padding='same' pads the input so the output has the shape as the input. However, this mode doesn’t support any stride values other than 1.
            module_list.append(convL(in_channels_conv, out_channels, kernel_size=conv_size, padding='same'))
            self.init_weight(module_list, weight_init)         

            if (conv < nb_conv_per_level - 1) or (not use_residuals):
                module_list.append(activation())

            self.convs.append(module_list)

        # residual block
        self.resblock = None        
        if use_residuals:
            self.resblock = nn.ModuleList()
            self.resblock.append(convL(in_channels, out_channels, kernel_size=conv_size, padding=1))
            self.init_weight(self.resblock, weight_init)                
            self.resblock.append(activation())


    # initialize weights/bias
    def init_weight(self, modulelist, weight_init="xavier_uniform"):
        for m in modulelist:
            nn.init.zeros_(m.bias)
            if (weight_init == "xavier_uniform"):
                nn.init.xavier_uniform_(m.weight)
            elif (weight_init == "zeros"):
                nn.init.zeros_(m.weight)
            else:
                raise ValueError(f"Invalid weight_init option: {weight_init}. It is either 'xavier_uniform' or 'zeros'.")

                
    def forward(self, x):
        residual = x

        # convolution + activation
        for conv in self.convs:
            for layer in conv:
                x = layer(x)

        # residual block
        if self.resblock is not None:
            residual = self.resblock[0](residual)  # residual conv
            x += residual
            x = self.resblock[1](x)  # activation

        return x


class UNet(nn.Module):
    def __init__(self, model_arch_dict):

        self._model_arch_dict = {}

        # set network defaults
        self._setdefault_arch_dict()
        # update network parameters with user input
        self._update_arch_dict(model_arch_dict)

        num_channels = self._model_arch_dict["num_channels"]
        ndims = self._model_arch_dict["ndims"]
        nb_features = self._model_arch_dict["nb_features"]
        nb_levels = self._model_arch_dict["nb_levels"]
        nb_labels = self._model_arch_dict["nb_labels"]
        feat_mult = self._model_arch_dict["feat_mult"]
        conv_size = self._model_arch_dict["conv_size"]
        pool_size = self._model_arch_dict["pool_size"]
        nb_conv_per_level = self._model_arch_dict["nb_conv_per_level"]
        use_residuals = self._model_arch_dict["use_residuals"]
        norm = self._model_arch_dict["norm"]
        activation = self._model_arch_dict["activation"]
        add_priors = self._model_arch_dict["add_priors"]
        refine_conv = self._model_arch_dict["refine_conv"]
        final_pred_activation = self._model_arch_dict["final_pred_activation"]
        weight_init = self._model_arch_dict["weight_init"]
        track_running_stats = self._model_arch_dict["track_running_stats"]
        upsample_interpolation = self._model_arch_dict["upsample_interpolation"]
        skip_connect = self._model_arch_dict["skip_connect"]

        if (norm is not None):
            assert (norm in ["batch", "instance"]), \
                f"norm '{norm}' is not supported. The options are either 'batch' or 'instance'"
        assert (weight_init in ["xavier_uniform", "zeros"]), \
            f"weight_init '{weight_init}' is not supported. The options are either 'xavier_uniform' or 'zeros'"
        assert (upsample_interpolation in ["linear", "nearest"]), \
            f"upsample_interpolation '{upsample_interpolation}' is not supported. The options are either 'linear' or 'nearest'"
        assert (skip_connect in ["norm", "encoder"]), \
            f"skip_connect '{skip_connect}' is not supported. The options are 'norm' or 'encoder'"
        if (skip_connect == "norm"):
            assert(norm is not None), f"norm needs to be 'batch' or 'instance' for skip_connect '{skip_connect}'"
                
        classifier_weight_init = weight_init
        """
        if (add_priors):
            classifier_weight_init = "zeros"
        """
        logging.info(f"UNet: encoder/decoder norm={norm}, track_running_stats={track_running_stats}, upsample_interpolation={upsample_interpolation}, skip_connect={skip_connect}")
        logging.info(f"UNet: weight_init={weight_init}, classifier weight_init={classifier_weight_init}")

        super().__init__()

        self.add_priors = add_priors
        self.refine_conv = refine_conv
        self.final_pred_activation = final_pred_activation

        convL = getattr(nn, "Conv%dd" % ndims)
        pool = getattr(nn, "MaxPool%dd" % ndims)
        
        """
        The default BatchNorm3d layer behavior is different between .train() and .eval().
        By default, during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation.
        This causes the evaluation to perform much worse than training if the data distribution of the training set and the evaluation/test set is very different.
        Set track_running_stats=False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
        When these buffers are None, this module always uses batch statistics in both training and eval modes.
        """
        self.norm = None
        if (norm is not None and norm == 'batch'):
            self.norm = getattr(nn, "BatchNorm%dd" % ndims)
        elif (norm is not None and norm == 'instance'):
            self.norm = getattr(nn, "InstanceNorm%dd" % ndims)
            
        # Encoder (Contracting path)
        self.encoder = nn.ModuleList()
        in_channels = num_channels
        for level in range(nb_levels - 1):
            nb_lvl_feats = nb_features * (feat_mult**level)

            encoder = nn.ModuleList()
            encoder.append(
                ConvBlock(
                    in_channels,
                    nb_lvl_feats,
                    ndims=ndims,
                    conv_size=conv_size,
                    nb_conv_per_level=nb_conv_per_level,
                    use_residuals=use_residuals,
                    activation=activation,
                    weight_init=weight_init
                )
            )
            if (self.norm is not None):
                if (track_running_stats):  # match synthseg
                    encoder.append(self.norm(nb_lvl_feats, track_running_stats=True, momentum=0.99, eps=0.001, affine=True))
                else:
                    encoder.append(self.norm(nb_lvl_feats, track_running_stats=False, affine=True))
            encoder.append(pool(kernel_size=pool_size, stride=pool_size))

            self.encoder.append(encoder)
            in_channels = nb_lvl_feats

        self.skip_connect_idx = 1 if (skip_connect == "norm") else 0
                
        # Bottleneck
        nb_lvl_feats = nb_features * (feat_mult**(nb_levels - 1))
        self.bottleneck = nn.ModuleList()
        self.bottleneck.append(
            ConvBlock(
                in_channels,
                nb_lvl_feats,
                ndims=ndims,
                conv_size=conv_size,
                nb_conv_per_level=nb_conv_per_level,
                use_residuals=use_residuals,
                activation=activation,
                weight_init=weight_init
            )
        )
        if (self.norm is not None):
            if (track_running_stats):  # match synthseg
                self.bottleneck.append(self.norm(nb_lvl_feats, track_running_stats=True, momentum=0.99, eps=0.001, affine=True))
            else:
                self.bottleneck.append(self.norm(nb_lvl_feats, track_running_stats=False, affine=True))

        # Decoder (Expansive path)
        self.decoder = nn.ModuleList()
        for level in reversed(range(nb_levels - 1)):
            in_channels = nb_lvl_feats # output channels from previous convolution level
            nb_lvl_feats = nb_features * (feat_mult**level)

            decoder = nn.ModuleList()
            if (upsample_interpolation == "nearest"):
                decoder.append(nn.Upsample(scale_factor=pool_size, mode='nearest'))
            else:
                if ndims == 2:
                    decoder.append(nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True))
                elif ndims == 3:
                    decoder.append(nn.Upsample(scale_factor=pool_size, mode='trilinear', align_corners=True))

            if self.refine_conv:
                decoder.append(convL(in_channels, nb_lvl_feats, kernel_size=conv_size, padding=1))  # Refinement convolution
                in_channels = nb_lvl_feats

            # add skip connection channels (nb_lvl_feats: output channels from the encoder convolution block at the same level)
            in_feats_convblock = in_channels + nb_lvl_feats
            decoder.append(
                ConvBlock(
                    in_feats_convblock,  # YJH: concatenated channels
                    nb_lvl_feats,
                    ndims=ndims,
                    conv_size=conv_size,
                    nb_conv_per_level=nb_conv_per_level,
                    use_residuals=use_residuals,
                    activation=activation,
                    weight_init=weight_init
                )
            )
            if (self.norm is not None):
                if (track_running_stats):  # match synthseg
                    decoder.append(self.norm(nb_lvl_feats, track_running_stats=True, momentum=0.99, eps=0.001, affine=True))
                else:
                    decoder.append(self.norm(nb_lvl_feats, track_running_stats=False, affine=True))
            self.decoder.append(decoder)

        # Classification layer (Compute likelihood prediction)
        self.classifier = convL(nb_features, nb_labels, kernel_size=1)
        nn.init.zeros_(self.classifier.bias)
        if (classifier_weight_init == "zeros"):
            nn.init.zeros_(self.classifier.weight)
        elif (classifier_weight_init == "xavier_uniform"):
            nn.init.xavier_uniform_(self.classifier.weight)

        # final activation layer
        if self.final_pred_activation == 'softmax':
            self.final_activation = nn.Softmax(dim=1)
        elif self.final_pred_activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif self.final_pred_activation == 'linear':
            self.final_activation = None  # No activation applied
        else:
            raise ValueError(f"Unknown final_pred_activation: {self.final_pred_activation}")            

    def forward(self, x, priors=None):
        skip_connections = []

        # Encoder (Contracting path)
        for encoder in self.encoder:
            for idx, layer in enumerate(encoder):
                x = layer(x) # ConvBlock + batch/instance norm (optional) + maxpool
                if (idx == self.skip_connect_idx): # ConvBlock or batch/instance norm
                    skip_connections.append(x)

        # Bottleneck
        for layer in self.bottleneck:
            x = layer(x)

        # Decoder (Expansive path)
        for decoder in (self.decoder):
            idx = 0
            x = decoder[idx](x) # Upsample
            idx = idx + 1
            if self.refine_conv:
                x = decoder[idx](x) # Refinement convolution
                idx = idx + 1
            skip_connection = skip_connections.pop()
            x = torch.cat([skip_connection, x], dim=1)
            x = decoder[idx](x) # ConvBlock
            idx = idx + 1
            if (self.norm is not None):
                x = decoder[idx](x) # batch/instance norm
        
        # Classification layern (Compute likelihood prediction)
        x1 = x = self.classifier(x)

        """
        # here we are adding the priors to output of classification layer;
        # the 'add_prior' implementation in https://github.com/BBillot/SynthSeg/blob/master/ext/neuron/models.py#L501
        # adds priors to the softmax output, then takes softmax again
        # to enable this section of codes, set classifier_weight_init = "zeros" (enable codes line 127 - 130)
        if (self.add_priors and priors is not None and priors.numel() != 0):
            x = torch.add(x, priors)
            x1 = torch.add(x1, priors)
        """

        # output prediction layer
        if (self.final_activation is not None):
            x = self.final_activation(x)

        # Benjamin's add_prior implementation (https://github.com/BBillot/SynthSeg/blob/master/ext/neuron/models.py#L501)
        # priors are added to the softmax output, then takes softmax again        
        if (self.add_priors and priors is not None and priors.numel() != 0):
            #logging.debug(f"UNet.forward(): add priors")
            x = torch.add(x, priors)
            if (self.final_activation is not None):
                x = self.final_activation(x)

        # also return penultimate layer output for WeightedL2Loss
        return [x, x1]


    # set network defaults
    def _setdefault_arch_dict(self):
        self._model_arch_dict["__unetver__"] = __unetver__
        self._model_arch_dict["num_channels"] = 1
        self._model_arch_dict["ndims"] = 3
        self._model_arch_dict["nb_features"] = 24
        self._model_arch_dict["nb_levels"] = 3
        #self._model_arch_dict["nb_labels"] = ???
        self._model_arch_dict["feat_mult"] = 1
        self._model_arch_dict["conv_size"] = 3
        self._model_arch_dict["pool_size"] = 2
        self._model_arch_dict["nb_conv_per_level"] = 1
        self._model_arch_dict["use_residuals"] = False
        self._model_arch_dict["norm"] = None
        self._model_arch_dict["activation"] = "elu"
        self._model_arch_dict["add_priors"] = False
        self._model_arch_dict["refine_conv"] = False
        self._model_arch_dict["final_pred_activation"] = "softmax"
        self._model_arch_dict["weight_init"] = "xavier_uniform"
        self._model_arch_dict["track_running_stats"] = False
        self._model_arch_dict["upsample_interpolation"] = "linear"
        self._model_arch_dict["skip_connect"] = "norm"


    # update network parameters with user input
    def _update_arch_dict(self, model_arch_dict):
        num_channels = model_arch_dict.get("num_channels", None)
        if (num_channels is None):
            # backward compatible - read older models with model_arch_dict["input_shape"] saved instead
            logging.warning(f"this is an older model file w/ 'input_shape' saved instead of 'num_channels'")
            input_shape = model_arch_dict["input_shape"]
            num_channels = input_shape[0]
            del(model_arch_dict["input_shape"])
        model_arch_dict["num_channels"] = num_channels

        # backward compatibility
        # read older models before 'norm' is introduced, model_arch_dict["use_batchnorm"] is saved instead
        if ("use_batchnorm" in model_arch_dict):
            logging.warning(f"read old config entry 'use_batchnorm', use 'norm' instead to specify normalization type")
            use_batchnorm = model_arch_dict["use_batchnorm"]
            norm = "batch" if (use_batchnorm) else None
            del(model_arch_dict["use_batchnorm"])
            model_arch_dict["norm"] = norm

        # backward compatibility
        # 'skip_connect_from' is renamed to 'skip_connect'
        if ("skip_connect_from" in model_arch_dict):
            logging.warning(f"read old config entry 'skip_connect_from', it is rename to 'skip_connect'")
            model_arch_dict["skip_connect"] = model_arch_dict["skip_connect_from"]
            del(model_arch_dict["skip_connect_from"])            
            if (model_arch_dict["skip_connect"] == "batchnorm"):
                model_arch_dict["skip_connect"] = "norm"

        # backward compatibility
        # 'bn_track_running_stats' is renamed to 'track_running_stats'
        if ("bn_track_running_stats" in model_arch_dict):
            logging.warning(f"read old config entry 'bn_track_running_stats', it is rename to 'track_running_stats'")
            model_arch_dict["track_running_stats"] = model_arch_dict["bn_track_running_stats"]
            del(model_arch_dict["bn_track_running_stats"])            

        # update self._model_arch_dict
        for k in (model_arch_dict.keys()):
            self._model_arch_dict[k] = model_arch_dict[k]

        if (self._model_arch_dict["norm"] is None):
            self._model_arch_dict["skip_connect"] = "encoder"


    @property
    def arch_dict(self):
        return self._model_arch_dict
