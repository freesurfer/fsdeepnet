import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ndims=3,
        conv_size=3,
        nb_conv_per_level=1,
        use_residuals=False,
        use_batchnorm=True,
        activation="elu",
        weight_init="xavier_uniform"
    ):
        super().__init__()

        # support both 3D and 2D
        convL = getattr(nn, "Conv%dd" % ndims)
        if convL is None:
            raise ValueError(f"Unsupported number of dimensions for the Unet: {ndims}")

        if activation.lower() == "elu":
            activation_func = nn.ELU()
        elif activation.lower() == "relu":
            activation_func = nn.ReLU()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        # convolution + activation layers
        self.convs = nn.ModuleList()
        for conv in range(nb_conv_per_level):
            in_channels_conv = in_channels if (conv == 0) else out_channels
            
            module_list = nn.ModuleList()
            module_list.append(convL(in_channels_conv, out_channels, kernel_size=conv_size, padding=1))
            self.init_weight(module_list, weight_init)         

            if (conv < nb_conv_per_level - 1) or (not use_residuals):
                module_list.append(activation_func)

            self.convs.append(module_list)

        # residual block
        self.resblock = None        
        if use_residuals:
            self.resblock = nn.ModuleList()
            self.resblock.append(convL(in_channels, out_channels, kernel_size=conv_size, padding=1))
            self.init_weight(self.resblock, weight_init)                
            self.resblock.append(activation_func)

        """
        The default BatchNorm3d layer behavior is different between .train() and .eval().
        By default, during training this layer keeps running estimates of its computed mean and variance, which are then used for normalization during evaluation.
        This causes the evaluation to perform much worse than training if the data distribution of the training set and the evaluation/test set is very different.
        Set track_running_stats=False, this module does not track such statistics, and initializes statistics buffers running_mean and running_var as None.
        When these buffers are None, this module always uses batch statistics in both training and eval modes.
        """
        self.bn = (
            getattr(nn, "BatchNorm%dd" % ndims)(out_channels, track_running_stats=False) if use_batchnorm else nn.Identity()
        )


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

        # batch norm
        x = self.bn(x)

        return x


class UNet(nn.Module):
    def __init__(self, model_arch_dict):

        num_channels = model_arch_dict.get("num_channels", None)
        if (num_channels is None):
            # backward compatible - read older models with model_arch_dict["input_shape"] saved instead
            print(f"this is an older model file w/ 'input_shape' saved insted of 'num_channels'")
            input_shape = model_arch_dict["input_shape"]
            num_channels = input_shape[0]
        ndims = model_arch_dict["ndims"]
        nb_features = model_arch_dict["nb_features"]
        nb_levels = model_arch_dict["nb_levels"]
        nb_labels = model_arch_dict["nb_labels"]
        feat_mult = model_arch_dict.get("feat_mult", 1)
        conv_size = model_arch_dict.get("conv_size", 3)
        pool_size = model_arch_dict.get("pool_size", 2)
        nb_conv_per_level = model_arch_dict.get("nb_conv_per_level", 1)
        use_residuals = model_arch_dict.get("use_residuals", False)
        use_batchnorm = model_arch_dict.get("use_batchnorm", True)
        activation = model_arch_dict.get("activation", "elu")
        add_priors = model_arch_dict.get("add_priors", False)
        refine_conv = model_arch_dict.get("refine_conv", False)
        final_pred_activation = model_arch_dict.get("final_pred_activation", "softmax")
        weight_init = model_arch_dict.get("weight_init", "xavier_uniform")

        assert (weight_init == "xavier_uniform" or weight_init == "zeros"), \
            f"weight_init {weight_init} is not supported. The options are either 'xavier_uniform' or 'zeros'"

        # if we are using the codes in line 273,
        #   1. set classifier_weight_init to 'xavier_uniform'
        #   2. comment out line 258 - 260
        classifier_weight_init = weight_init
        if (add_priors):
            classifier_weight_init = "zeros"
        print(f"[INFO] UNet: encoder/decoder weight_init={weight_init}, classifier weight_init={classifier_weight_init}")

        super().__init__()

        self.add_priors = add_priors
        self.refine_conv = refine_conv
        self.final_pred_activation = final_pred_activation

        convL = getattr(nn, "Conv%dd" % ndims)
        pool = getattr(nn, "MaxPool%dd" % ndims)(kernel_size=pool_size, stride=pool_size)

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
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                    weight_init=weight_init
                )
            )
            encoder.append(pool)

            self.encoder.append(encoder)
            in_channels = nb_lvl_feats

        # Bottleneck
        nb_lvl_feats = nb_features * (feat_mult**(nb_levels - 1))
        self.bottleneck = ConvBlock(
            in_channels,
            nb_lvl_feats,
            ndims=ndims,
            conv_size=conv_size,
            nb_conv_per_level=nb_conv_per_level,
            use_residuals=use_residuals,
            use_batchnorm=use_batchnorm,
            activation=activation,
            weight_init=weight_init
        )

        # Decoder (Expansive path)
        self.decoder = nn.ModuleList()
        for level in reversed(range(nb_levels - 1)):
            in_channels = nb_lvl_feats # output channels from previous convolution level
            nb_lvl_feats = nb_features * (feat_mult**level)

            decoder = nn.ModuleList()
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
                    use_batchnorm=use_batchnorm,
                    activation=activation,
                    weight_init=weight_init
                )
            )
            
            self.decoder.append(decoder)

        # Classification layer (Compute likelihood prediction)
        self.classifier = convL(nb_features, nb_labels, kernel_size=1)
        nn.init.zeros_(self.classifier.bias)
        if (classifier_weight_init == "zeros"):
            nn.init.zeros_(self.classifier.weight)
        elif (classifier_weight_init == "xavier_uniform"):
            nn.init.xavier_uniform_(self.classifier.weight)


    def forward(self, x, priors=None):
        skip_connections = []

        # Encoder (Contracting path)
        for encoder in (self.encoder):
            x = encoder[0](x) # ConvBlock
            skip_connections.append(x)
            x = encoder[1](x) # MaxPool

        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder (Expansive path)
        for decoder in (self.decoder):
            x = decoder[0](x) # Upsample
            if self.refine_conv:
                x = decoder[1](x) # Refinement convolution
            skip_connection = skip_connections.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = decoder[-1](x) # ConvBlock
        
        # Classification layern (Compute likelihood prediction)
        x1 = x = self.classifier(x)

        """
        https://stackoverflow.com/questions/63479765/whats-the-best-way-of-checking-whether-a-torchtensor-is-empty
        To know whether a tensor is allocated (type and storage), use defined().
        To know whether an allocated tensor has zero elements, use numel()
        To know whether a tensor is allocated and whether it has zero elements, use defined() and then numel()
        """
        # here we are adding the priors to output of classification layer;
        # the 'add_prior' implementation in https://github.com/BBillot/SynthSeg/blob/master/ext/neuron/models.py#L501
        # adds priors to the softmax output, then takes softmax again
        if (self.add_priors and priors is not None and priors.numel() != 0):
            x = torch.add(x, priors)
            x1 = torch.add(x1, priors)

        # output prediction layer
        if self.final_pred_activation == 'softmax':
            x = nn.functional.softmax(x, dim=1)
        elif self.final_pred_activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        elif self.final_pred_activation == 'linear':
            pass  # No activation applied
        else:
            raise ValueError(f"Unknown final_pred_activation: {self.final_pred_activation}")

        """
        # Benjamin's add_prior implementation (https://github.com/BBillot/SynthSeg/blob/master/ext/neuron/models.py#L501)
        # if we are using this section of codes,
        #   1. set classifier_weight_init to 'xavier_uniform'
        #   2. comment out line 258 - 260
        if (self.add_priors and priors is not None and priors.numel() != 0):
            x = torch.add(x, priors)
            if self.final_pred_activation == 'softmax':
                x = nn.functional.softmax(x, dim=1)
            elif self.final_pred_activation == 'sigmoid':
                x = nn.functional.sigmoid(x)
            elif self.final_pred_activation == 'linear':
                pass  # No activation applied
            else:
                raise ValueError(f"Unknown final_pred_activation: {self.final_pred_activation}")
        """

        # also return penultimate layer output for WeightedL2Loss
        return [x, x1]
