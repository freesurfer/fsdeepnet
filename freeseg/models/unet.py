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
        weightinit="xavier_uniform"
    ):
        super().__init__()

        # support both 3D and 2D
        convL = getattr(nn, "Conv%dd" % ndims)

        if convL is None:
            raise ValueError(f"Unsupported number of dimensions for the Unet: {ndims}")

        self.convs = nn.ModuleList()
        for i in range(nb_conv_per_level):
            if i == 0:
                in_channels_conv = in_channels
            else:
                in_channels_conv = out_channels

            self.convs.append(
                convL(in_channels_conv, out_channels, kernel_size=conv_size, padding=1)
            )

        self.weight_init(self.convs, weightinit)

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

        if activation.lower() == "elu":
            self.activation = nn.ELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Invalid activation function: {activation}")

        self.use_residuals = use_residuals
        self.nb_conv_per_level = nb_conv_per_level

        if use_residuals and in_channels != out_channels:
            self.residual_conv = convL(
                in_channels, out_channels, kernel_size=conv_size, padding=1
            )
            self.weight_init(self.residual_conv, weightinit)
        else:
            self.residual_conv = None


    # initialize weights/bias
    def weight_init(self, modulelist, weightinit="xavier_uniform"):
        for m in modulelist:
            nn.init.zeros_(m.bias)
            if (weightinit == "xavier_uniform"):
                nn.init.xavier_uniform_(m.weight)
            elif (weightinit == "zeros"):
                nn.init.zeros_(m.weight)
            else:
                raise ValueError(f"Invalid weightinit option: {weightinit}. It is either 'xavier_uniform' or 'zeros'.")

                
    def forward(self, x):
        residual = x

        for level, conv in enumerate(self.convs):
            x = conv(x)
            if (level < self.nb_conv_per_level - 1) or (not self.use_residuals):
                x = self.activation(
                    x
                )

        if self.use_residuals:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x += residual
            x = self.activation(x)
    
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

        super().__init__()
        self.num_channels = num_channels
        self.nb_features = nb_features
        self.nb_levels = nb_levels
        self.feat_mult = feat_mult
        self.use_residuals = use_residuals
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.add_priors = add_priors
        self.refine_conv = refine_conv
        self.final_pred_activation = final_pred_activation

        convL = getattr(nn, "Conv%dd" % ndims)
        self.pool = getattr(nn, "MaxPool%dd" % ndims)(kernel_size=pool_size, stride=pool_size)

        weightinit = "zeros" if (self.add_priors) else "xavier_uniform"

        # Encoding path
        self.encoder = nn.ModuleList()
        in_channels = self.num_channels
        for level in range(self.nb_levels):
            nb_lvl_feats = int(self.nb_features * (self.feat_mult**level))
            self.encoder.append(
                ConvBlock(
                    in_channels,
                    nb_lvl_feats,
                    ndims=ndims,
                    conv_size=conv_size,
                    nb_conv_per_level=nb_conv_per_level,
                    use_residuals=self.use_residuals,
                    use_batchnorm=self.use_batchnorm,
                    activation=self.activation,
                    weightinit=weightinit
                )
            )
            in_channels = nb_lvl_feats
        
        # Decoding path
        self.decoder = nn.ModuleList()
        for level in reversed(range(self.nb_levels - 1)):
            nb_lvl_feats = int(self.nb_features * (self.feat_mult**level))

            module_list = nn.ModuleList()
            # module_list.append(
            #     nn.Upsample(scale_factor=pool_size, mode="nearest")
            # )
            if ndims == 2:
                module_list.append(nn.Upsample(scale_factor=pool_size, mode='bilinear', align_corners=True))
            elif ndims == 3:
                module_list.append(nn.Upsample(scale_factor=pool_size, mode='trilinear', align_corners=True))


            if self.refine_conv:
                module_list.append(
                    convL(in_channels, nb_lvl_feats, kernel_size=conv_size, padding=1)
                )  # Refinement convolution
                in_channels = nb_lvl_feats

            in_feats_convblock = in_channels + nb_lvl_feats
            module_list.append(
                ConvBlock(
                    in_feats_convblock,  # YJH: concatenated channels
                    nb_lvl_feats,
                    ndims=ndims,
                    conv_size=conv_size,
                    nb_conv_per_level=nb_conv_per_level,
                    use_residuals=self.use_residuals,
                    use_batchnorm=self.use_batchnorm,
                    activation=self.activation,
                    weightinit=weightinit
                )
            )
            
            self.decoder.append(module_list)
            in_channels = nb_lvl_feats

        # Classification layer
        self.classifier = convL(self.nb_features, nb_labels, kernel_size=1)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        

    def forward(self, x, priors=None):
        skip_connections = []

        # Encoding path
        for level in range(self.nb_levels):
            x = self.encoder[level](x)
            if level < self.nb_levels - 1:
                skip_connections.append(x)
                x = self.pool(x)
        
        # Decoding path
        for level in range(self.nb_levels - 1):
            n = 0 # keep track of decoder module index
            x = self.decoder[level][n](x) # Upsample
            n += 1
            if self.refine_conv:
                x = self.decoder[level][n](x) # Refinement convolution
                n += 1
            skip_connection = skip_connections.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[level][n](x) # ConvBlock
        
        # Classification layer
        x1 = x = self.classifier(x)

        """
        https://stackoverflow.com/questions/63479765/whats-the-best-way-of-checking-whether-a-torchtensor-is-empty
        To know whether a tensor is allocated (type and storage), use defined().
        To know whether an allocated tensor has zero elements, use numel()
        To know whether a tensor is allocated and whether it has zero elements, use defined() and then numel()
        """
        if (self.add_priors and priors is not None and priors.numel() != 0):
            x = torch.add(x, priors)
            x1 = torch.add(x1, priors)
        

        if self.final_pred_activation == 'softmax':
            x = nn.functional.softmax(x, dim=1)
        elif self.final_pred_activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        elif self.final_pred_activation == 'linear':
            pass  # No activation applied
        else:
            raise ValueError(f"Unknown final_pred_activation: {self.final_pred_activation}")


        # also return penultimate layer output for WeightedL2Loss
        return [x, x1]
