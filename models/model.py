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
            # self.convs.append(nn.Conv3d(in_channels_conv, out_channels, kernel_size=conv_size, padding=1))
            self.convs.append(
                convL(in_channels_conv, out_channels, kernel_size=conv_size, padding=1)
            )

        # self.bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        self.bn = (
            getattr(nn, "BatchNorm%dd" % ndims)(out_channels) if use_batchnorm else nn.Identity()
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
        else:
            self.residual_conv = None

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
    
        x= self.bn(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        # in_channels,
        input_shape,
        ndims,
        nb_features,
        nb_levels,
        nb_labels,
        feat_mult=1,
        conv_size=3,
        pool_size=2,
        nb_conv_per_level=1,
        use_residuals=False,
        use_batchnorm=True,
        activation="elu",
        refine_conv=False,
        final_pred_activation="softmax",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nb_features = nb_features
        self.nb_levels = nb_levels
        self.feat_mult = feat_mult
        self.use_residuals = use_residuals
        self.use_batchnorm = use_batchnorm
        self.activation = activation
        self.refine_conv = refine_conv
        self.final_pred_activation = final_pred_activation
                
        convL = getattr(nn, "Conv%dd" % ndims)
        self.pool = getattr(nn, "MaxPool%dd" % ndims)(kernel_size=pool_size, stride=pool_size)

        # Encoding path
        self.encoder = nn.ModuleList()
        in_channels = self.input_shape[0]
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
                )
            )
            
            self.decoder.append(module_list)
            in_channels = nb_lvl_feats

        # Classification layer
        self.classifier = convL(self.nb_features, nb_labels, kernel_size=1)

    def forward(self, x):
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
        x = self.classifier(x)

        if self.final_pred_activation == 'softmax':
            x = nn.functional.softmax(x, dim=1)
        elif self.final_pred_activation == 'sigmoid':
            x = nn.functional.sigmoid(x)
        elif self.final_pred_activation == 'linear':
            pass  # No activation applied
        else:
            raise ValueError(f"Unknown final_pred_activation: {self.final_pred_activation}")


        return x
