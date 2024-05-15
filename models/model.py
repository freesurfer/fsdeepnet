import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, nb_conv_per_level=1,
        use_skip=True, use_batchnorm=True, activation="elu"
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        for i in range(nb_conv_per_level):
            if i == 0:
                in_channels_conv = in_channels
            else:
                in_channels_conv = out_channels
            self.convs.append(nn.Conv3d(in_channels_conv, out_channels, kernel_size=3, padding=1))

        self.bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()

        if activation.lower() == 'elu':
            self.activation = nn.ELU()
        elif activation.lower() == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Invalid activation function: {activation}") 

        self.use_skip = use_skip

        if use_skip and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x

        for conv in self.convs:
            x = conv(x)

        x = self.bn(x)

        if self.use_skip:
            if self.residual_conv is not None:
                residual = self.residual_conv(residual)
            x += residual

        x = self.activation(x)
        return x


class UNet3D(nn.Module):
    def __init__(
        self,
        input_shape,
        nb_features,
        nb_levels,
        nb_labels,
        feat_mult=1,  # Feature multiplier
        nb_conv_per_level=1,  # Convolutions per level
        use_skip=True,
        use_batchnorm=True,
        activation="elu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nb_features = nb_features
        self.nb_levels = nb_levels
        self.feat_mult = feat_mult
        self.use_skip = use_skip
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        # Encoding path
        self.encoder = nn.ModuleList()
        in_channels = self.input_shape[0]
        for level in range(self.nb_levels):
            nb_lvl_feats = int(self.nb_features * (self.feat_mult ** level))
            self.encoder.append(
                ConvBlock(
                    in_channels,
                    nb_lvl_feats,
                    nb_conv_per_level=nb_conv_per_level,
                    use_skip=self.use_skip,
                    use_batchnorm=self.use_batchnorm,
                    activation=self.activation,
                )
            )
            in_channels = nb_lvl_feats 

        # Decoding path
        self.decoder = nn.ModuleList()
        for level in reversed(range(self.nb_levels - 1)):
            nb_lvl_feats = int(self.nb_features * (self.feat_mult ** level))
            self.decoder.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'), # Nearest neighbor upsampling
                    nn.Conv3d(in_channels, nb_lvl_feats, kernel_size=3, padding=1), # Refinement convolution
                    ConvBlock(
                        nb_lvl_feats * 2,  # Input channels doubled due to concatenation
                        nb_lvl_feats, 
                        nb_conv_per_level=nb_conv_per_level,
                        use_skip=self.use_skip,
                        use_batchnorm=self.use_batchnorm,
                        activation=self.activation,
                    ),
                )
            )
            in_channels = nb_lvl_feats

        # Classification layer
        self.classifier = nn.Conv3d(self.nb_features, nb_labels, kernel_size=1)

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
            x = self.decoder[level][0](x)  # Upsample 
            x = self.decoder[level][1](x)  # Refinement convolution
            skip_connection = skip_connections.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[level][2](x)  # ConvBlock

        # Classification layer
        x = self.classifier(x)
        return x
