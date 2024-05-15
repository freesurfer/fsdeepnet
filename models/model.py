import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, use_skip=True, use_batchnorm=True, activation="elu"
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels) if use_batchnorm else nn.Identity()
        self.activation = nn.ELU() if activation == "elu" else nn.ReLU()
        self.use_skip = use_skip

        if use_skip and in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
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
        use_skip=True,
        use_batchnorm=True,
        activation="elu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nb_features = nb_features
        self.nb_levels = nb_levels
        self.use_skip = use_skip
        self.use_batchnorm = use_batchnorm
        self.activation = activation

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)  # Define pooling layer once
        self.upsample = nn.Upsample(
            size=input_shape[1:], mode="trilinear", align_corners=True
        )  # Upsampling layer

        # Encoding path
        self.encoder = nn.ModuleList()
        in_channels = self.input_shape[0]
        for level in range(self.nb_levels):
            out_channels = self.nb_features * (2**level)
            self.encoder.append(
                ConvBlock(
                    in_channels,
                    out_channels,
                    use_skip=self.use_skip,
                    use_batchnorm=self.use_batchnorm,
                    activation=self.activation,
                )
            )
            in_channels = out_channels

        # Decoding path
        self.decoder = nn.ModuleList()
        for level in reversed(range(self.nb_levels - 1)):
            out_channels = self.nb_features * (2**level)
            self.decoder.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
                    ConvBlock(
                        out_channels * 2,
                        out_channels,
                        use_skip=self.use_skip,
                        use_batchnorm=self.use_batchnorm,
                        activation=self.activation,
                    ),
                )
            )
            in_channels = out_channels

        # Classification layer
        self.classifier = nn.Conv3d(self.nb_features, nb_labels, kernel_size=1)

        self._initialize_weights()

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
            x = self.decoder[level][0](x)  # ConvTranspose3d
            skip_connection = skip_connections.pop()
            x = torch.cat([x, skip_connection], dim=1)
            x = self.decoder[level][1](x)  # ConvBlock

        # Classification layer
        x = self.classifier(x) # logits

        # Apply softmax
        x = nn.functional.softmax(x, dim=1) # posterior probablities

        # Upsample to match input spatial dimensions
        x = self.upsample(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
