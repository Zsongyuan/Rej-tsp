try:
    import MinkowskiEngine as ME
    from MinkowskiEngine.modules.resnet_block import BasicBlock
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

from torch import nn
import torch

class MinkowskiFeatureFusionBlock(nn.Module):
    """
    Block to fuse backbone features with text features in Minkowski space.
    """
    def __init__(self, backbone_channels, text_channels, output_channels, dimension=3):
        super(MinkowskiFeatureFusionBlock, self).__init__()
        self.conv = ME.MinkowskiConvolution(
            backbone_channels + text_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            dimension=dimension
        )
        self.norm = ME.MinkowskiBatchNorm(output_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, backbone_feats, text_feats):
        # Extract batch indices from the coordinates of backbone features
        batch_indices = backbone_feats.C[:, 0].long()  # Last column is batch index
        
        # Repeat text features for each point in the corresponding batch
        repeated_text_feats = text_feats[batch_indices]  # Use indexing to repeat text features
        
        # Combine the backbone and text features
        combined_features = torch.cat([backbone_feats.F, repeated_text_feats], dim=1)
        combined_feats = ME.SparseTensor(
            features=combined_features,
            coordinate_map_key=backbone_feats.coordinate_map_key,
            coordinate_manager=backbone_feats.coordinate_manager
        )
        
        # Convolution and normalization
        x = self.conv(combined_feats)
        x = self.norm(x)
        return self.relu(x)
    
class TR3DNeck(nn.Module):
    def __init__(self, in_channels=(64, 128, 128, 128), out_channels=128):
        super(TR3DNeck, self).__init__()
        self._init_layers(in_channels[1:], out_channels)
        self.fuse_3 = MinkowskiFeatureFusionBlock(128, 288, 128)
        self.fuse_2 = MinkowskiFeatureFusionBlock(128, 288, 128)

    def _init_layers(self, in_channels, out_channels):
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    make_up_block(in_channels[i], in_channels[i - 1], generative=True))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    make_block(in_channels[i], in_channels[i]))
            if i == 0:
                self.__setattr__(
                    f'out_block_{i}',
                    make_block(in_channels[i], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x, text_feats):
        x = x[1:]
        outs = []
        inputs = x
        x = inputs[-1]
        x = self.fuse_3(x, text_feats[:, 0])
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
                if i == 1:
                    x = self.fuse_2(x, text_feats[:, 0])
                else:
                    out = self.__getattr__(f'out_block_{i}')(x)
                # outs.append(out)
        return out


def make_block(in_channels, out_channels, kernel_size=3):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels,
                                kernel_size=kernel_size, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_down_block(in_channels, out_channels):
    return nn.Sequential(
        ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                stride=2, dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))


def make_up_block(in_channels, out_channels, generative=False):
    conv = ME.MinkowskiGenerativeConvolutionTranspose if generative \
        else ME.MinkowskiConvolutionTranspose
    return nn.Sequential(
        conv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            dimension=3),
        ME.MinkowskiBatchNorm(out_channels),
        ME.MinkowskiReLU(inplace=True))
