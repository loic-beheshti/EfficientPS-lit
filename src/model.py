import torch.nn as nn
import torch
from inplace_abn import ABN
from utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    Conv2dStaticSamePadding,
    MaxPool2dStaticSamePadding,
    iABNConv1dBlock,
    iABNSeparableConvBlock
)

class DualFPN(nn.Module):
    """
    2 way FPN for EfficientPS, currently fixed for efficientNet b-5
    """
    def __init__(self):
        super().__init__()

        self.conv1_up = iABNConv1dBlock(in_channels=40)
        self.conv2_up = iABNConv1dBlock(in_channels=64)
        self.conv3_up = iABNConv1dBlock(in_channels=176)
        self.conv4_up = iABNConv1dBlock(in_channels=2048)
        self.conv1_down = iABNConv1dBlock(in_channels=40)
        self.conv2_down = iABNConv1dBlock(in_channels=64)
        self.conv3_down = iABNConv1dBlock(in_channels=176)
        self.conv4_down = iABNConv1dBlock(in_channels=2048)

        self.upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_1 = MaxPool2dStaticSamePadding(3, 2)
        self.downsample_2 = MaxPool2dStaticSamePadding(3, 2)
        self.downsample_3 = MaxPool2dStaticSamePadding(3, 2)
        
        self.p32_conv = iABNSeparableConvBlock()
        self.p16_conv = iABNSeparableConvBlock()
        self.p8_conv = iABNSeparableConvBlock()
        self.p4_conv = iABNSeparableConvBlock()

    def forward(self, inputs):
        b1, b2, b3, b4 = inputs
        """
        left to right way fpn
        """
        b1_up = self.conv1_up(b1)
        b2_up = self.conv2_up(b2) + self.downsample_1(b1_up)
        b3_up = self.conv3_up(b3) + self.downsample_2(b2_up)
        b4_up = self.conv4_up(b4) + self.downsample_3(b3_up)

        """
        right to left fpn
        """
        b4_down = self.conv4_down(b4)
        b3_down = self.conv3_down(b3) + self.upsample_1(b4_down)
        b2_down = self.conv2_down(b2) + self.upsample_2(b3_down)
        b1_down = self.conv1_down(b1) + self.upsample_3(b2_down)

        """
        p32 to p4 extraction
        """
        p32 = self.p32_conv(b4_up + b4_down)
        p16 = self.p16_conv(b3_up + b3_down)
        p8 = self.p8_conv(b2_up + b2_down)
        p4 = self.p4_conv(b1_up + b1_down)
        #print(p32.size(), p16.size(), p8.size(), p4.size())
        return [p32, p16, p8, p4]


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block
    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above
    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args

        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = ABN(oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = ABN(oup)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = ABN(final_oup)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)

        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

class EfficientNet(nn.Module):
    """
    An EfficientNet model using the pytorch implementation mirroring the tendorrflow repo.
    This specific Efficient follows the modifications done in EfficintPS paper, 
    No squeeze and exitation, in__place batch norm and no classification layer.
    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = ABN(out_channels)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = ABN(out_channels)

    def forward(self, x):
        x = self._conv_stem(x)
        x = self._bn0(x)
        feature_maps = []

        last_x = x
        last_size = x.size()[1]
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if last_size != x.size()[1]:
                feature_maps.append(last_x)
            elif idx == len(self._blocks) - 1:
                feature_maps.append(x)
            
            last_x = x
            last_size = x.size()[1]
        
        feature_maps.append(self._bn1(self._conv_head(x)))

        return feature_maps

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)
    
    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b'+str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


class EfficientPS(nn.Module):
    def __init__(self, num_classes=30, effNet_name = 'efficientnet-b5',  **kwargs):
        super(EfficientPS, self).__init__()

        self.dualfpn = DualFPN()

        self.num_classes = num_classes
        
        blocks_args, global_params = get_model_params(effNet_name, override_params={'num_classes': num_classes}) # tbm (1.6, 2.2, 456, 0.4)

        self.backbone_net = EfficientNet(blocks_args, global_params)

    def freeze_bn(self):
        for m in self.modules():
            #if isinstance(m, nn.BatchNorm2d):
            if isinstance(m, ABN):
                m.eval()

    def forward(self, inputs):

        p1, p2, p3, p4, p5, p6, p7, p8, p9 = self.backbone_net(inputs)

        #There is a difference between the levels written in the paper and the level in the graph
        #In this example I chose to trust figure 2 and get (p3, p4, p6, p9) instead of (p2,p3,p5,p9)
        features = (p3, p4, p6, p9) 

        features = self.dualfpn(features)

        return features