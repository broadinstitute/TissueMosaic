import torch
import lightly
import torchvision
from typing import Tuple
from pl_bolts.models.autoencoders.components import DecoderBlock, ResNetDecoder, DecoderBottleneck


def make_resnet_backbone(
        backbone_in_ch: int,
        backbone_type: str) -> torch.nn.Module:
    """ Simple wrapper """
    return make_resnet_backbone_large_resolution(
        backbone_in_ch=backbone_in_ch,
        backbone_type=backbone_type,
        pretrained=True)


def make_resnet_backbone_large_resolution(
        backbone_in_ch: int,
        backbone_type: str,
        pretrained: bool = True) -> torch.nn.Module:
    """
    This is based on torchvision standard implementation of resnet for large resolutions.
    The spatial resolution is decreased by a factor of 16.
    For example a 64x64 input is reduced to a 4x4 outputs (before torch.nn.AdaptiveAvgPool2d)
    The input resolution must be a multiple of 16. The minimum resolution of the input is 32x32.
    For example: 32, 48, 64, 80, 96, ...
    """
    if backbone_type == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained)
    elif backbone_type == 'resnet34':
        net = torchvision.models.resnet34(pretrained=pretrained)
    elif backbone_type == 'resnet50':
        net = torchvision.models.resnet50(pretrained=pretrained)
    else:
        raise Exception("Invalid enc_dec_type. Received {0}".format(backbone_type))

    first_conv_out_channels = list(net.children())[0].out_channels
    new_net = torch.nn.Sequential(
        torch.nn.Conv2d(
            backbone_in_ch,
            first_conv_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        *list(net.children())[1:-2],  # I am excluding the last (fc) layer and AveragePool2D
        torch.nn.AdaptiveAvgPool2d(1),  # adding adaptive pooling
        torch.nn.Flatten(start_dim=1)  # adding flattening
    )
    return new_net


def make_resnet_backbone_tiny_resolution(
        backbone_in_ch: int,
        backbone_type: str) -> torch.nn.Module:
    """
    This is based on lightly reimplementation of resnet for small resolutions.
    The spatial resolution is decreased by a factor of 8.
    For example a 16x16 input is reduced to a 2x2 outputs (before torch.nn.AdaptiveAvgPool2d)
    The input resolution must be a multiple of 8. The minimum resolution of the input is 16x16.
    For example: 16, 24, 32, 40, 48, ...
    """

    if backbone_type == 'resnet18':
        net = lightly.models.resnet.ResNetGenerator(name='resnet-18', num_classes=10)
    elif backbone_type == 'resnet34':
        net = lightly.models.resnet.ResNetGenerator(name='resnet-34', num_classes=10)
    elif backbone_type == 'resnet50':
        net = lightly.models.resnet.ResNetGenerator(name='resnet-50', num_classes=10)
    else:
        raise Exception("backbone_type not recognized. Received ->", backbone_type)

    first_conv_out_channels = list(net.children())[0].out_channels

    new_net = torch.nn.Sequential(
        torch.nn.Conv2d(
            backbone_in_ch,
            first_conv_out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        *list(net.children())[1:-1],  # note that I am excluding the last_fc_layer.
        torch.nn.AdaptiveAvgPool2d(1),  # adding adaptive pooling
        torch.nn.Flatten(start_dim=1)  # adding flattening
    )
    return new_net


def make_vae_encoder_backbone_from_resnet(in_channels: int, resnet_type: str):
    if resnet_type == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
    elif resnet_type == 'resnet34':
        net = torchvision.models.resnet34(pretrained=True)
    elif resnet_type == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    else:
        raise Exception("Invalid enc_dec_type. Received {0}".format(resnet_type))

    first_conv_out_channels = list(net.children())[0].out_channels
    encoder_backbone = torch.nn.Sequential(
        torch.nn.Conv2d(
            in_channels,
            first_conv_out_channels,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        ),
        *list(net.children())[1:-2],  # note that I am excluding the last (fc) layer and the average_pool2D
    )
    return encoder_backbone


def make_vae_decoder_backbone_from_resnet(resnet_type: str):
    if resnet_type == 'resnet18':
        net = ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim=1,
                            input_height=1, first_conv=True, maxpool1=True)
        backbone_dec = torch.nn.Sequential(*list(net.children())[1:-3])  # remove the first and last layer
    elif resnet_type == 'resnet34':
        net = ResNetDecoder(DecoderBlock, [3, 4, 6, 3], latent_dim=1,
                            input_height=1, first_conv=True, maxpool1=True)
        backbone_dec = torch.nn.Sequential(*list(net.children())[1:-3])  # remove the first and last layer
    elif resnet_type == 'resnet50':
        net = ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], latent_dim=1,
                            input_height=1, first_conv=True, maxpool1=True)
        backbone_dec = torch.nn.Sequential(*list(net.children())[1:-3])  # remove the first and last layer
    else:
        raise NotImplementedError

    return backbone_dec


def make_vae_encoder_backbone_from_scratch(in_channels: int, hidden_dims: Tuple[int]):
    modules = []
    ch_in = in_channels
    for h_dim in hidden_dims:
        modules.append(
            torch.nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=ch_in,
                    out_channels=h_dim,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1),
                torch.nn.BatchNorm2d(h_dim),
                torch.nn.LeakyReLU())
        )
        ch_in = h_dim
    encoder_backbone = torch.nn.Sequential(*modules)
    return encoder_backbone


def make_vae_decoder_backbone_from_scratch(hidden_dims: Tuple[int]):
    modules = []

    for i in range(len(hidden_dims) - 1):
        modules.append(
            torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    in_channels=hidden_dims[i],
                    out_channels=hidden_dims[i + 1],
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    padding=1,
                    output_padding=1),
                torch.nn.BatchNorm2d(hidden_dims[i + 1]),
                torch.nn.LeakyReLU())
        )
    decoder_backbone = torch.nn.Sequential(*modules)
    return decoder_backbone
