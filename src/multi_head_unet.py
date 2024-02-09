import segmentation_models_pytorch as smp

# from segmentation_models_pytorch.encoders import get_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from segmentation_models_pytorch.base import modules as md
import segmentation_models_pytorch.base.initialization as init
import timm


def load_checkpoint(model, cp_path, device):
    """
    load checkpoint and fix DataParallel/DistributedDataParallel
    """

    cp = torch.load(cp_path, map_location=device)
    try:
        model.load_state_dict(cp["model_state_dict"])

        print("succesfully loaded model weights")
    except:
        print("trying secondary checkpoint loading")
        state_dict = cp["model_state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        print("succesfully loaded model weights")
    return model


class TimmEncoderFixed(nn.Module):
    """
    Modified version of timm encoder.
    Original from: https://github.com/huggingface/pytorch-image-models

    """

    def __init__(
        self,
        name,
        pretrained=True,
        in_channels=3,
        depth=5,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.25,
    ):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        features = [
            x,
        ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


def get_model(
    enc="convnextv2_tiny.fcmae_ft_in22k_in1k",
    out_channels_cls=8,
    out_channels_inst=5,
    pretrained=True,
):
    depth = 4 if "next" in enc else 5
    encoder = TimmEncoderFixed(
        name=enc,
        pretrained=pretrained,
        in_channels=3,
        depth=depth,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.0,
    )

    decoder_channels = (256, 128, 64, 32, 16)[:depth]
    decoder_inst = UnetDecoder(
        encoder_channels=encoder.out_channels,
        decoder_channels=decoder_channels,
        n_blocks=len(decoder_channels),
        use_batchnorm=False,
        center=False,
        attention_type=None,
        next="next" in enc,
    )
    decoder_ct = UnetDecoder(
        encoder_channels=encoder.out_channels,
        decoder_channels=decoder_channels,
        n_blocks=len(decoder_channels),
        use_batchnorm=False,
        center=False,
        attention_type=None,
        next="next" in enc,
    )
    head_inst = smp.base.SegmentationHead(
        in_channels=decoder_inst.blocks[-1].conv2[0].out_channels,
        out_channels=out_channels_inst,  # instance channels
        activation=None,
        kernel_size=1,
    )
    head_ct = smp.base.SegmentationHead(
        in_channels=decoder_ct.blocks[-1].conv2[0].out_channels,
        out_channels=out_channels_cls,
        activation=None,
        kernel_size=1,
    )

    decoders = [decoder_inst, decoder_ct]
    heads = [head_inst, head_ct]
    model = MultiHeadModel(encoder, decoders, heads)
    return model


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU()

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        use_batchnorm=True,
        attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(
            attention_type, in_channels=in_channels + skip_channels
        )
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=False,
        attention_type=None,
        center=False,
        next=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        if next:
            blocks.append(
                DecoderBlock(out_channels[-1], 0, out_channels[-1] // 2, **kwargs)
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class MultiHeadModel(torch.nn.Module):
    def __init__(self, encoder, decoder_list, head_list):
        super(MultiHeadModel, self).__init__()
        self.encoder = nn.ModuleList([encoder])[0]
        self.decoders = nn.ModuleList(decoder_list)
        self.heads = nn.ModuleList(head_list)
        self.initialize()

    def initialize(self):
        for decoder in self.decoders:
            init.initialize_decoder(decoder)
        for head in self.heads:
            init.initialize_head(head)

    def check_input_shape(self, x):
        h, w = x.shape[-2:]
        output_stride = self.encoder.output_stride
        if h % output_stride != 0 or w % output_stride != 0:
            new_h = (
                (h // output_stride + 1) * output_stride
                if h % output_stride != 0
                else h
            )
            new_w = (
                (w // output_stride + 1) * output_stride
                if w % output_stride != 0
                else w
            )
            raise RuntimeError(
                f"Wrong input shape height={h}, width={w}. Expected image height and width "
                f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w})."
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        # self.check_input_shape(x)

        features = self.encoder(x)
        decoder_outputs = []
        for decoder in self.decoders:
            decoder_outputs.append(decoder(*features))

        masks = []
        for head, decoder_output in zip(self.heads, decoder_outputs):
            masks.append(head(decoder_output))

        return torch.cat(masks, 1)

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`
        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)
        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)
        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
