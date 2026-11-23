"""ResNet configuration shared by road-image training, evaluation and export."""

from MyFlows.layers.resnet import BASIC_RESNETS, ResNet18, ResNet34
from MyFlows.core.device import xp
from MyFlows.utils.initializers import make_initializer
from MyFlows.ops.convolution import Flatten_Op, MaxPool2d_Op


def input_size(args, first_image) -> tuple[int, int]:
    """Use the training resize when specified, otherwise the source frame size."""
    if args.image_size:
        try:
            h, w = (int(v) for v in str(args.image_size).lower().split("x"))
        except ValueError:
            raise SystemExit(f"--image-size 应为 HxW，例如 224x224：{args.image_size}") from None
        if h < 32 or w < 32:
            raise SystemExit("--image-size 的高和宽至少为 32")
        return h, w
    return tuple(first_image.shape[:2])


class _SpatialHead:
    """Keep the pooled feature grid before the steering/throttle head."""

    def __init__(self, *args, image_h=120, image_w=160, initializer=None, **kwargs):
        super().__init__(*args, initializer=initializer, **kwargs)
        h, w = int(image_h), int(image_w)
        if self.stem_type == "imagenet":
            h, w = ((h + 1) // 2 - 3) // 2 + 1, ((w + 1) // 2 - 3) // 2 + 1
        h, w = (h + 7) // 8, (w + 7) // 8
        if min(h, w) < 1:
            raise ValueError("input is too small for the spatial regression head")
        self.pool_kernel = (min(2, h), min(2, w))
        self.grid = (max(1, h // 2), max(1, w // 2))
        fan_in = self.widths[-1] * self.grid[0] * self.grid[1]
        # Keep the registered parameter objects/names; change their shape before constructing the graph.
        self.fc_W.value = xp.asarray(make_initializer(initializer)((fan_in, self.output_dim)), dtype=self.fc_W.value.dtype)

    def forward_features(self, input_node):
        features = Flatten_Op(MaxPool2d_Op(self.spatial_features(input_node), self.pool_kernel, stride=2,
                                         backend=self.conv_backend))
        return self.dropout(features) if self.dropout is not None else features


class SpatialResNet18(_SpatialHead, ResNet18):
    """ResNet-18 with a spatial regression head."""


class SpatialResNet34(_SpatialHead, ResNet34):
    """ResNet-34 with a spatial regression head."""


def build_regression_resnet(arch, head, image_h, image_w, **kwargs):
    if head == "global":
        return BASIC_RESNETS[arch](**kwargs)
    if head == "spatial":
        model_type = {"resnet18": SpatialResNet18, "resnet34": SpatialResNet34}[arch]
        return model_type(image_h=image_h, image_w=image_w, **kwargs)
    raise ValueError("regression_head must be global or spatial")
