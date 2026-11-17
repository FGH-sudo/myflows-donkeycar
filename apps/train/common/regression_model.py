"""A ResNet regression head that retains a coarse spatial layout of the road."""

from MyFlows.layers.resnet import ResNet18
from MyFlows.core.device import xp
from MyFlows.utils.initializers import make_initializer
from MyFlows.ops.convolution import Flatten_Op, MaxPool2d_Op


class SpatialResNet18(ResNet18):
    """Keep the pooled feature grid before the steering/throttle head.

    Global averaging discards the feature locations; steering also depends on where a road feature lies in
    the picture. This optional head retains those locations. The established ResNet18 path stays separate.
    """

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
