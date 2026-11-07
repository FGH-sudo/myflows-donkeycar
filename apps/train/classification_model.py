"""A small RGB CNN shared by CIFAR-10 system tests and the steering-class ablations."""

from MyFlows.core.device import asnumpy
from MyFlows.layers.layer import Conv2D, Dense, Dropout, Flatten, Layer, MaxPool2d
from MyFlows.ops.activation import ReLU
from MyFlows.ops.pooling import AdaptiveAvgPool2d_Op
from MyFlows.utils.initializers import make_initializer


class SmallClassifier(Layer):
    arch = "small"
    def __init__(self, classes=10, *, width=16, dropout=0., seed=1, backend="auto"):
        super().__init__("small_cnn")
        init = make_initializer("kaiming_normal", seed=seed)
        self.conv1 = Conv2D(3, width, kernel_size=3, padding=1, activation=ReLU, initializer=init, backend=backend, name="conv1")
        self.conv2 = Conv2D(width, width * 2, kernel_size=3, padding=1, activation=ReLU, initializer=init, backend=backend, name="conv2")
        self.pool1 = MaxPool2d(2, 2, backend=backend)
        self.pool2 = MaxPool2d(2, 2, backend=backend)
        self.hidden = Dense(width * 2 * 4 * 4, width * 4, activation=ReLU, initializer=init, backend=backend, name="hidden")
        self.dropout = Dropout(dropout, seed=seed + 1) if dropout else None
        self.output = Dense(width * 4, classes, initializer=init, backend=backend, name="output")
        self.sub_layers = [self.conv1, self.pool1, self.conv2, self.pool2, self.hidden, self.output]
        if self.dropout:
            self.sub_layers.append(self.dropout)
        self.params = [p for layer in self.sub_layers for p in layer.params]
        self.fc_W, self.fc_b = self.output.W, self.output.b

    def forward_features(self, x):
        x = self.pool2(self.conv2(self.pool1(self.conv1(x))))
        x = self.hidden(Flatten()(AdaptiveAvgPool2d_Op(x, (4, 4))))
        return self.dropout(x) if self.dropout else x

    def forward_head(self, features):
        return self.output(features)

    def forward(self, x):
        self._last_transfer_features = self.forward_features(x)
        return self.forward_head(self._last_transfer_features)

    def stages(self):
        return [("conv1", [self.conv1]), ("conv2", [self.conv2]), ("hidden", [self.hidden])]

    def train(self, mode=True):
        if self.dropout:
            self.dropout.train(mode)

    def eval(self):
        self.train(False)

    def torch_state(self):
        return {"conv1.weight": asnumpy(self.conv1.kernel.value), "conv1.bias": asnumpy(self.conv1.b.value),
                "conv2.weight": asnumpy(self.conv2.kernel.value), "conv2.bias": asnumpy(self.conv2.b.value),
                "hidden.weight": asnumpy(self.hidden.W.value).T.copy(), "hidden.bias": asnumpy(self.hidden.b.value).ravel(),
                "output.weight": asnumpy(self.output.W.value).T.copy(), "output.bias": asnumpy(self.output.b.value).ravel()}
