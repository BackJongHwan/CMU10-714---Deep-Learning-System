"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        weight_tensor = init.kaiming_uniform(in_features, out_features)
        self.weight = Parameter(weight_tensor)
        if bias:
            bias_tensor = init.kaiming_uniform(out_features, 1)
            self.bias = Parameter(bias_tensor.reshape((1, out_features)))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias:
            out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for module in self.modules:
            out = module.forward(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch, num_class = logits.shape
        one_hot = init.one_hot(logits.shape[1], y)
        log_softmax = ops.logsoftmax(logits)
        loss_unreduced = -log_softmax * one_hot
        loss = loss_unreduced.sum() / batch
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = x.shape[0]
        feature_dim = x.shape[1]
        w = self.weight.reshape((1, feature_dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, feature_dim)).broadcast_to(x.shape)
        if self.training:
            mean_x = x.sum(axes=(0,)) / batch_size
            mean_x_broadcast = mean_x.reshape((1, feature_dim)).broadcast_to(x.shape)
            var_x = ((x - mean_x_broadcast) ** 2).sum(axes=(0,)) / batch_size
            var_x_broadcast = var_x.reshape((1, feature_dim)).broadcast_to(x.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x.data
            norm_x = (x - mean_x_broadcast) / (var_x_broadcast + self.eps) ** 0.5
            return w * norm_x + b
        else:
            mean_x_broadcast = self.running_mean.reshape((1, feature_dim)).broadcast_to(x.shape)
            var_x_broadcast = self.running_var.reshape((1, feature_dim)).broadcast_to(x.shape)
            norm_x = (x - mean_x_broadcast) / (var_x_broadcast + self.eps) ** 0.5
            return w * norm_x + b
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim))
        self.bias = Parameter(init.zeros(dim))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = list(x.shape)
        shape[-1] = 1
        mean_x = x.sum(axes=(1,)).reshape(shape).broadcast_to(x.shape) / self.dim
        var_temp = (x - mean_x) ** 2
        var_x = var_temp.sum(axes=(1,)).reshape(shape).broadcast_to(x.shape) / self.dim
        out = self.weight.broadcast_to(x.shape) * ((x - mean_x) / (var_x + self.eps) ** (1/2)) + self.bias.broadcast_to(x.shape)
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_matrix = init.randb(*x.shape, p=1 - self.p) / (1 - self.p)
            return x * drop_matrix
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
