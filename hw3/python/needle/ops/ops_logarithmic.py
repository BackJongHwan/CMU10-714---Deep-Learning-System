from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_keepdims = array_api.max(Z, axis=1, keepdims=True)
        Z_sub_max = Z - max_z_keepdims
        Z_sub_max_exp = array_api.exp(Z_sub_max)
        out = array_api.sum(Z_sub_max_exp, axis=1, keepdims=True)
        out = array_api.log(out)
        out = out + max_z_keepdims
        return Z - out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A = node.inputs[0]
        shape = list(A.shape)
        max_A = A.realize_cached_data().max(axis=1, keepdims=True)
        exp_A = exp(A - max_A)
        sum_exp_A = summation(exp_A, axes=(1,))
        shape[1] = 1
        sum_exp_A_broadcast = sum_exp_A.reshape(shape).broadcast_to(A.shape)
        softmax_A = exp_A / sum_exp_A_broadcast
        sum_out_grad = summation(out_grad, axes=(1,))
        sum_out_grad_broadcast = sum_out_grad.reshape(shape).broadcast_to(A.shape)
        return out_grad - sum_out_grad_broadcast * softmax_A
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_keepdims = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_reduced = array_api.max(Z, axis=self.axes)
        Z_sub_max = Z - max_z_keepdims
        Z_sub_max_exp = array_api.exp(Z_sub_max)
        out = array_api.sum(Z_sub_max_exp, axis=self.axes)
        out = array_api.log(out)
        out = out + max_z_reduced
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A = node.inputs[0]
        shape = list(A.shape)
        max_A = A.realize_cached_data().max(axis=self.axes, keepdims=True)
        exp_A = exp(A - max_A)
        sum_exp_A = summation(exp_A, axes=self.axes)
        grad_log = out_grad / sum_exp_A
        if self.axes is None:
            axes = range(len(shape))
        elif isinstance(self.axes, tuple):
            axes = self.axes
        else:
            axes = (self.axes,)
        for axis in axes:
            shape[axis] = 1
        grad_sum = grad_log.reshape(shape).broadcast_to(exp_A.shape)
        return grad_sum * exp_A
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

