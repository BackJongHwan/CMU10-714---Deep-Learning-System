from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        # assume Z's dimesion (3, 4)
        # corresponding dimesion is collasepd
        
        # let Z(3, 4), axis = 1 and output is (3, 4) same with input
        
        # max_z_keepdims (3, 4) for subtracting
        max_z_keepdims = array_api.max(Z, axis=1, keepdims=True) # (3, 1)

        # (3, 4) - (3, 1) -> (3,4) broadcasting.
        Z_sub_max = Z - max_z_keepdims
        Z_sub_max_exp = array_api.exp(Z_sub_max)

        # (3, 4) -> (3,1)
        out = array_api.sum(Z_sub_max_exp, axis=1, keepdims=True)
        out = array_api.log(out)
        # (3, 1) + (3, 1) => (3, 1)
        out = out + max_z_keepdims

        # (3, 4)  - (3, 1) = > (3, 4) broadcast
        return Z - out 

    def gradient(self, out_grad: Tensor, node: Tensor):
        # A shape = (3, 4)
        A = node.inputs[0]
        # [3, 4]
        shape = list(A.shape)

        # gradient of logsumexp(A)
        # (3, 1)
        max_A = A.realize_cached_data().max(axis=1, keepdims=True)
        # (3, 4) - (3, 1) => (3, 4)
        exp_A = exp(A - max_A)

        # (3, 4) -> (3,)
        sum_exp_A = summation(exp_A, axes=(1,))
        # (3, ) -> (3, 1) -> (3, 4)
        shape[1] = 1
        sum_exp_A_broadcast = sum_exp_A.reshape(shape).broadcast_to(A.shape)
        softmax_A = exp_A / sum_exp_A_broadcast
        # 2. Apply the Chain Rule / Vector-Jacobian Product
        # Sum the incoming gradients across the row: e.g., (3, 3) -> (3,)
        sum_out_grad = summation(out_grad, axes=(1,))
        
        # Fix the shape to broadcast safely back to the original size
        sum_out_grad_broadcast = sum_out_grad.reshape(shape).broadcast_to(A.shape)

        # 3. Final Gradient Calculation
        return out_grad - sum_out_grad_broadcast * softmax_A



        

def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:

        # assume Z's dimesion (2,3,4)
        # if axis=(0) -> out = (3, 4)
        # if axis=(1,2) -> out = (2,)
        # corresponding dimesion is collasepd
        
        # let Z(2,3,4) and axis = 1 then output is (2,4)
        
        # max_z_keepdims (2,1,4) for subtracting
        max_z_keepdims = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_reduced = array_api.max(Z, axis=self.axes)

        # (2,3,4) - (2, 1, 4) -> (2,3,4)
        Z_sub_max = Z - max_z_keepdims
        Z_sub_max_exp = array_api.exp(Z_sub_max)
        out = array_api.sum(Z_sub_max_exp, axis=self.axes)
        out = array_api.log(out)
        out = out + max_z_reduced
        
        # not exist !!  out = array_api.logsumexp(Z) #-> this is not exist?
        return out


    def gradient(self, out_grad: Tensor, node: Tensor):
        # A is Tensor not a NDArray
        A = node.inputs[0] 
        shape = list(A.shape)
        '''
        # just not thinking overflow..
        exp_A = exp(A)
        sum_exp_A = summation(exp_A, axes=self.axes)
        log_sum_exp_A = log(sum_exp_A)

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

        grad = grad_sum * exp_A
        '''
        ## I'll thinking overflow!!
        '''
        # then I'll change this 
        max_reduced = array_api.max(A, axis=self.axes)
        max_keepdims = array_api.max(A, axis=self.axes, keepdims=True)
        '''
        max_A = A.realize_cached_data().max(axis=self.axes, keepdims=True)
        exp_A = exp(A - max_A)
        sum_exp_A = summation(exp_A, axes=self.axes)
        # log_sum_exp_A = log(sum_exp_A)

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

        grad = grad_sum * exp_A

        return grad

def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)