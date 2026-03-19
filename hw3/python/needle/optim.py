"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for para in self.params:
            curr_grad = para.grad.data + self.weight_decay * para.data
            if para not in self.u:
                self.u[para] = (1-self.momentum) * curr_grad.data
            else:
                self.u[para] = self.momentum * self.u[para].data + (1-self.momentum) * curr_grad.data
            para.data = ndl.Tensor(para.data - self.lr * self.u[para].data, dtype=para.dtype)
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for para in self.params:
            curr_grad = para.grad.data + self.weight_decay * para.data
            if para not in self.m or para not in self.v:
                self.m[para] = (1-self.beta1) * curr_grad.data
                self.v[para] = (1-self.beta2) * (curr_grad.data ** 2)
            else:
                self.m[para] = (1-self.beta1) * curr_grad.data + self.beta1 * self.m[para].data
                self.v[para] = (1-self.beta2) * (curr_grad.data ** 2) + self.beta2 * self.v[para].data
            m_bias = self.m[para].data / (1 - self.beta1 ** self.t)
            v_bias = self.v[para].data / (1 - self.beta2 ** self.t)
            para.data = ndl.Tensor(para.data - self.lr * m_bias.data / ((v_bias.data ** 0.5) + self.eps), dtype=para.dtype)
        ### END YOUR SOLUTION
