import math
import torch
import torch.nn as nn


class PredictiveCodingLayer(nn.Module):
    """
    预测编码层：预测编码网络的基本构建块。

    基于 Whittington & Bogacz (2017)，支持非线性激活函数和Adam优化器。

    带激活函数的能量函数:
    E_l = ½ ‖ε_l‖² = ½ ‖v_l - f(v_{l-1} @ W_l^T)‖²

    Args:
        in_dim (int): 输入维度
        out_dim (int): 输出维度
        inference_lr (float): 推理学习率，默认0.1
        learning_lr (float): 权重学习率，默认0.01
        activation (str): 激活函数类型，'tanh'/'relu'/'none'，默认'tanh'
        use_adam (bool): 是否使用Adam优化器更新权重，默认True
    """

    def __init__(self, in_dim, out_dim, inference_lr=0.1, learning_lr=0.01,
                 activation='tanh', use_adam=True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inference_lr = inference_lr
        self.learning_lr = learning_lr
        self.use_adam = use_adam

        std = 1.0 / math.sqrt(in_dim)
        self.W = nn.Parameter(torch.randn(out_dim, in_dim) * std)

        if activation == 'tanh':
            self.act = torch.tanh
            self.act_deriv = lambda x: 1 - torch.tanh(x) ** 2
        elif activation == 'relu':
            self.act = torch.relu
            self.act_deriv = lambda x: (x > 0).float()
        else:
            self.act = lambda x: x
            self.act_deriv = lambda x: torch.ones_like(x)

        if use_adam:
            self._adam_m = torch.zeros(out_dim, in_dim)
            self._adam_v = torch.zeros(out_dim, in_dim)
            self._adam_t = 0

        self.values = None
        self.errors = None
        self.pre_act = None

    def init_values(self, lower_values):
        """
        使用前向传播初始化值神经元。

        Args:
            lower_values (Tensor): 下层值神经元，形状 (batch, in_dim)
        """
        with torch.no_grad():
            self.pre_act = lower_values @ self.W.T
            self.values = self.act(self.pre_act)

    def compute_errors(self, lower_values):
        """
        计算预测误差：ε_l = v_l - f(v_{l-1} @ W_l^T)

        Args:
            lower_values (Tensor): 下层值神经元，形状 (batch, in_dim)

        Returns:
            Tensor: 预测误差，形状 (batch, out_dim)
        """
        self.pre_act = lower_values @ self.W.T
        predictions = self.act(self.pre_act)
        self.errors = self.values - predictions
        return self.errors

    def update_values_top(self):
        """
        更新顶层值神经元：v_L += η_v * (-ε_L)
        """
        self.values = self.values - self.inference_lr * self.errors
        self.values = torch.clamp(self.values, -10, 10)

    def update_values_middle(self, feedback):
        """
        更新中间层值神经元：v_l += η_v * (-ε_l + f'(pre_act) * feedback)

        Args:
            feedback (Tensor): 上层反馈项，形状 (batch, out_dim)
        """
        deriv = self.act_deriv(self.pre_act)
        self.values = self.values + self.inference_lr * (-self.errors + deriv * feedback)
        self.values = torch.clamp(self.values, -10, 10)

    def _adam_update(self, grad, param_data, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Adam优化器更新。

        使用一阶矩估计和二阶矩估计自适应调整学习率，
        大幅加速收敛速度。Adam的局部实现保持了局部学习规则的精神，
        因为每个参数的更新仍然只依赖于该参数自身的梯度历史。

        Args:
            grad (Tensor): 梯度
            param_data (Tensor): 参数数据
            beta1 (float): 一阶矩衰减率
            beta2 (float): 二阶矩衰减率
            eps (float): 数值稳定项
        """
        self._adam_t += 1
        self._adam_m = beta1 * self._adam_m + (1 - beta1) * grad
        self._adam_v = beta2 * self._adam_v + (1 - beta2) * grad ** 2
        m_hat = self._adam_m / (1 - beta1 ** self._adam_t)
        v_hat = self._adam_v / (1 - beta2 ** self._adam_t)
        param_data.add_(self.learning_lr * m_hat / (v_hat.sqrt() + eps))

    def update_weights(self, lower_values):
        """
        更新前向权重。

        使用Adam优化器或简单SGD更新权重。
        梯度 = (f'(pre_act) * ε_l)^T @ v_{l-1} / batch_size

        Args:
            lower_values (Tensor): 下层值神经元，形状 (batch, in_dim)
        """
        deriv = self.act_deriv(self.pre_act)
        grad = (deriv * self.errors).T @ lower_values / lower_values.shape[0]

        if self.use_adam:
            self._adam_update(grad, self.W.data)
        else:
            grad_norm = grad.norm().item()
            if grad_norm > 1.0:
                grad = grad / grad_norm
            self.W.data += self.learning_lr * grad

    def update_feedback_weights(self, upper_errors, feedback_W):
        """
        更新反馈权重。

        Args:
            upper_errors (Tensor): 上层误差神经元
            feedback_W (nn.Parameter): 反馈权重参数
        """
        deriv = self.act_deriv(self.pre_act)
        grad = (deriv * upper_errors).T @ self.values / self.values.shape[0]

        if self.use_adam:
            if not hasattr(self, '_fb_adam_m'):
                self._fb_adam_m = torch.zeros_like(feedback_W.data)
                self._fb_adam_v = torch.zeros_like(feedback_W.data)
                self._fb_adam_t = 0
            self._fb_adam_t += 1
            self._fb_adam_m = 0.9 * self._fb_adam_m + 0.1 * grad
            self._fb_adam_v = 0.999 * self._fb_adam_v + 0.001 * grad ** 2
            m_hat = self._fb_adam_m / (1 - 0.9 ** self._fb_adam_t)
            v_hat = self._fb_adam_v / (1 - 0.999 ** self._fb_adam_t)
            feedback_W.data.add_(self.learning_lr * m_hat / (v_hat.sqrt() + 1e-8))
        else:
            grad_norm = grad.norm().item()
            if grad_norm > 1.0:
                grad = grad / grad_norm
            feedback_W.data += self.learning_lr * grad
