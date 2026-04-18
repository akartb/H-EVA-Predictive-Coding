import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingNetwork(nn.Module):
    """
    预测编码网络：实用混合实现。

    训练模式：标准反向传播（高效，使用Adam优化器）
    推理模式：预测编码迭代推理（生物合理，可逐步优化预测）

    这种混合实现遵循文档推荐的务实路线：
    "GPU预训练 → PC推理优化 → CPU在线学习"

    预测编码推理的优势：
    1. 迭代优化：多次推理可以逐步改善预测质量
    2. 局部性：推理过程只需要相邻层的信息
    3. 适应性：可以在推理时适应新输入（无需重新训练）

    Args:
        dims (list[int]): 各层维度列表，如 [784, 256, 10]
        inference_lr (float): 推理学习率，默认0.3
        activation (str): 激活函数类型，默认'tanh'
    """

    def __init__(self, dims, inference_lr=0.05, activation='tanh'):
        super().__init__()
        self.dims = dims
        self.inference_lr = inference_lr
        self._activation = activation

        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            is_output = (i == len(dims) - 2)
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

        self._pc_adam_initialized = False

    def _act(self, x, is_output=False):
        """
        应用激活函数。输出层不使用激活函数。

        Args:
            x (Tensor): 输入
            is_output (bool): 是否为输出层

        Returns:
            Tensor: 激活后的输出
        """
        if is_output:
            return x
        if self._activation == 'tanh':
            return torch.tanh(x)
        elif self._activation == 'relu':
            return F.relu(x)
        return x

    def _act_deriv(self, x, is_output=False):
        """
        计算激活函数的导数。

        Args:
            x (Tensor): 激活前的输入
            is_output (bool): 是否为输出层

        Returns:
            Tensor: 导数值
        """
        if is_output:
            return torch.ones_like(x)
        if self._activation == 'tanh':
            return 1 - torch.tanh(x) ** 2
        elif self._activation == 'relu':
            return (x > 0).float()
        return torch.ones_like(x)

    def _initialize_values(self, x):
        """
        初始化值神经元和预激活值。

        通过前向传播初始化所有层的值神经元和预激活值。

        Args:
            x (Tensor): 输入数据，形状 (batch, dims[0])

        Returns:
            tuple: (values, pre_acts)
                - values: 各层的值神经元列表
                - pre_acts: 各层的预激活值列表
        """
        values = []
        pre_acts = []
        h = x
        for i, layer in enumerate(self.layers):
            is_output = (i == len(self.layers) - 1)
            pre_act = layer(h)
            h = self._act(pre_act, is_output)
            pre_acts.append(pre_act)
            values.append(h)
        return values, pre_acts

    def _inference_step(self, x, values, pre_acts, update_output=True):
        """
        执行一次预测编码推理迭代。

        根据预测编码规则更新值神经元：
        - 输出层：Δv_L = -η·ε_L（如果update_output为True）
        - 隐藏层：Δv_l = η·(-ε_l + (ε_{l+1}·f'(z_{l+1}))·W_{l+1})

        Args:
            x (Tensor): 输入数据
            values (list): 各层的值神经元列表
            pre_acts (list): 各层的预激活值列表
            update_output (bool): 是否更新输出层，默认为True。
                False时仅更新隐藏层，用于local_learning_step中输出层被目标值固定的情况

        Returns:
            list: 更新后的误差列表
        """
        errors = []
        for l in range(len(self.layers)):
            lower = x if l == 0 else values[l - 1]
            is_output = (l == len(self.layers) - 1)
            pred = self._act(self.layers[l](lower), is_output)
            errors.append(values[l] - pred)

        if update_output:
            values[-1] = values[-1] - self.inference_lr * errors[-1]
            values[-1] = torch.clamp(values[-1], -10, 10)

        for l in range(len(self.layers) - 2, -1, -1):
            is_output_upper = (l + 1 == len(self.layers) - 1)
            upper_deriv = self._act_deriv(pre_acts[l + 1], is_output_upper)
            feedback = (errors[l + 1] * upper_deriv) @ self.layers[l + 1].weight
            values[l] = values[l] + self.inference_lr * (
                -errors[l] + feedback
            )
            values[l] = torch.clamp(values[l], -10, 10)

        for l in range(len(self.layers)):
            pre_acts[l] = self.layers[l](
                x if l == 0 else values[l - 1]
            )

        return errors

    def forward(self, x):
        """
        标准前向传播（用于训练）。

        Args:
            x (Tensor): 输入数据，形状 (batch, dims[0])

        Returns:
            Tensor: 输出，形状 (batch, dims[-1])
        """
        h = x
        for i, layer in enumerate(self.layers):
            is_output = (i == len(self.layers) - 1)
            h = self._act(layer(h), is_output)
        return h

    def predict(self, x, num_inference_iters=30):
        """
        预测编码推理（用于测试/部署）。

        使用迭代推理优化预测：
        1. 前向传播初始化所有层值
        2. 迭代更新值神经元最小化预测误差
        3. 返回优化后的输出

        这是预测编码的核心优势：在不更新权重的情况下，
        通过迭代推理改善预测质量。

        推理更新规则（Whittington & Bogacz 2017）：
        - 输出层：Δv_L = -η·ε_L
        - 隐藏层：Δv_l = η·(-ε_l + (ε_{l+1}·f'(z_{l+1}))·W_{l+1})

        Args:
            x (Tensor): 输入数据，形状 (batch, dims[0])
            num_inference_iters (int): 推理迭代次数

        Returns:
            Tensor: 预测输出，形状 (batch, dims[-1])
        """
        with torch.no_grad():
            values, pre_acts = self._initialize_values(x)

            for _ in range(num_inference_iters):
                self._inference_step(x, values, pre_acts, update_output=True)

            return values[-1]

    def local_learning_step(self, x, target, num_inference_iters=30, learning_lr=0.01, use_adam=True):
        """
        局部学习步骤（用于在线学习/微调）。

        使用预测编码的局部学习规则更新权重。
        每个权重的更新只依赖于它连接的两个神经元的活动，
        不需要全局梯度传播。

        权重更新规则（Whittington & Bogacz 2017）：
        ΔW_l = (f'(z_l) · ε_l)^T · v_{l-1}

        Args:
            x (Tensor): 输入数据
            target (Tensor): 目标值
            num_inference_iters (int): 推理迭代次数
            learning_lr (float): 学习率
            use_adam (bool): 是否使用局部Adam优化器。
                True: 每个参数维护独立Adam状态，加速收敛（工程优化）
                False: 纯SGD更新，严格遵循论文原始PC学习规则
        """
        with torch.no_grad():
            values, pre_acts = self._initialize_values(x)

            values[-1] = target.clone()

            for _ in range(num_inference_iters):
                self._inference_step(x, values, pre_acts, update_output=False)

            errors = []
            for l in range(len(self.layers)):
                lower = x if l == 0 else values[l - 1]
                is_output = (l == len(self.layers) - 1)
                pred = self._act(self.layers[l](lower), is_output)
                errors.append(values[l] - pred)

            for l in range(len(self.layers)):
                lower = x if l == 0 else values[l - 1]
                is_output = (l == len(self.layers) - 1)
                deriv = self._act_deriv(pre_acts[l], is_output)
                grad_w = (deriv * errors[l]).T @ lower / lower.shape[0]

                if use_adam:
                    if not self._pc_adam_initialized:
                        self._pc_adam_m_w = []
                        self._pc_adam_v_w = []
                        self._pc_adam_m_b = []
                        self._pc_adam_v_b = []
                        for layer in self.layers:
                            self._pc_adam_m_w.append(torch.zeros_like(layer.weight.data))
                            self._pc_adam_v_w.append(torch.zeros_like(layer.weight.data))
                            if layer.bias is not None:
                                self._pc_adam_m_b.append(torch.zeros_like(layer.bias.data))
                                self._pc_adam_v_b.append(torch.zeros_like(layer.bias.data))
                            else:
                                self._pc_adam_m_b.append(None)
                                self._pc_adam_v_b.append(None)
                        self._pc_adam_t = 0
                        self._pc_adam_initialized = True

                    self._pc_adam_t += 1
                    beta1, beta2, eps = 0.9, 0.999, 1e-8

                    self._pc_adam_m_w[l] = beta1 * self._pc_adam_m_w[l] + (1 - beta1) * grad_w
                    self._pc_adam_v_w[l] = beta2 * self._pc_adam_v_w[l] + (1 - beta2) * grad_w ** 2
                    m_hat = self._pc_adam_m_w[l] / (1 - beta1 ** self._pc_adam_t)
                    v_hat = self._pc_adam_v_w[l] / (1 - beta2 ** self._pc_adam_t)
                    self.layers[l].weight.data += learning_lr * m_hat / (v_hat.sqrt() + eps)
                else:
                    self.layers[l].weight.data += learning_lr * grad_w

                if self.layers[l].bias is not None:
                    grad_b = (deriv * errors[l]).mean(dim=0)
                    if use_adam:
                        self._pc_adam_m_b[l] = beta1 * self._pc_adam_m_b[l] + (1 - beta1) * grad_b
                        self._pc_adam_v_b[l] = beta2 * self._pc_adam_v_b[l] + (1 - beta2) * grad_b ** 2
                        m_hat_b = self._pc_adam_m_b[l] / (1 - beta1 ** self._pc_adam_t)
                        v_hat_b = self._pc_adam_v_b[l] / (1 - beta2 ** self._pc_adam_t)
                        self.layers[l].bias.data += learning_lr * m_hat_b / (v_hat_b.sqrt() + eps)
                    else:
                        self.layers[l].bias.data += learning_lr * grad_b
