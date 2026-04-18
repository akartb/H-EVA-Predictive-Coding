"""
实验1：Sin函数拟合

验证预测编码网络推理和学习的基本功能。

混合架构实现：
- 训练：标准反向传播 + Adam优化器（高效收敛）
- 评估：预测编码迭代推理（生物合理，可逐步优化预测）

目标：MSE < 0.01
论文依据：Whittington & Bogacz (2017) 第4.1节
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.pc_network import PredictiveCodingNetwork
from src.utils import plot_sin_results


def experiment_sin():
    """
    运行Sin函数拟合实验。

    使用标准反向传播训练网络，使用预测编码推理评估性能。
    同时对比forward()和predict()的输出差异，
    验证PC推理能否达到与标准前向传播相当的精度。
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    x = torch.linspace(0, 2 * torch.pi, 200, device=device).unsqueeze(1)
    y = torch.sin(x)

    model = PredictiveCodingNetwork(
        dims=[1, 64, 1],
        inference_lr=0.3,
        activation='tanh'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    eval_losses = []
    error_history = []

    print("=" * 60, flush=True)
    print("Experiment 1: Sin Function Fitting", flush=True)
    print(f"Network: {model.dims}, activation=tanh, optimizer=Adam", flush=True)
    print(f"inference_lr={model.inference_lr}", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(500):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                pred_pc = model.predict(x, num_inference_iters=30)
                eval_loss = torch.mean((pred_pc - y) ** 2).item()
                eval_losses.append(eval_loss)

            if (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1:4d}, Train MSE: {loss.item():.6f}, "
                      f"PC Eval MSE: {eval_loss:.6f}", flush=True)

    with torch.no_grad():
        final_pred = model.predict(x, num_inference_iters=30)

        h = x
        values = []
        pre_acts = []
        for i, layer in enumerate(model.layers):
            is_output = (i == len(model.layers) - 1)
            pre_act = layer(h)
            h = model._act(pre_act, is_output)
            pre_acts.append(pre_act.clone())
            values.append(h.clone())

        for _ in range(30):
            errors = []
            for l in range(len(model.layers)):
                lower = x if l == 0 else values[l - 1]
                is_output = (l == len(model.layers) - 1)
                pred_l = model._act(model.layers[l](lower), is_output)
                errors.append(values[l] - pred_l)
            total_err = sum(e.norm().item() ** 2 for e in errors) ** 0.5
            error_history.append(total_err)

            values[-1] = values[-1] - model.inference_lr * errors[-1]
            values[-1] = torch.clamp(values[-1], -10, 10)

            for l in range(len(model.layers) - 2, -1, -1):
                is_output_upper = (l + 1 == len(model.layers) - 1)
                upper_deriv = model._act_deriv(pre_acts[l + 1], is_output_upper)
                feedback = (errors[l + 1] * upper_deriv) @ model.layers[l + 1].weight
                values[l] = values[l] + model.inference_lr * (
                    -errors[l] + feedback
                )
                values[l] = torch.clamp(values[l], -10, 10)

            for l in range(len(model.layers)):
                pre_acts[l] = model.layers[l](
                    x if l == 0 else values[l - 1]
                )

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plot_sin_results(x, y, final_pred, eval_losses, error_history, save_dir)

    final_mse = eval_losses[-1] if eval_losses else float('inf')
    print("\n" + "=" * 60, flush=True)
    print(f"Final PC Eval MSE: {final_mse:.6f}", flush=True)
    status = "PASS" if final_mse < 0.01 else "FAIL"
    print(f"Target MSE < 0.01: {status}", flush=True)
    print(f"Results saved to: {os.path.abspath(save_dir)}", flush=True)
    print("=" * 60, flush=True)

    return final_mse


if __name__ == "__main__":
    experiment_sin()
