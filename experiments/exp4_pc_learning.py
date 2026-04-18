"""
实验4：纯预测编码学习 vs 反向传播训练 对比

验证预测编码的局部学习规则（local_learning_step）能否替代反向传播。
这是预测编码架构的"另一半"——之前只验证了PC推理，现在验证PC学习。

对比三种模式：
1. BP训练 + BP推理（标准神经网络）
2. BP训练 + PC推理（之前的混合架构）
3. PC学习 + PC推理（纯预测编码系统）

论文依据：Whittington & Bogacz (2017) — PC学习与BP数学等价（当推理迭代→∞）
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from src.pc_network import PredictiveCodingNetwork


def compare_sin():
    """
    Sin函数拟合：BP训练 vs PC学习 对比。
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    x = torch.linspace(0, 2 * torch.pi, 200, device=device).unsqueeze(1)
    y = torch.sin(x)

    print("=" * 70, flush=True)
    print("Experiment 4: Pure PC Learning vs BP Training (Sin Fitting)", flush=True)
    print("=" * 70, flush=True)

    # --- Mode 1: BP训练 + BP推理 ---
    torch.manual_seed(42)
    model_bp = PredictiveCodingNetwork(dims=[1, 64, 1], inference_lr=0.01, activation='tanh')
    optimizer = torch.optim.Adam(model_bp.parameters(), lr=0.005)
    loss_fn = nn.MSELoss()

    print("\n[Mode 1] BP Training + BP Inference", flush=True)
    for epoch in range(1000):
        optimizer.zero_grad()
        pred = model_bp(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 200 == 0:
            print(f"  Epoch {epoch + 1:4d}: MSE = {loss.item():.6f}", flush=True)
    bp_train_mse = loss.item()

    with torch.no_grad():
        bp_pred = model_bp(x)
        bp_mse = torch.mean((bp_pred - y) ** 2).item()

    # --- Mode 2: BP训练 + PC推理 ---
    with torch.no_grad():
        pc_from_bp = model_bp.predict(x, num_inference_iters=30)
        pc_from_bp_mse = torch.mean((pc_from_bp - y) ** 2).item()

    # --- Mode 3: PC学习 + PC推理 ---
    torch.manual_seed(42)
    model_pc = PredictiveCodingNetwork(dims=[1, 64, 1], inference_lr=0.01, activation='tanh')

    print("\n[Mode 3] PC Learning + PC Inference", flush=True)
    for epoch in range(1000):
        model_pc.local_learning_step(x, y, num_inference_iters=50, learning_lr=0.005)
        if (epoch + 1) % 200 == 0:
            with torch.no_grad():
                pred = model_pc.predict(x, num_inference_iters=30)
                mse = torch.mean((pred - y) ** 2).item()
            print(f"  Epoch {epoch + 1:4d}: MSE = {mse:.6f}", flush=True)

    with torch.no_grad():
        pc_pred = model_pc.predict(x, num_inference_iters=30)
        pc_mse = torch.mean((pc_pred - y) ** 2).item()

    print("\n" + "=" * 70, flush=True)
    print("COMPARISON RESULTS (Sin Fitting)", flush=True)
    print("-" * 70, flush=True)
    print(f"  Mode 1: BP Train + BP Infer  = {bp_mse:.6f}", flush=True)
    print(f"  Mode 2: BP Train + PC Infer  = {pc_from_bp_mse:.6f}", flush=True)
    print(f"  Mode 3: PC Learn + PC Infer  = {pc_mse:.6f}", flush=True)
    print("-" * 70, flush=True)
    print(f"  Target: MSE < 0.01", flush=True)
    print(f"  Mode 1: {'PASS' if bp_mse < 0.01 else 'FAIL'}", flush=True)
    print(f"  Mode 2: {'PASS' if pc_from_bp_mse < 0.01 else 'FAIL'}", flush=True)
    print(f"  Mode 3: {'PASS' if pc_mse < 0.01 else 'FAIL'}", flush=True)
    print("=" * 70, flush=True)

    return bp_mse, pc_from_bp_mse, pc_mse


def compare_digits():
    """
    手写数字分类：BP训练 vs PC学习 对比。
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    digits = load_digits()
    X = digits.data.astype('float32') / 16.0
    y = digits.target.astype('int64')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train = torch.tensor(X_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_test = torch.tensor(y_test, device=device)

    in_dim = X_train.shape[1]
    batch_size = 64

    print("\n" + "=" * 70, flush=True)
    print("Experiment 4b: Pure PC Learning vs BP Training (Digit Classification)", flush=True)
    print("=" * 70, flush=True)

    # --- Mode 1: BP训练 ---
    torch.manual_seed(42)
    model_bp = PredictiveCodingNetwork(dims=[in_dim, 128, 10], inference_lr=0.01, activation='tanh')
    optimizer = torch.optim.Adam(model_bp.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    print("\n[Mode 1] BP Training", flush=True)
    for epoch in range(100):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model_bp(X_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                acc = (model_bp(X_test).argmax(dim=1) == y_test).float().mean().item()
            print(f"  Epoch {epoch + 1:3d}: Acc = {acc:.4f}", flush=True)

    with torch.no_grad():
        bp_acc = (model_bp(X_test).argmax(dim=1) == y_test).float().mean().item()
        pc_from_bp = model_bp.predict(X_test, num_inference_iters=50).argmax(dim=1)
        pc_from_bp_acc = (pc_from_bp == y_test).float().mean().item()

    # --- Mode 3: PC学习 ---
    torch.manual_seed(42)
    model_pc = PredictiveCodingNetwork(dims=[in_dim, 128, 10], inference_lr=0.01, activation='tanh')

    print("\n[Mode 3] PC Learning", flush=True)
    for epoch in range(100):
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]
            target_onehot = torch.zeros(len(idx), 10, device=device)
            target_onehot.scatter_(1, batch_y.unsqueeze(1), 1.0)
            model_pc.local_learning_step(
                batch_x, target_onehot,
                num_inference_iters=50,
                learning_lr=0.005
            )
        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                pred = model_pc.predict(X_test, num_inference_iters=50).argmax(dim=1)
                acc = (pred == y_test).float().mean().item()
            print(f"  Epoch {epoch + 1:3d}: Acc = {acc:.4f}", flush=True)

    with torch.no_grad():
        pc_pred = model_pc.predict(X_test, num_inference_iters=50).argmax(dim=1)
        pc_acc = (pc_pred == y_test).float().mean().item()

    print("\n" + "=" * 70, flush=True)
    print("COMPARISON RESULTS (Digit Classification)", flush=True)
    print("-" * 70, flush=True)
    print(f"  Mode 1: BP Train + BP Infer  = {bp_acc:.4f}", flush=True)
    print(f"  Mode 2: BP Train + PC Infer  = {pc_from_bp_acc:.4f}", flush=True)
    print(f"  Mode 3: PC Learn + PC Infer  = {pc_acc:.4f}", flush=True)
    print("-" * 70, flush=True)
    print(f"  Target: Acc > 0.90", flush=True)
    print(f"  Mode 1: {'PASS' if bp_acc > 0.90 else 'FAIL'}", flush=True)
    print(f"  Mode 2: {'PASS' if pc_from_bp_acc > 0.90 else 'FAIL'}", flush=True)
    print(f"  Mode 3: {'PASS' if pc_acc > 0.90 else 'FAIL'}", flush=True)
    print("=" * 70, flush=True)

    return bp_acc, pc_from_bp_acc, pc_acc


if __name__ == "__main__":
    compare_sin()
    compare_digits()
