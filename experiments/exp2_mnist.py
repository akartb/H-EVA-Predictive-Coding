"""
实验2：手写数字分类

验证预测编码网络在图像分类任务上的学习能力。

混合架构实现：
- 训练：标准反向传播 + CrossEntropyLoss + Adam优化器
- 评估：预测编码迭代推理（PC推理后取argmax分类）

使用sklearn digits数据集（8x8手写数字，1797样本，10类），
无需下载，可快速验证架构。

目标：准确率 > 92%
论文依据：Whittington & Bogacz (2017) 第4.2节
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from src.pc_network import PredictiveCodingNetwork
from src.utils import plot_mnist_results


def experiment_mnist():
    """
    运行手写数字分类实验。

    使用sklearn digits数据集，标准反向传播训练，
    预测编码推理评估分类准确率。
    同时对比前向传播和PC推理的准确率差异。
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    digits = load_digits()
    X = digits.data.astype('float32')
    y = digits.target.astype('int64')

    X = X / 16.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = torch.tensor(X_train, device=device)
    y_train = torch.tensor(y_train, device=device)
    X_test = torch.tensor(X_test, device=device)
    y_test = torch.tensor(y_test, device=device)

    in_dim = X_train.shape[1]

    model = PredictiveCodingNetwork(
        dims=[in_dim, 128, 10],
        inference_lr=0.01,
        activation='tanh'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    train_losses = []
    test_accs_fwd = []
    test_accs_pc = []

    batch_size = 64

    print("=" * 60, flush=True)
    print("Experiment 2: Digit Classification (sklearn digits)", flush=True)
    print(f"Network: {model.dims}, activation=tanh, optimizer=Adam", flush=True)
    print(f"Loss: CrossEntropy, inference_lr={model.inference_lr}", flush=True)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            idx = perm[i:i + batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            fwd_pred = model(X_test).argmax(dim=1)
            fwd_acc = (fwd_pred == y_test).float().mean().item()

            pc_pred = model.predict(X_test, num_inference_iters=50).argmax(dim=1)
            pc_acc = (pc_pred == y_test).float().mean().item()

        test_accs_fwd.append(fwd_acc)
        test_accs_pc.append(pc_acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:3d}, Loss: {avg_loss:.4f}, "
                  f"Fwd Acc: {fwd_acc:.4f}, PC Acc: {pc_acc:.4f}", flush=True)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plot_mnist_results(train_losses, test_accs_pc, save_dir)

    final_acc = test_accs_pc[-1] if test_accs_pc else 0.0
    final_fwd = test_accs_fwd[-1] if test_accs_fwd else 0.0
    print("\n" + "=" * 60, flush=True)
    print(f"Final Forward Acc: {final_fwd:.4f}", flush=True)
    print(f"Final PC Inference Acc: {final_acc:.4f}", flush=True)
    status = "PASS" if final_acc > 0.92 else "FAIL"
    print(f"Target Acc > 92%: {status}", flush=True)
    print(f"Results saved to: {os.path.abspath(save_dir)}", flush=True)
    print("=" * 60, flush=True)

    return final_acc


if __name__ == "__main__":
    experiment_mnist()
