import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch


def plot_training_curves(losses, title, ylabel, save_path, yscale='log'):
    """
    绘制训练损失/误差曲线并保存。

    Args:
        losses (list[float]): 损失值列表
        title (str): 图表标题
        ylabel (str): Y轴标签
        save_path (str): 图片保存路径
        yscale (str): Y轴刻度类型，默认'log'
    """
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.yscale(yscale)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_inference_convergence(error_history, save_path):
    """
    绘制推理误差收敛曲线。

    用于判断推理迭代次数是否足够——如果曲线未收敛到平台，
    说明需要增加推理迭代次数。

    Args:
        error_history (list[float]): 每次推理迭代的总误差
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(8, 5))
    plt.plot(error_history)
    plt.title("Inference Error Convergence")
    plt.xlabel("Inference Iteration")
    plt.ylabel("Total Prediction Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_sin_results(x, y_true, y_pred, losses, error_history, save_dir):
    """
    绘制Sin函数拟合实验的完整结果（三子图）。

    Args:
        x (Tensor): 输入x值
        y_true (Tensor): 真实sin(x)值
        y_pred (Tensor): 模型预测值
        losses (list[float]): 训练损失历史
        error_history (list[float]): 推理误差历史
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x.cpu().numpy(), y_true.cpu().numpy(), label="True sin(x)", linewidth=2)
    axes[1].plot(x.cpu().numpy(), y_pred.cpu().numpy(), label="Predicted", linewidth=2, linestyle="--")
    axes[1].set_title("Sin Function Fitting")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(error_history)
    axes[2].set_title("Inference Error Convergence")
    axes[2].set_xlabel("Inference Iteration")
    axes[2].set_ylabel("Total Prediction Error")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "exp1_sin_results.png"), dpi=150)
    plt.close()


def plot_mnist_results(train_losses, test_accs, save_dir):
    """
    绘制MNIST分类实验的完整结果（双子图）。

    Args:
        train_losses (list[float]): 训练损失历史
        test_accs (list[float]): 测试准确率历史
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_losses)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(test_accs)
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "exp2_mnist_results.png"), dpi=150)
    plt.close()


def plot_language_model_results(losses, save_dir):
    """
    绘制字符级语言建模实验的损失曲线。

    Args:
        losses (list[float]): 交叉熵损失历史
        save_dir (str): 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Character-level Language Modeling Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "exp3_shakespeare_results.png"), dpi=150)
    plt.close()
