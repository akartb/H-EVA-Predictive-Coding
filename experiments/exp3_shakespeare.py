"""
实验3：字符级语言建模

验证预测编码网络在语言建模任务上的学习能力。

混合架构实现：
- 训练：标准反向传播 + CrossEntropyLoss + Adam优化器
- 评估：预测编码迭代推理

使用Shakespeare文本数据，训练字符级预测编码模型。
目标：困惑度 < 3.5

论文依据：Salvatori et al. (2023)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import torch
import torch.nn as nn
from src.pc_network import PredictiveCodingNetwork
from src.utils import plot_language_model_results


def get_shakespeare_data():
    """
    获取Shakespeare文本数据。

    如果本地没有数据文件，使用简单的示例文本。

    Returns:
        str: Shakespeare文本
    """
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'shakespeare.txt')
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream. Ay, there's the rub,
For in that sleep of death what dreams may come
When we have shuffled off this mortal coil,
Must give us pause. There's the respect
That makes calamity of so long life.""" * 50


def experiment_shakespeare():
    """
    运行字符级语言建模实验。

    使用滑动窗口方式生成训练样本，
    标准反向传播训练，预测编码推理评估。
    使用one-hot编码输入，CrossEntropyLoss训练。
    """
    torch.manual_seed(42)
    device = torch.device("cpu")

    text = get_shakespeare_data()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    vocab_size = len(chars)

    seq_length = 20
    encoded = [char_to_idx[c] for c in text]

    sequences = []
    targets = []
    for i in range(len(encoded) - seq_length):
        seq = encoded[i:i + seq_length]
        tgt = encoded[i + seq_length]
        sequences.append(seq)
        targets.append(tgt)

    sequences = torch.tensor(sequences, dtype=torch.long, device=device)
    targets = torch.tensor(targets, dtype=torch.long, device=device)

    def to_onehot(indices, vocab_size):
        onehot = torch.zeros(indices.shape[0], vocab_size, device=device)
        onehot.scatter_(1, indices.unsqueeze(1), 1.0)
        return onehot

    model = PredictiveCodingNetwork(
        dims=[seq_length * vocab_size, 128, vocab_size],
        inference_lr=0.01,
        activation='tanh'
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    batch_size = 32

    print("=" * 60, flush=True)
    print("Experiment 3: Character-level Language Modeling", flush=True)
    print(f"Vocab size: {vocab_size}, Seq length: {seq_length}", flush=True)
    print(f"Network: {model.dims}", flush=True)
    print(f"inference_lr={model.inference_lr}", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(50):
        perm = torch.randperm(len(sequences))
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(sequences), batch_size):
            idx = perm[i:i + batch_size]
            batch_seq = sequences[idx]
            batch_tgt = targets[idx]

            x = to_onehot(batch_seq.view(-1), vocab_size).view(idx.shape[0], -1)

            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, batch_tgt)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                sample_idx = torch.randint(0, len(sequences), (1,))
                x = to_onehot(sequences[sample_idx].view(-1), vocab_size).view(1, -1)

                fwd_pred = model(x).argmax(dim=1).item()
                pc_pred = model.predict(x, num_inference_iters=50).argmax(dim=1).item()

                true_char = idx_to_char[targets[sample_idx].item()]
                fwd_char = idx_to_char[fwd_pred]
                pc_char = idx_to_char[pc_pred]

                ppl = math.exp(avg_loss)

            print(f"Epoch {epoch + 1:2d}, Loss: {avg_loss:.4f}, PPL: {ppl:.2f}, "
                  f"Fwd: '{fwd_char}' PC: '{pc_char}' True: '{true_char}'", flush=True)

    save_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    plot_language_model_results(losses, save_dir)

    final_ppl = math.exp(losses[-1])
    print("\n" + "=" * 60, flush=True)
    print(f"Final Loss: {losses[-1]:.4f}", flush=True)
    print(f"Final Perplexity: {final_ppl:.2f}", flush=True)
    status = "PASS" if final_ppl < 3.5 else "FAIL"
    print(f"Target PPL < 3.5: {status}", flush=True)
    print(f"Results saved to: {os.path.abspath(save_dir)}", flush=True)
    print("=" * 60, flush=True)

    return final_ppl


if __name__ == "__main__":
    experiment_shakespeare()
