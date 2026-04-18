Here's an English version of the `README.md` based on the provided Chinese documentation. It maintains the structure and key technical details while presenting them clearly for an English-speaking audience.

```markdown
# H-EVA: Predictive Coding Architecture Project

## Overview

This project implements a complete Predictive Coding (PC) neural network architecture based on the foundational works of Whittington & Bogacz (2017), Millidge et al. (2022), and Salvatori et al. (2023). The implementation is validated on three tasks: **Sine function fitting**, **handwritten digit classification**, and **character-level language modeling**.

The architecture supports three operational modes:
- **BP Training + BP Inference**: Standard neural network (baseline).
- **BP Training + PC Inference**: Hybrid architecture (efficient training with biologically plausible inference).
- **PC Learning + PC Inference**: Pure predictive coding system (fully local learning, no backpropagation required).

## Theoretical Foundation

**Free Energy Principle → Energy Function → Inference Phase (Fix W, Update v) + Learning Phase (Fix v, Update W) → Local Learning Rules → Mathematical Equivalence to Backpropagation (as inference iterations → ∞).**

## Project Structure

```
H-EVA/
├── src/
│   ├── __init__.py
│   ├── pc_layer.py            # Predictive coding layer implementation
│   ├── pc_network.py          # Core network (supports three modes)
│   └── utils.py               # Utilities (visualization, metrics)
├── experiments/
│   ├── exp1_sin.py            # Exp 1: Sine fitting
│   ├── exp2_mnist.py          # Exp 2: Digit classification (sklearn digits)
│   ├── exp3_shakespeare.py    # Exp 3: Character-level language modeling
│   └── exp4_pc_learning.py    # Exp 4: Pure PC Learning vs. BP comparison
├── results/                   # Output figures and logs
└── README.md                  # This document
```

## Core Mathematical Formulation

- **Energy Function**:  
  \( E = \frac{1}{2} \sum_l \| \varepsilon_l \|^2 = \frac{1}{2} \sum_l \| v_l - f(W_l v_{l-1}) \|^2 \)

- **Inference Update (Output Layer)**:  
  \( \Delta v_L = -\eta \cdot \varepsilon_L \)

- **Inference Update (Hidden Layer)**:  
  \( \Delta v_l = \eta \cdot \left( -\varepsilon_l + (\varepsilon_{l+1} \odot f'(z_{l+1})) W_{l+1}^T \right) \)

- **Weight Update**:  
  \( \Delta W_l = (f'(z_l) \odot \varepsilon_l)^T v_{l-1} \)

## Critical Bug Fixes & Implementation Notes

During development, several critical issues were identified and resolved to ensure theoretical correctness and numerical stability.

| Bug ID | Description | Impact | Resolution |
| :--- | :--- | :--- | :--- |
| **1** | **Incorrect feedback derivative placement** | PC inference diverged (accuracy dropped to 25%). | Fixed gradient flow to use \( f'(z_{l+1}) \) instead of \( f'(z_l) \). |
| **2** | **High default inference learning rate** | Inference diverged at `lr=0.3`. | Reduced default `inference_lr` to `0.05` (classification uses `0.01`). |
| **3** | **Stale errors in `local_learning_step`** | Weight updates used mismatched values/errors. | Recompute errors after inference loop finishes. |
| **4** | **Pre-activation staleness** | `pre_acts` not synchronized during inference loop. | Recompute all `pre_acts` after completing value updates for all layers. |
| **5** | **Manual loop formula mismatch in Exp 1** | Visualization logs used wrong derivative logic. | Aligned manual loop with corrected `pc_network.py` logic. |
| **6** | **Missing pure SGD option** | Could not validate original PC learning rule without Adam. | Added `use_adam=False` parameter for weight updates. |

## Hyperparameter Recommendations

| Parameter | Sine (BP) | Digits (BP) | Sine (PC Learn) | Digits (PC Learn) |
| :--- | :--- | :--- | :--- | :--- |
| `inference_lr` | 0.05 | 0.01 | 0.01 | 0.01 |
| `inference_iters` | 30 | 50 | 50 | 50 |
| `learning_lr` | 0.005 (Adam) | 0.001 (Adam) | 0.005 (Adam) | 0.005 (Adam) |
| `epochs` | 500 | 100 | 1000 | 100 |

## Experimental Results

### Experiments 1–3: BP Training + PC Inference

| Experiment | Metric | BP Result | PC Inference Result | Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Sine Fitting | MSE | 0.006092 | 0.006034 | < 0.01 | ✅ Pass |
| Digit Classification | Accuracy | 97.78% | 96.67% | > 92% | ✅ Pass |
| Char Modeling | Perplexity | 1.00 | 1.00 | < 3.5 | ✅ Pass |

### Experiment 4: Pure PC Learning vs. BP Training (Core Validation)

#### Sine Function Fitting (After Staleness Fix)
| Mode | MSE | Target < 0.01 |
| :--- | :--- | :--- |
| BP Train + BP Infer | 0.006092 | ✅ |
| BP Train + PC Infer | 0.006034 | ✅ |
| **PC Learn (Adam) + PC Infer** | **0.000677** | **✅ (Best)** |
| PC Learn (SGD) + PC Infer | 0.026993 | ❌ Fail |

#### Handwritten Digit Classification
| Mode | Accuracy | Target > 90% |
| :--- | :--- | :--- |
| BP Train + BP Infer | 97.78% | ✅ |
| BP Train + PC Infer | 96.67% | ✅ |
| **PC Learn (Adam) + PC Infer** | **97.78%** | **✅ (Matches BP)** |

### Key Findings
1.  **Equivalence Validated**: Pure PC Learning (with Adam optimizer) achieves **97.78% accuracy** on digit classification, perfectly matching standard backpropagation. This empirically validates the theoretical equivalence described by Whittington & Bogacz (2017).
2.  **Superior Sine Fit**: PC Learning actually achieved a **lower MSE** (0.000677) than BP training (0.006092) on the sine fitting task.
3.  **Optimizer Dependence**: Local Adam optimization is **critical** for PC learning convergence. Pure SGD failed to converge on the sine task even after 2000 epochs.
4.  **Staleness Matters**: The `pre_acts` staleness fix improved PC learning MSE from 0.001051 to 0.000677.

## Project Timeline & Completion Log

| Phase | Task | Status | Date |
| :--- | :--- | :--- | :--- |
| 0 | Project Initialization & Documentation | ✅ Complete | 2026-04-18 |
| 1 | Core Module Implementation & Bug Fixes | ✅ Complete | 2026-04-18 |
| 2 | Experiments 1–3 (BP Train + PC Infer) | ✅ Complete | 2026-04-18 |
| 3 | Experiment 4 (Pure PC Learning) | ✅ Complete | 2026-04-18 |
| 4 | Deep Code Review & Staleness Fixes | ✅ Complete | 2026-04-18 |

## Future Work / Improvements

- [ ] Validate language modeling generalization using the full Shakespeare text corpus.
- [ ] Scale validation to full MNIST dataset (replace `sklearn` digits).
- [ ] Investigate robustness of PC inference to noisy/missing input modalities.
- [ ] Extend verification to deeper networks (>2 hidden layers).
- [ ] Apply pure PC learning to the character-level language modeling task.

## References

1.  Whittington, J. C., & Bogacz, R. (2017). *An Approximation of the Error Backpropagation Algorithm in a Predictive Coding Network with Local Hebbian Synaptic Plasticity*. Neural Computation.
2.  Millidge, B., Tschantz, A., & Buckley, C. L. (2022). *Predictive Coding Approximates Backprop along Arbitrary Computation Graphs*. Neural Computation.
3.  Salvatori, T., et al. (2023). *Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?* arXiv preprint.
```
