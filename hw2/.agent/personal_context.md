# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Project: CMU 10-714 Homework 2

### Overview
This is **Homework 2** of CMU's *Deep Learning Systems* course (10-714). It extends the `needle` framework from hw1 by adding **neural network modules (`nn`)**, **optimizers**, **data loading**, and additional ops — enabling end-to-end training of networks like MLP and ResNet.

### Project Structure
```
hw2/
├── python/
│   └── needle/
│       ├── __init__.py          # Package exports
│       ├── autograd.py          # Inherited from hw1: computational graph + reverse-mode autodiff
│       ├── backend_numpy.py     # NumPy-backed array device
│       ├── optim.py             # Optimizers: SGD, Adam
│       ├── ops/
│       │   ├── __init__.py
│       │   ├── ops_mathematic.py  # Core math ops (from hw1, extended)
│       │   ├── ops_logarithmic.py # Log, exp ops
│       │   └── ops_tuple.py       # TensorTuple ops
│       ├── nn/
│       │   ├── __init__.py
│       │   └── nn_basic.py        # nn.Module, Linear, ReLU, Sequential, BatchNorm, LayerNorm, etc.
│       ├── init/                # Tensor initializers
│       └── data/                # Dataset and DataLoader utilities
├── apps/
│   └── mlp_resnet.py            # MLP / ResNet training script using needle
├── tests/                       # Unit tests (pytest)
├── hw2.ipynb                    # Jupyter Notebook with exercises
└── data/                        # Datasets (e.g., MNIST)
```

### Key Concepts Covered
- **`nn.Module`** — base class for all neural network layers
- **`nn.Linear`, `nn.ReLU`, `nn.Sequential`** — standard building blocks
- **`nn.BatchNorm1d`, `nn.LayerNorm1d`** — normalization layers
- **Optimizers**: `optim.SGD`, `optim.Adam` — parameter update rules using needle tensors
- **Data pipeline**: `Dataset`, `DataLoader` for batched iteration over data
- **Extended ops**: logarithmic and tuple ops for more complex architectures

### Development Notes
- Neural network layers live in `python/needle/nn/nn_basic.py`
- Each `nn.Module` must implement `forward(self, X)`, no explicit `backward()` needed
- Optimizer `step()` uses `tensor.data` (detached) to avoid grad tracking on param updates
- Run tests: `pytest tests/`
- Training script: `python apps/mlp_resnet.py`
