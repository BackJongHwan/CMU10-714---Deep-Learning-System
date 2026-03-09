# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Project: CMU 10-714 Homework 1

### Overview
This is **Homework 1** of CMU's *Deep Learning Systems* course (10-714). It focuses on building the **`needle`** deep learning framework from scratch — specifically its **automatic differentiation (autograd)** engine and core tensor operations.

### Project Structure
```
hw1/
├── python/
│   └── needle/
│       ├── __init__.py          # Package exports
│       ├── autograd.py          # Core: Value, Op, Tensor, TensorOp, computational graph + backprop
│       ├── backend_numpy.py     # NumPy-backed array device
│       ├── ops/
│       │   ├── __init__.py
│       │   └── ops_mathematic.py  # Math ops: EWiseAdd, AddScalar, MatMul, Reshape, etc.
│       └── init/                # Tensor initializers (xavier, kaiming, etc.)
├── apps/
│   └── simple_ml.py             # High-level training utilities using needle
├── tests/                       # Unit tests (pytest)
├── hw1.ipynb                    # Jupyter Notebook with exercises
└── data/                        # Datasets
```

### Key Concepts Covered
- **Computational graph** construction via `Value` and `Op` classes
- **`Tensor`** and **`TensorOp`** — the primary user-facing abstractions
- **Reverse-mode autodiff**: `backward()`, gradient accumulation, topological sort
- **Element-wise and matrix ops**: implemented as `TensorOp` subclasses with `compute()` + `gradient()` methods

### Development Notes
- Core autograd logic is in `python/needle/autograd.py`
- New ops are added in `python/needle/ops/ops_mathematic.py`
- Each op must implement:
  - `compute(self, *args)` → forward pass using raw NDArray
  - `gradient(self, out_grad, node)` → backward pass returning `Tensor` gradients
- Run tests: `pytest tests/`
