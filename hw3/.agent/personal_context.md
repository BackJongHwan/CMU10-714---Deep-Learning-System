# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Project: CMU 10-714 Homework 3

### Overview
This is **Homework 3** of CMU's *Deep Learning Systems* course (10-714). It focuses on implementing a **custom NDArray backend** — a low-level multi-dimensional array library with CPU and CUDA support — that replaces NumPy under the `needle` framework.

### Project Structure
```
hw3/
├── python/
│   └── needle/
│       ├── autograd.py              # Computational graph + reverse-mode autodiff
│       ├── backend_numpy.py         # NumPy fallback backend
│       ├── backend_selection.py     # Selects between numpy / cpu / cuda backends
│       ├── backend_ndarray/
│       │   ├── ndarray.py           # NDArray class: tiling, striding, reshape, broadcast
│       │   └── ndarray_backend_numpy.py  # Pure-Python reference backend
│       ├── nn/
│       │   └── nn_basic.py          # nn.Module, Linear, ReLU, BatchNorm, etc.
│       ├── ops/                     # Tensor ops with compute() + gradient()
│       ├── init/                    # Tensor initializers
│       ├── optim.py                 # SGD, Adam
│       └── data/                    # DataLoader
├── src/
│   ├── ndarray_backend_cpu.cc       # C++ CPU backend (BLAS, tiling, striding)
│   └── ndarray_backend_cuda.cu      # CUDA GPU backend (parallel kernels)
├── tests/                           # Unit tests (pytest)
├── hw3.ipynb                        # Jupyter Notebook with exercises
├── CMakeLists.txt                   # Build system for C++/CUDA extensions
└── Makefile                         # Build shortcuts
```

### Key Concepts Covered
- **Strided NDArray**: compact/non-compact memory layouts, shape/strides/offset
- **Broadcasting** and **reshape** via stride manipulation (no data copy)
- **CPU backend** (`ndarray_backend_cpu.cc`): dense matrix multiply with tiling/BLAS
- **CUDA backend** (`ndarray_backend_cuda.cu`): GPU kernels for elementwise ops, matmul, reduction
- **Backend selection**: `needle.backend_ndarray.ndarray` dispatches ops to cpu or cuda

### Development Notes
- Main implementation work: `python/needle/backend_ndarray/ndarray.py` + `src/ndarray_backend_cpu.cc` / `ndarray_backend_cuda.cu`
- Build C++ extensions: `make` (uses CMake)
- Run tests: `pytest tests/`
- Switch backends via `needle.init_backend("cpu")` or `needle.init_backend("cuda")`
