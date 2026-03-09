# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Project: CMU 10-714 Homework 4

### Overview
This is **Homework 4** of CMU's *Deep Learning Systems* course (10-714). It extends the full `needle` framework (with custom NDArray backend) to support **convolutional neural networks (CNNs)** and **recurrent neural networks (RNNs/LSTMs)**, and trains models like **ResNet-9** on CIFAR-10.

### Project Structure
```
hw4/
├── python/
│   └── needle/
│       ├── autograd.py              # Computational graph + autodiff
│       ├── backend_numpy.py         # NumPy fallback
│       ├── backend_selection.py     # Backend dispatcher (numpy / cpu / cuda)
│       ├── backend_ndarray/
│       │   └── ndarray.py           # Custom NDArray with strides, tiling, etc.
│       ├── nn/
│       │   └── nn_basic.py          # Linear, ReLU, BatchNorm, Conv, RNN, LSTM, etc.
│       ├── ops/                     # Tensor ops (including conv ops)
│       ├── init/                    # Initializers
│       ├── optim.py                 # SGD, Adam
│       └── data/                    # DataLoader, CIFAR-10 dataset
├── src/
│   ├── ndarray_backend_cpu.cc       # CPU backend (C++)
│   └── ndarray_backend_cuda.cu      # CUDA GPU backend
├── apps/
│   ├── simple_ml.py                 # Training utilities
│   └── models.py                    # ResNet-9 architecture definition
├── tests/                           # Unit tests
├── hw4.ipynb                        # Jupyter Notebook with exercises
├── ResNet9.png                      # Architecture diagram
├── CMakeLists.txt                   # Build system
└── Makefile
```

### Key Concepts Covered
- **Conv2D** op: forward (im2col + matmul) and backward (gradient w.r.t. input and weights)
- **RNN / LSTM**: sequential models with hidden state, handling variable-length sequences
- **ResNet-9**: skip connections, BatchNorm, training on CIFAR-10 with GPU acceleration
- **Full training pipeline**: DataLoader → forward → loss → backward → optimizer step

### Development Notes
- Conv and RNN/LSTM layers live in `python/needle/nn/nn_basic.py`
- Build C++/CUDA extensions: `make`
- Run tests: `pytest tests/`
- Train ResNet: `python apps/simple_ml.py` (requires compiled CUDA backend for GPU)
