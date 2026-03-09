# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Lecture 13 & 14: Hardware Acceleration — Architecture Overview

### Overview
This folder contains the **Lecture 13 and 14** exercise materials for CMU's *Deep Learning Systems* course (10-714). These lectures cover **hardware acceleration architecture** — how to efficiently map deep learning computations onto CPUs and GPUs, and how to build the low-level backend that hw3 implements.

### Project Structure
```
lecture13/
├── python/
│   └── needle/              # Needle framework stub for lecture exercises
├── src/
│   ├── ndarray_backend_cpu.cc    # CPU backend reference/stub (C++)
│   └── ndarray_backend_cuda.cu   # CUDA GPU backend reference/stub
├── 13_hardware_acceleration_architecture_overview.ipynb   # Lecture 13 notebook
├── 14_hardware_acceleration_architecture_overview.ipynb   # Lecture 14 notebook
├── CMakeLists.txt                # Build system for C++/CUDA
├── Makefile
└── README.md
```

### Key Concepts Covered
- **Memory layouts**: row-major vs. column-major, strides, compact vs. non-compact arrays
- **CPU acceleration**: cache tiling, blocking for matmul, BLAS integration
- **CUDA GPU kernels**: thread/block hierarchy, shared memory, parallel reduction
- **NDArray design**: how `ndarray.py` wraps the compiled C++/CUDA extension
- This is the conceptual foundation and reference for **hw3**

### Development Notes
- Two notebooks: lecture 13 covers architecture overview, lecture 14 covers specific optimization techniques
- `src/` contains reference/stub C++ and CUDA code
- Build extensions: `make` (uses CMake)
- Prefer **clear, structured explanations** with tables and code blocks
- When explaining kernels, break down **thread indexing** and **memory access patterns** explicitly
