# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Project: CMU 10-714 Homework 4 (Extra Credit)

### Overview
This is the **extra credit extension** of Homework 4 for CMU's *Deep Learning Systems* course (10-714). It builds on the full `needle` framework (including the custom NDArray CPU/CUDA backend) and explores additional advanced topics beyond the main hw4 scope.

### Project Structure
```
hw4_extra/
├── python/
│   └── needle/
│       ├── autograd.py              # Computational graph + autodiff (same as hw4)
│       ├── backend_ndarray/         # Custom NDArray backend (strided, CPU/CUDA)
│       ├── backend_selection.py     # Backend dispatcher
│       ├── nn/                      # nn.Module and layers
│       ├── ops/                     # Tensor ops
│       ├── init/                    # Initializers
│       ├── optim.py                 # Optimizers
│       └── data/                    # DataLoader
├── src/
│   ├── ndarray_backend_cpu.cc       # CPU backend (C++)
│   └── ndarray_backend_cuda.cu      # CUDA GPU backend
├── apps/                            # Application scripts
├── tests/                           # Unit tests
├── hw4_extra.ipynb                  # Notebook with extra-credit exercises
├── CMakeLists.txt
└── Makefile
```

### Key Concepts Covered
- Extends hw4 with additional advanced model components or optimizations (see notebook for specifics)
- Same underlying `needle` framework stack as hw4 (NDArray backend, autograd, nn, optim)

### Development Notes
- Implementation target varies per exercise — refer to `hw4_extra.ipynb` for the exact tasks
- Build C++/CUDA extensions: `make`
- Run tests: `pytest tests/`
