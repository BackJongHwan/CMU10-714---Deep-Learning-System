# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Project: CMU 10-714 Homework 0

### Overview
This is **Homework 0** of CMU's *Deep Learning Systems* course (10-714). It covers fundamental machine learning primitives implemented from scratch in NumPy — no deep learning framework is used.

### Project Structure
```
hw0/
├── src/
│   ├── simple_ml.py         # Main implementation file (softmax, loss, SGD, batched training)
│   └── simple_ml_ext.cpp    # C++ extension for performance-critical ops
├── tests/
│   └── test_simple_ml.py    # Unit tests (run via pytest / Makefile)
├── data/                    # MNIST dataset files
├── hw0.ipynb                # Jupyter Notebook with exercises and visualizations
└── Makefile                 # Build/test commands
```

### Key Concepts Covered
- **Softmax** and **cross-entropy loss** (numerically stable implementation)
- **Stochastic Gradient Descent (SGD)** for logistic regression
- **Batched forward/backward passes**
- **C++ extension** (`simple_ml_ext.cpp`) for efficient matrix ops interfacing with Python

### Development Notes
- Main implementation work lives in `src/simple_ml.py`
- Tests are run with `pytest tests/` or `make test`
- NumPy is the only dependency for the core logic
