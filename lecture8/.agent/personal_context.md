# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Lecture 8: Neural Network Library — Implementation

### Overview
This folder contains the **Lecture 8** exercise materials for CMU's *Deep Learning Systems* course (10-714). The focus is on **building a neural network module system** (`nn.Module`) on top of the `needle` autograd engine — the pattern that hw2 formalizes into a full assignment.

### Project Structure
```
lecture8/
├── python/
│   └── needle/              # Needle framework stub for lecture exercises
├── 8_nn_library_implementation.ipynb   # Main lecture notebook
└── README.md
```

### Key Concepts Covered
- **`nn.Module`** design pattern: `parameters()`, `forward()`, `__call__()`
- Building reusable layers: `Linear`, `ReLU`, `Sequential`
- **Parameter management**: how modules own and expose their `Tensor` parameters
- Connecting `nn.Module` to the autograd engine (no explicit `backward()` in user code)
- This supplements and motivates the hw2 implementation

### Development Notes
- The main material is the Jupyter notebook: `8_nn_library_implementation.ipynb`
- `python/needle/` contains stubs to fill in during the lecture
- Prefer **clear, structured explanations** with tables and code blocks
- When explaining code, break it down **class by class** or **function by function**
