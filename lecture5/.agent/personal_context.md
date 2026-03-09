# Personal Context Rules

## Language Preference

- **Always respond in English**, regardless of whether the user writes in Korean, English, or a mix (Konglish).
- When the user writes in Korean or Konglish:
  1. First, provide the most natural, native-level **English rephrasing** of what they said.
  2. Then, answer the question or continue the conversation entirely in English.

- **Voice/Speech Input**: If the user uses voice input or explicitly asks, evaluate their pronunciation, intonation (억양), and phrasing. Append a small section providing gentle, constructive feedback on how they can sound more natural.

---


## Lecture 5: Automatic Differentiation — Implementation

### Overview
This folder contains the **Lecture 5** exercise materials for CMU's *Deep Learning Systems* course (10-714). The focus is on **implementing automatic differentiation (autograd)** from scratch as a live coding exercise accompanying the lecture.

### Project Structure
```
lecture5/
├── python/
│   └── needle/              # Needle framework stub for lecture exercises
├── 5_automatic_differentiation_implementation.ipynb  # Main lecture notebook
└── README.md
```

### Key Concepts Covered
- **Forward-mode vs. reverse-mode** automatic differentiation
- Building a **computational graph** with `Value` and `Op` nodes
- **Topological sort** over the graph for correct gradient propagation
- Understanding `compute()` (forward) and `gradient()` (backward) in `TensorOp`
- This is the **conceptual foundation** that hw1 builds on

### Development Notes
- The main material is the Jupyter notebook: `5_automatic_differentiation_implementation.ipynb`
- `python/needle/` contains stubs to fill in during the lecture
- Prefer **clear, structured explanations** with tables and code blocks
- When explaining code, break it down **class by class** or **function by function**

