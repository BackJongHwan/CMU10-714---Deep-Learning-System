# Windows Local Setup Guide (Colab → Local)

This document describes all changes made to run `hw3`, `hw4`, and `hw4_extra`
on a local **Windows** machine instead of Google Colab.

---

## Background

The original homework was designed to run on Google Colab (Linux/Ubuntu).
The C++/CUDA backends need to be compiled, and Colab uses `gcc`/`clang` + `make`.
On Windows, Microsoft's **MSVC** compiler and **MSBuild** are used instead.

---

## Changes Made (Applied to hw3, hw4, hw4_extra)

### 1. `CMakeLists.txt` — Build System Fixes

| Change | Reason |
|---|---|
| Removed `python3-config --prefix` call | `python3-config` doesn't exist on Windows |
| Replaced `python3 -m pybind11` with `${Python_EXECUTABLE} -m pybind11` | `python3` command doesn't exist on Windows |
| Added `${Python_LIBRARIES}` to `LINKER_LIBS` | MSVC linker needs explicit `python312.lib` path |
| Added `LIBRARY_OUTPUT_DIRECTORY_RELEASE` and `LIBRARY_OUTPUT_DIRECTORY_DEBUG` | On Windows, MSVC puts `.pyd` in a `Release/` subfolder by default without these |
| Replaced `/std:c++11 -march=native` with `/O2 /wd4819 /utf-8` | `/std:c++11` is invalid in MSVC; `/wd4819` suppresses Korean charset warnings; `/utf-8` fixes encoding |

### 2. `src/ndarray_backend_cpu.cc` — Windows C++ Compatibility

| Change | Reason |
|---|---|
| Replaced `posix_memalign()` with `_aligned_malloc()` | `posix_memalign` is POSIX-only (Linux/macOS); Windows uses `_aligned_malloc` |
| Replaced `free()` with `_aligned_free()` | Memory allocated with `_aligned_malloc` must be freed with `_aligned_free` |
| Added MSVC macros for `__restrict__` and `__builtin_assume_aligned` | These are GCC-specific compiler hints; MSVC uses `__restrict` (no underscores) |
| Added `#include <algorithm>` and `#include <numeric>` | MSVC strictly requires explicit includes for `std::max_element`, `std::accumulate` |

> Both changes use `#ifdef _WIN32` guards, so the code still compiles on Linux/Colab.

### 3. `src/ndarray_backend_cuda.cu` — CUDA Kernel Fixes (hw3 only)

| Change | Reason |
|---|---|
| Fixed `__global__void` → `__global__ void` (missing space) | MSVC is stricter about token separation |
| Fixed `redice_size` → `reduce_size` | Typo in variable name |
| Fixed `<<dim.grid` → `<<<dim.grid` | Missing `<` in CUDA kernel launch syntax |
| Fixed `CudaDim` → `CudaDims` | Wrong struct name |

### 4. `python/needle/__init__.py` — Import Fix

| Change | Reason |
|---|---|
| Commented out `from . import data` | The `data` module does not exist in hw3/hw4/hw4_extra yet, causing a fake "circular import" crash |

### 5. `hw3.ipynb` / `hw4.ipynb` / `hw4_extra.ipynb` — Notebook Build Cell

| Change | Reason |
|---|---|
| Replaced `!make` with cmake commands using `!{_cmake}` | `make` is a Linux tool; not available on Windows |
| Added `os.environ["VSLANG"] = "1033"` | Forces MSVC to display errors in English instead of Korean |

The build cell in all notebooks now looks like:
```python
import sys, os
os.environ["VSLANG"] = "1033"  # Force MSVC to output errors in English
_cmake = os.path.join(os.path.dirname(sys.executable), "Scripts", "cmake.exe")
if not os.path.exists(_cmake): _cmake = "cmake"
os.makedirs("build", exist_ok=True)
!{_cmake} -B build
!{_cmake} --build build --config Release
```

---

## How to Build Locally (Without Notebook)

Run these commands from the homework directory (e.g. `hw3/`):

```powershell
# 1. Install dependencies (one-time)
pip install cmake pybind11

# 2. Configure
$cmake = "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\cmake.exe"
& $cmake -B build

# 3. Build
& $cmake --build build --config Release
```

The compiled `.pyd` files will appear in:
```
python/needle/backend_ndarray/ndarray_backend_cpu.cp312-win_amd64.pyd
python/needle/backend_ndarray/ndarray_backend_cuda.cp312-win_amd64.pyd
```

## How to Run Tests

```powershell
$env:PYTHONPATH = ".\python"
python -m pytest -v tests/
```

---

## Requirements

- Python 3.12
- Visual Studio 2022 (MSVC)
- CUDA Toolkit 12.x
- `pip install cmake pybind11 pytest numpy`
