# CMU 10-714: Deep Learning Systems 프로젝트 요약 (Needle 프레임워크)

이 문서는 'Needle(Necessary Deep Learning)' 프레임워크를 밑바닥부터 구현하며 진행한 주요 프로젝트 및 과제들에 대한 요약입니다.

---

## 📂 HW 0: Softmax Regression
딥러닝 시스템 구축을 위한 워밍업 단계입니다.
- **주요 구현 사항:**
  - Python 및 C++(내부 pybind11 활용)를 이용한 Softmax Regression 구현.
  - MNIST 데이터셋 처리 및 분류기 학습.
  - 행렬 연산을 위한 기초적인 Linear Algebra 라이브러리 연동.

## 📂 HW 1: Automatic Differentiation (AD) Engine
프레임워크의 핵심인 연산 그래프와 자동 미분 엔진을 구축했습니다.
- **주요 구현 사항:**
  - **Computational Graph:** `Node`, `Op`, `Value` 구조를 설계하여 연산의 흐름을 추상화.
  - **Reverse-mode AD:** `compute_gradient_of_variables` 함수를 통해 후진 모드 자동 미분 구현.
  - **Topological Sort:** DAG(Directed Acyclic Graph) 형태의 연산 그래프를 DFS 기반 포스트 오더(Post-order)로 탐색하여 역행 정렬(Reverse Topological Order) 수행.
  - **Gradient Accumulation:** 각 노드에서의 Gradient를 합계(Accumulate)하여 다중 경로에서의 미분을 정확히 계산.
  - **기본 연산자:** Addition, Multiplication, Division, Log, Exp, Power, Reshape, Broadcast, Transpose 등 10개 이상의 기본 연산자와 각 연산의 Gradient 식 구현.

## 📂 HW 2: Neural Network Library
본격적인 딥러닝 구성 요소를 Needle 프레임워크 위에 구축했습니다.
- **주요 구현 사항:**
  - **nn.Module System:** 계층적 연산 수행을 위한 Base Class 설계 및 재귀적인 parameter 및 submodule 관리 (`_unpack_params`).
  - **Initialization:** `kaiming_uniform`, `kaiming_normal` 등 가중치 초기화 기법 구현.
  - **Layers:**
    - `Linear`: Fully Connected 레이어 구현.
    - `ReLU`: 비선형 활성화 함수.
    - `Sequential`: 여러 레이어를 순차적으로 실행하는 컨테이너.
    - `BatchNorm1d`: 학습 시 Mean/Variance의 Exponential Moving Average를 활용한 정규화 구현.
    - `LayerNorm1d`: 단일 샘플 내 피처 정규화.
    - `Dropout`: 학습 시 노드를 확률적으로 생략하는 Regularization 기법.
    - `Residual`: Skip-connection을 이용한 잔차 블록 구현.
  - **Optimization Algorithms:**
    - **SGD:** Momentum 및 Weight Decay가 적용된 버전 구현.
    - **Adam:** 1차/2차 모멘트 추정 및 Bias Correction이 포함된 적응형 학습률 최적화 구현.
  - **Loss Function:** 수치적 안정성을 고려한 `SoftmaxLoss` (`logsoftmax` 활용) 구현.
  - **Data Logistics:** 데이터 처리를 위한 `Dataset` 및 `DataLoader` 시스템 구축.

## 📂 HW 3: Backend & Tensor Operations
성능 최적화를 위해 CPU와 GPU(CUDA) 백엔드를 직접 구현했습니다.
- **주요 구현 사항:**
  - **Memory Layout:** NDArray의 Strides, Offset, Shape 관리 시스템 구축.
  - **Matrix Multiplication (Matmul) 최적화:**
    - **CPU Backend:**
      - **Tiling (Blocking):** 데이터를 TILE 단위(4D array)로 나누어 Cache Locality 극대화.
      - **Loop Reordering:** 루프 순서를 `i -> k -> j`로 변경하여 innermost loop에서 연속적인 메모리 접근(Row-major) 보장.
      - **Vectorization:** `__restrict__` 및 `__builtin_assume_aligned`를 사용하여 컴파일러의 SIMD 최적화 유도.
    - **CUDA Backend:**
      - **Shared Memory Tiling:** 반복되는 Global Memory 접근을 줄이기 위해 데이터를 `__shared__` 메모리에 캐싱.
      - **Cooperative Fetching:** Block 내의 Thread들이 협력하여 Global Memory의 데이터를 Shared Memory로 병렬 로드.
      - **Register Tiling:** 각 Thread가 V x V 크기의 출력을 담당하며, 중간 결과를 Register에 저장하여 연산 효율 개선.
      - **Thread Tiling:** 한 Thread가 여러 결과값을 계산하게 하여 메모리 접근 대비 연산 비중(Arithmetic Intensity) 향상.
  - **CUDA Backend Kernels:** GPU 메모리 관리 및 커널(Reductions, Ewise operations) 작성.
  - **Broadcasting & Compactness:** 효율적인 텐서 조작을 위한 알고리즘 구현.

---
**기술 스택:** Python, C++, CUDA, CMake, Pybind11
**학습 목표:** 프레임워크의 추상화 계층부터 하드웨어 가속기(GPU) 레벨의 최적화까지 딥러닝 시스템의 전 과정을 이해함.
