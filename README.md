# TensorFlow C++ Examples - Final Project

This repository contains example programs demonstrating the utility of the TensorFlow C++ library. The examples showcase fundamental operations, linear regression, and neural network classification using TensorFlow's C++ API.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Building the Examples](#building-the-examples)
- [Examples](#examples)
  - [Basic Operations](#example-1-basic-operations)
  - [Linear Regression](#example-2-linear-regression)
  - [Neural Network Classification](#example-3-neural-network-classification)
- [Project Structure](#project-structure)
- [License](#license)

## Overview

TensorFlow provides a powerful C++ API that allows developers to build and deploy machine learning models in C++ applications. This is particularly useful for:

- **High-performance applications** where C++ is required
- **Embedded systems** and IoT devices
- **Production deployments** requiring low-latency inference
- **Integration** with existing C++ codebases

This project demonstrates three key use cases:
1. Basic tensor operations and mathematical functions
2. Training a linear regression model
3. Building and training a neural network for classification

## Prerequisites

Before building these examples, you need:

- **C++ Compiler** with C++17 support (GCC 7+, Clang 5+, or MSVC 2019+)
- **CMake** version 3.16 or higher
- **TensorFlow C++ Library** (libtensorflow_cc)

### Installing TensorFlow C++ Library

#### Option 1: Build from Source (Recommended for full functionality)

```bash
# Clone TensorFlow repository
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Configure the build
./configure

# Build the C++ library
bazel build //tensorflow:libtensorflow_cc.so

# The library will be in bazel-bin/tensorflow/
```

#### Option 2: Using Pre-built Binaries

Download pre-built TensorFlow C libraries from:
https://www.tensorflow.org/install/lang_c

Note: The C API has limited functionality compared to the full C++ API.

#### Option 3: Using Package Managers

**Ubuntu/Debian:**
```bash
# Add TensorFlow PPA (if available) or build from source
sudo apt-get install libtensorflow-dev
```

**macOS with Homebrew:**
```bash
brew install tensorflow
```

## Building the Examples

### Step 1: Clone the Repository

```bash
git clone https://github.com/JuanDigut/Proyectofinalprogra.git
cd Proyectofinalprogra
```

### Step 2: Create Build Directory

```bash
mkdir build
cd build
```

### Step 3: Configure with CMake

If TensorFlow is installed system-wide:
```bash
cmake ..
```

If TensorFlow is in a custom location:
```bash
cmake -DTENSORFLOW_DIR=/path/to/tensorflow ..
```

Or using environment variable:
```bash
export TENSORFLOW_DIR=/path/to/tensorflow
cmake ..
```

### Step 4: Build

```bash
make -j$(nproc)
```

### Step 5: Run Examples

```bash
# Run basic operations example
./examples/basic_operations

# Run linear regression example
./examples/linear_regression

# Run neural network example
./examples/neural_network
```

## Examples

### Example 1: Basic Operations

**File:** `examples/basic_operations.cpp`

This example demonstrates fundamental TensorFlow C++ operations:

- **Scalar Operations**: Addition, subtraction, multiplication, division
- **Vector Operations**: Element-wise operations on 1D tensors
- **Matrix Operations**: Matrix multiplication, transpose, element-wise operations
- **Tensor Shapes**: Reshaping tensors, flattening
- **Mathematical Functions**: sin, cos, exp, sqrt

**Sample Output:**
```
=== Scalar Operations ===
a = 5.0, b = 3.0
a + b = 8
a - b = 2
a * b = 15
a / b = 1.66667

=== Matrix Operations ===
mat1 @ mat2 (matrix multiplication) =
  [[19, 22],
   [43, 50]]
```

### Example 2: Linear Regression

**File:** `examples/linear_regression.cpp`

This example implements a simple linear regression model that learns to fit a line `y = mx + b` to synthetic training data.

**Key Concepts:**
- Creating trainable variables (weights and biases)
- Defining a loss function (Mean Squared Error)
- Computing gradients for backpropagation
- Gradient descent optimization
- Training loop implementation

**Sample Output:**
```
=== Training ===
Epoch    0 | Loss: 125.3456
Epoch  100 | Loss: 0.2541
Epoch  200 | Loss: 0.2512
...
=== Results ===
Learned slope:     2.4823 (true: 2.5)
Learned intercept: 1.0234 (true: 1.0)
```

### Example 3: Neural Network Classification

**File:** `examples/neural_network.cpp`

This example builds a 2-layer neural network for binary classification on an XOR-like pattern.

**Network Architecture:**
- Input layer: 2 neurons (features)
- Hidden layer: 8 neurons with Sigmoid activation
- Output layer: 1 neuron with Sigmoid activation

**Key Concepts:**
- Multi-layer neural network construction
- Activation functions (Sigmoid)
- Binary Cross-Entropy loss
- Manual backpropagation implementation
- Classification accuracy metrics

**Sample Output:**
```
=== Training ===
Epoch    0 | Loss: 0.6931 | Accuracy: 50.00%
Epoch  200 | Loss: 0.5234 | Accuracy: 72.50%
Epoch  400 | Loss: 0.3156 | Accuracy: 88.00%
...
=== Final Evaluation ===
Final Loss: 0.0823
Final Accuracy: 97.50%
```

## Project Structure

```
Proyectofinalprogra/
├── CMakeLists.txt           # Main CMake configuration
├── README.md                # This file
├── examples/
│   ├── CMakeLists.txt       # Examples CMake configuration
│   ├── basic_operations.cpp # Example 1: Basic tensor operations
│   ├── linear_regression.cpp# Example 2: Linear regression model
│   └── neural_network.cpp   # Example 3: Neural network classifier
├── include/                 # Header files (if needed)
└── src/                     # Source files (if needed)
```

## TensorFlow C++ API Key Concepts

### Scope
The `Scope` class defines a namespace for operations. It helps organize the computational graph.

```cpp
Scope root = Scope::NewRootScope();
```

### Operations
Operations are the nodes in the computational graph:

```cpp
auto a = Const(root, 5.0f);      // Constant
auto b = Placeholder(root, DT_FLOAT);  // Placeholder for input
auto c = Add(root, a, b);        // Addition operation
```

### Session
The `ClientSession` executes operations in the graph:

```cpp
ClientSession session(root);
std::vector<Tensor> outputs;
session.Run({c}, &outputs);
```

### Tensors
Tensors are multi-dimensional arrays:

```cpp
Tensor tensor(DT_FLOAT, TensorShape({3, 4}));
auto matrix = tensor.matrix<float>();
matrix(0, 0) = 1.0f;
```

## Troubleshooting

### Common Issues

1. **TensorFlow not found**
   - Ensure `TENSORFLOW_DIR` is set correctly
   - Check that `libtensorflow_cc.so` exists in the lib directory

2. **Missing headers**
   - Verify TensorFlow include path is correct
   - Ensure you have the full TensorFlow C++ headers (not just C API)

3. **Linking errors**
   - Add TensorFlow lib path to `LD_LIBRARY_PATH`:
     ```bash
     export LD_LIBRARY_PATH=$TENSORFLOW_DIR/lib:$LD_LIBRARY_PATH
     ```

4. **Runtime errors**
   - Ensure you're using compatible TensorFlow version
   - Check CUDA/cuDNN compatibility if using GPU

## License

This project is for educational purposes as part of a final programming project.