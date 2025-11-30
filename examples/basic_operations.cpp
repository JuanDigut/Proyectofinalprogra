/**
 * @file basic_operations.cpp
 * @brief Example 1: Basic TensorFlow C++ Operations
 * 
 * This example demonstrates fundamental TensorFlow C++ operations including:
 * - Creating and initializing a TensorFlow session
 * - Creating tensors (constants)
 * - Performing basic mathematical operations
 * - Working with tensor shapes and data types
 * 
 * TensorFlow C++ API provides low-level access to TensorFlow's computational
 * graph functionality, allowing for efficient execution of machine learning
 * and numerical computation tasks.
 */

#include <iostream>
#include <vector>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @brief Demonstrates creation and manipulation of scalar tensors
 * @param root The TensorFlow scope for operations
 * @param session The client session to run operations
 */
void demonstrateScalarOperations(Scope& root, ClientSession& session) {
    std::cout << "\n=== Scalar Operations ===" << std::endl;
    
    // Create scalar constants
    auto a = Const(root, 5.0f);
    auto b = Const(root, 3.0f);
    
    // Perform arithmetic operations
    auto sum = Add(root, a, b);
    auto diff = Sub(root, a, b);
    auto prod = Mul(root, a, b);
    auto quot = Div(root, a, b);
    
    // Run the operations
    std::vector<Tensor> outputs;
    
    TF_CHECK_OK(session.Run({sum, diff, prod, quot}, &outputs));
    
    std::cout << "a = 5.0, b = 3.0" << std::endl;
    std::cout << "a + b = " << outputs[0].scalar<float>()() << std::endl;
    std::cout << "a - b = " << outputs[1].scalar<float>()() << std::endl;
    std::cout << "a * b = " << outputs[2].scalar<float>()() << std::endl;
    std::cout << "a / b = " << outputs[3].scalar<float>()() << std::endl;
}

/**
 * @brief Demonstrates vector and matrix tensor operations
 * @param root The TensorFlow scope for operations
 * @param session The client session to run operations
 */
void demonstrateVectorOperations(Scope& root, ClientSession& session) {
    std::cout << "\n=== Vector Operations ===" << std::endl;
    
    // Create vector tensors
    auto vec1 = Const(root, {{1.0f, 2.0f, 3.0f}});
    auto vec2 = Const(root, {{4.0f, 5.0f, 6.0f}});
    
    // Element-wise operations
    auto vec_sum = Add(root, vec1, vec2);
    auto vec_prod = Mul(root, vec1, vec2);
    
    // Run operations
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({vec_sum, vec_prod}, &outputs));
    
    std::cout << "vec1 = [1, 2, 3]" << std::endl;
    std::cout << "vec2 = [4, 5, 6]" << std::endl;
    
    // Print results
    auto sum_data = outputs[0].flat<float>();
    std::cout << "vec1 + vec2 = [";
    for (int i = 0; i < sum_data.size(); ++i) {
        std::cout << sum_data(i);
        if (i < sum_data.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    auto prod_data = outputs[1].flat<float>();
    std::cout << "vec1 * vec2 = [";
    for (int i = 0; i < prod_data.size(); ++i) {
        std::cout << prod_data(i);
        if (i < prod_data.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Demonstrates matrix operations including multiplication
 * @param root The TensorFlow scope for operations
 * @param session The client session to run operations
 */
void demonstrateMatrixOperations(Scope& root, ClientSession& session) {
    std::cout << "\n=== Matrix Operations ===" << std::endl;
    
    // Create 2x2 matrices
    auto mat1 = Const(root, {{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto mat2 = Const(root, {{5.0f, 6.0f}, {7.0f, 8.0f}});
    
    // Matrix multiplication
    auto mat_mul = MatMul(root, mat1, mat2);
    
    // Element-wise addition
    auto mat_add = Add(root, mat1, mat2);
    
    // Transpose
    auto mat_transpose = Transpose(root, mat1, {1, 0});
    
    // Run operations
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({mat_mul, mat_add, mat_transpose}, &outputs));
    
    std::cout << "mat1 = [[1, 2], [3, 4]]" << std::endl;
    std::cout << "mat2 = [[5, 6], [7, 8]]" << std::endl;
    
    // Print matrix multiplication result
    std::cout << "\nmat1 @ mat2 (matrix multiplication) = " << std::endl;
    auto mul_data = outputs[0].matrix<float>();
    std::cout << "  [[" << mul_data(0, 0) << ", " << mul_data(0, 1) << "]," << std::endl;
    std::cout << "   [" << mul_data(1, 0) << ", " << mul_data(1, 1) << "]]" << std::endl;
    
    // Print element-wise addition result
    std::cout << "\nmat1 + mat2 (element-wise) = " << std::endl;
    auto add_data = outputs[1].matrix<float>();
    std::cout << "  [[" << add_data(0, 0) << ", " << add_data(0, 1) << "]," << std::endl;
    std::cout << "   [" << add_data(1, 0) << ", " << add_data(1, 1) << "]]" << std::endl;
    
    // Print transpose result
    std::cout << "\nTranspose(mat1) = " << std::endl;
    auto trans_data = outputs[2].matrix<float>();
    std::cout << "  [[" << trans_data(0, 0) << ", " << trans_data(0, 1) << "]," << std::endl;
    std::cout << "   [" << trans_data(1, 0) << ", " << trans_data(1, 1) << "]]" << std::endl;
}

/**
 * @brief Demonstrates tensor shape manipulation
 * @param root The TensorFlow scope for operations
 * @param session The client session to run operations
 */
void demonstrateTensorShapes(Scope& root, ClientSession& session) {
    std::cout << "\n=== Tensor Shapes ===" << std::endl;
    
    // Create a tensor with specific shape
    auto tensor_3x4 = Const(root, {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f}
    });
    
    // Reshape operation
    auto reshaped_2x6 = Reshape(root, tensor_3x4, {2, 6});
    auto reshaped_6x2 = Reshape(root, tensor_3x4, {6, 2});
    auto flattened = Reshape(root, tensor_3x4, {-1});
    
    // Get shapes
    auto original_shape = Shape(root, tensor_3x4);
    auto new_shape = Shape(root, reshaped_2x6);
    
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({tensor_3x4, reshaped_2x6, reshaped_6x2, flattened, 
                             original_shape, new_shape}, &outputs));
    
    std::cout << "Original tensor (3x4):" << std::endl;
    auto orig_data = outputs[0].matrix<float>();
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 4; ++j) {
            std::cout << orig_data(i, j);
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nReshaped to (2x6):" << std::endl;
    auto reshape_data = outputs[1].matrix<float>();
    for (int i = 0; i < 2; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 6; ++j) {
            std::cout << reshape_data(i, j);
            if (j < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nFlattened (12 elements):" << std::endl;
    auto flat_data = outputs[3].flat<float>();
    std::cout << "  [";
    for (int i = 0; i < 12; ++i) {
        std::cout << flat_data(i);
        if (i < 11) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Demonstrates mathematical functions (sin, cos, exp, log)
 * @param root The TensorFlow scope for operations
 * @param session The client session to run operations
 */
void demonstrateMathFunctions(Scope& root, ClientSession& session) {
    std::cout << "\n=== Mathematical Functions ===" << std::endl;
    
    // Create input values
    auto x = Const(root, {{0.0f, 0.5f, 1.0f, 1.5f, 2.0f}});
    
    // Apply mathematical functions
    auto sin_x = Sin(root, x);
    auto cos_x = Cos(root, x);
    auto exp_x = Exp(root, x);
    auto sqrt_x = Sqrt(root, Add(root, x, Const(root, 1.0f))); // sqrt(x+1) to avoid sqrt(0)
    
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session.Run({x, sin_x, cos_x, exp_x, sqrt_x}, &outputs));
    
    auto x_data = outputs[0].flat<float>();
    auto sin_data = outputs[1].flat<float>();
    auto cos_data = outputs[2].flat<float>();
    auto exp_data = outputs[3].flat<float>();
    auto sqrt_data = outputs[4].flat<float>();
    
    std::cout << std::fixed;
    std::cout.precision(4);
    
    std::cout << "x values: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << x_data(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "sin(x):   [";
    for (int i = 0; i < 5; ++i) {
        std::cout << sin_data(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "cos(x):   [";
    for (int i = 0; i < 5; ++i) {
        std::cout << cos_data(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "exp(x):   [";
    for (int i = 0; i < 5; ++i) {
        std::cout << exp_data(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "sqrt(x+1):[";
    for (int i = 0; i < 5; ++i) {
        std::cout << sqrt_data(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  TensorFlow C++ Basic Operations Demo   " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Create a root scope for TensorFlow operations
    Scope root = Scope::NewRootScope();
    
    // Check if scope creation was successful
    if (!root.ok()) {
        std::cerr << "Error creating TensorFlow scope: " << root.status().ToString() << std::endl;
        return 1;
    }
    
    // Create a client session to run operations
    ClientSession session(root);
    
    try {
        // Run all demonstrations
        demonstrateScalarOperations(root, session);
        demonstrateVectorOperations(root, session);
        demonstrateMatrixOperations(root, session);
        demonstrateTensorShapes(root, session);
        demonstrateMathFunctions(root, session);
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "  All demonstrations completed successfully!" << std::endl;
        std::cout << "==========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
