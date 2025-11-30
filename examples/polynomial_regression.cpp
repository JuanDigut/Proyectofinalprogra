/**
 * @file polynomial_regression.cpp
 * @brief Example 4: Polynomial Regression using TensorFlow C++
 * 
 * This example demonstrates how to implement polynomial regression
 * using TensorFlow's C++ API. The model learns to fit a polynomial
 * function y = 0.5x³ - x² + 0.5x + 1 to a set of training data points.
 * 
 * Key concepts demonstrated:
 * - Creating tensors and operations
 * - Building computational graphs
 * - Training with gradient descent
 * - Using variables and placeholders
 * - Feature engineering (polynomial features)
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @brief Generates synthetic training data for polynomial regression
 * 
 * Generates data points following y = 0.5x³ - x² + 0.5x + 1 + noise
 * 
 * @param num_samples Number of data points to generate
 * @param noise_level Standard deviation of Gaussian noise
 * @param x_data Output vector for x values
 * @param y_data Output vector for y values
 */
void generatePolynomialData(int num_samples, float noise_level,
                            std::vector<float>& x_data,
                            std::vector<float>& y_data) {
    std::random_device rd;
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> noise(0.0f, noise_level);
    std::uniform_real_distribution<float> x_dist(-2.0f, 2.0f);
    
    x_data.resize(num_samples);
    y_data.resize(num_samples);
    
    // True coefficients: y = 0.5x³ - x² + 0.5x + 1
    const float c3 = 0.5f;   // x³ coefficient
    const float c2 = -1.0f;  // x² coefficient
    const float c1 = 0.5f;   // x coefficient
    const float c0 = 1.0f;   // intercept
    
    for (int i = 0; i < num_samples; ++i) {
        float x = x_dist(gen);
        x_data[i] = x;
        float x2 = x * x;
        float x3 = x2 * x;
        y_data[i] = c3 * x3 + c2 * x2 + c1 * x + c0 + noise(gen);
    }
}

/**
 * @brief Creates a tensor with polynomial features [x, x², x³]
 * 
 * @param x_data Input x values
 * @return TensorFlow tensor with shape (num_samples, 3)
 */
Tensor createPolynomialFeatures(const std::vector<float>& x_data) {
    int num_samples = static_cast<int>(x_data.size());
    Tensor tensor(DT_FLOAT, TensorShape({num_samples, 3}));
    auto tensor_map = tensor.matrix<float>();
    
    for (int i = 0; i < num_samples; ++i) {
        float x = x_data[i];
        tensor_map(i, 0) = x;           // x
        tensor_map(i, 1) = x * x;       // x²
        tensor_map(i, 2) = x * x * x;   // x³
    }
    return tensor;
}

/**
 * @brief Creates a tensor from y values
 * 
 * @param y_data Input y values
 * @return TensorFlow tensor with shape (num_samples, 1)
 */
Tensor createLabelTensor(const std::vector<float>& y_data) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(y_data.size()), 1}));
    auto tensor_map = tensor.matrix<float>();
    for (size_t i = 0; i < y_data.size(); ++i) {
        tensor_map(i, 0) = y_data[i];
    }
    return tensor;
}

/**
 * @brief Main polynomial regression demonstration
 */
int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  TensorFlow C++ Polynomial Regression   " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Ground truth parameters (we'll try to learn these)
    // y = 0.5x³ - x² + 0.5x + 1
    const float TRUE_C3 = 0.5f;   // x³ coefficient
    const float TRUE_C2 = -1.0f;  // x² coefficient
    const float TRUE_C1 = 0.5f;   // x coefficient
    const float TRUE_C0 = 1.0f;   // intercept
    
    const float NOISE_LEVEL = 0.1f;
    const int NUM_SAMPLES = 100;
    const int NUM_EPOCHS = 2000;
    const float LEARNING_RATE = 0.01f;
    
    std::cout << "\n=== Polynomial Function ===" << std::endl;
    std::cout << "y = " << TRUE_C3 << "x³ + (" << TRUE_C2 << ")x² + " 
              << TRUE_C1 << "x + " << TRUE_C0 << std::endl;
    std::cout << "Noise level: " << NOISE_LEVEL << std::endl;
    std::cout << "Number of samples: " << NUM_SAMPLES << std::endl;
    
    // Generate training data
    std::vector<float> x_data, y_data;
    generatePolynomialData(NUM_SAMPLES, NOISE_LEVEL, x_data, y_data);
    
    // Create tensors from data
    // Features: [x, x², x³]
    Tensor x_tensor = createPolynomialFeatures(x_data);
    Tensor y_tensor = createLabelTensor(y_data);
    
    std::cout << "Generated " << NUM_SAMPLES << " training samples." << std::endl;
    std::cout << "Feature shape: [" << x_tensor.dim_size(0) << ", " 
              << x_tensor.dim_size(1) << "]" << std::endl;
    
    // Create TensorFlow scope
    Scope root = Scope::NewRootScope();
    
    // Create placeholders for input data
    // x_placeholder will receive polynomial features [x, x², x³]
    auto x_placeholder = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                     Placeholder::Shape({-1, 3}));
    auto y_placeholder = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    
    // Create trainable variables
    // Weights for polynomial coefficients [c1, c2, c3] (for x, x², x³)
    auto w_init = Variable(root.WithOpName("weights"), {3, 1}, DT_FLOAT);
    auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                          RandomNormal(root, {3, 1}, DT_FLOAT));
    
    // Bias (intercept c0)
    auto b_init = Variable(root.WithOpName("bias"), {1, 1}, DT_FLOAT);
    auto b_assign = Assign(root.WithOpName("b_assign"), b_init,
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    // Model: y_pred = x * w + b
    // Where x contains polynomial features [x, x², x³]
    auto y_pred = Add(root.WithOpName("prediction"),
                     MatMul(root, x_placeholder, w_init),
                     b_init);
    
    // Loss function: Mean Squared Error = mean((y_pred - y)²)
    auto error = Sub(root, y_pred, y_placeholder);
    auto squared_error = Square(root, error);
    auto loss = Mean(root.WithOpName("loss"), squared_error, {0, 1});
    
    // Compute gradients manually
    // d(loss)/dw = 2 * mean(error * x)
    // d(loss)/db = 2 * mean(error)
    auto grad_w = Mul(root, 
                     Const(root, 2.0f),
                     Mean(root, Mul(root, 
                         Reshape(root, error, {-1, 1}),
                         x_placeholder), {0}));
    auto grad_b = Mul(root,
                     Const(root, 2.0f),
                     Mean(root, error, {0}));
    
    // Gradient descent update
    auto lr = Const(root, LEARNING_RATE);
    auto w_update = AssignSub(root, w_init, 
                             Mul(root, lr, Reshape(root, grad_w, {3, 1})));
    auto b_update = AssignSub(root, b_init, 
                             Mul(root, lr, Reshape(root, grad_b, {1, 1})));
    
    // Create session
    ClientSession session(root);
    
    // Initialize variables
    TF_CHECK_OK(session.Run({w_assign, b_assign}, nullptr));
    
    std::cout << "\n=== Training ===" << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Number of epochs: " << NUM_EPOCHS << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Training loop
    std::vector<Tensor> outputs;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // Run training step
        TF_CHECK_OK(session.Run(
            {{x_placeholder, x_tensor}, {y_placeholder, y_tensor}},
            {loss, w_update, b_update},
            &outputs));
        
        // Print progress every 200 epochs
        if (epoch % 200 == 0 || epoch == NUM_EPOCHS - 1) {
            float current_loss = outputs[0].scalar<float>()();
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << " | Loss: " << current_loss << std::endl;
        }
    }
    
    // Get final learned parameters
    std::vector<Tensor> final_params;
    TF_CHECK_OK(session.Run({w_init, b_init}, &final_params));
    
    auto weights = final_params[0].matrix<float>();
    float learned_c1 = weights(0, 0);  // x coefficient
    float learned_c2 = weights(1, 0);  // x² coefficient
    float learned_c3 = weights(2, 0);  // x³ coefficient
    float learned_c0 = final_params[1].matrix<float>()(0, 0);  // intercept
    
    std::cout << "\n=== Learned Coefficients ===" << std::endl;
    std::cout << "Coefficient  | Learned     | True" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << "x³ (c3)      | " << std::setw(10) << learned_c3 
              << " | " << TRUE_C3 << std::endl;
    std::cout << "x² (c2)      | " << std::setw(10) << learned_c2 
              << " | " << TRUE_C2 << std::endl;
    std::cout << "x  (c1)      | " << std::setw(10) << learned_c1 
              << " | " << TRUE_C1 << std::endl;
    std::cout << "intercept    | " << std::setw(10) << learned_c0 
              << " | " << TRUE_C0 << std::endl;
    
    // Make predictions on new data
    std::cout << "\n=== Predictions ===" << std::endl;
    std::cout << "Testing on new x values:" << std::endl;
    
    std::vector<float> test_x = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor test_tensor = createPolynomialFeatures(test_x);
    
    std::vector<Tensor> predictions;
    TF_CHECK_OK(session.Run(
        {{x_placeholder, test_tensor}},
        {y_pred},
        &predictions));
    
    auto pred_data = predictions[0].matrix<float>();
    std::cout << std::setw(8) << "x" << " | " 
              << std::setw(12) << "Predicted" << " | "
              << std::setw(12) << "True" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (size_t i = 0; i < test_x.size(); ++i) {
        float x = test_x[i];
        float true_y = TRUE_C3 * x * x * x + TRUE_C2 * x * x + TRUE_C1 * x + TRUE_C0;
        std::cout << std::setw(8) << x << " | "
                  << std::setw(12) << pred_data(i, 0) << " | "
                  << std::setw(12) << true_y << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  Polynomial regression completed!       " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
