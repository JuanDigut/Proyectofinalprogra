/**
 * @file linear_regression.cpp
 * @brief Example 2: Linear Regression using TensorFlow C++
 * 
 * This example demonstrates how to implement a simple linear regression
 * model using TensorFlow's C++ API. The model learns to fit a line
 * y = mx + b to a set of training data points.
 * 
 * Key concepts demonstrated:
 * - Creating variables (trainable parameters)
 * - Defining a loss function (Mean Squared Error)
 * - Computing gradients
 * - Implementing gradient descent optimization
 * - Training loop implementation
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
 * @brief Generates synthetic training data for linear regression
 * 
 * Generates data points following y = true_slope * x + true_intercept + noise
 * 
 * @param num_samples Number of data points to generate
 * @param true_slope The actual slope of the line
 * @param true_intercept The actual y-intercept
 * @param noise_level Standard deviation of Gaussian noise
 * @param x_data Output vector for x values
 * @param y_data Output vector for y values
 */
void generateTrainingData(int num_samples, float true_slope, float true_intercept,
                          float noise_level, std::vector<float>& x_data, 
                          std::vector<float>& y_data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, noise_level);
    std::uniform_real_distribution<float> x_dist(0.0f, 10.0f);
    
    x_data.resize(num_samples);
    y_data.resize(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        x_data[i] = x_dist(gen);
        y_data[i] = true_slope * x_data[i] + true_intercept + noise(gen);
    }
}

/**
 * @brief Creates a tensor from a vector of floats
 * @param data The input vector
 * @return A TensorFlow tensor containing the data
 */
Tensor createTensor(const std::vector<float>& data) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(data.size()), 1}));
    auto tensor_map = tensor.matrix<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        tensor_map(i, 0) = data[i];
    }
    return tensor;
}

/**
 * @brief Main linear regression demonstration
 */
int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  TensorFlow C++ Linear Regression Demo  " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Ground truth parameters (we'll try to learn these)
    const float TRUE_SLOPE = 2.5f;
    const float TRUE_INTERCEPT = 1.0f;
    const float NOISE_LEVEL = 0.5f;
    const int NUM_SAMPLES = 100;
    const int NUM_EPOCHS = 1000;
    const float LEARNING_RATE = 0.01f;
    
    std::cout << "\n=== Data Generation ===" << std::endl;
    std::cout << "True slope: " << TRUE_SLOPE << std::endl;
    std::cout << "True intercept: " << TRUE_INTERCEPT << std::endl;
    std::cout << "Noise level: " << NOISE_LEVEL << std::endl;
    std::cout << "Number of samples: " << NUM_SAMPLES << std::endl;
    
    // Generate training data
    std::vector<float> x_data, y_data;
    generateTrainingData(NUM_SAMPLES, TRUE_SLOPE, TRUE_INTERCEPT, NOISE_LEVEL,
                         x_data, y_data);
    
    // Create tensors from data
    Tensor x_tensor = createTensor(x_data);
    Tensor y_tensor = createTensor(y_data);
    
    std::cout << "Generated " << NUM_SAMPLES << " training samples." << std::endl;
    
    // Create TensorFlow scope
    Scope root = Scope::NewRootScope();
    
    // Create placeholders for input data
    auto x_placeholder = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    auto y_placeholder = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    
    // Create trainable variables (initialized with random values)
    // Weight (slope)
    auto w_init = Variable(root.WithOpName("w"), {1, 1}, DT_FLOAT);
    auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    // Bias (intercept)
    auto b_init = Variable(root.WithOpName("b"), {1, 1}, DT_FLOAT);
    auto b_assign = Assign(root.WithOpName("b_assign"), b_init,
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    // Model: y_pred = x * w + b
    auto y_pred = Add(root.WithOpName("prediction"),
                     MatMul(root, x_placeholder, w_init),
                     b_init);
    
    // Loss function: Mean Squared Error = mean((y_pred - y)^2)
    auto error = Sub(root, y_pred, y_placeholder);
    auto squared_error = Square(root, error);
    auto loss = Mean(root.WithOpName("loss"), squared_error, {0, 1});
    
    // Compute gradients manually
    // d(loss)/dw = 2 * mean(error * x)
    // d(loss)/db = 2 * mean(error)
    auto grad_w = Mul(root, 
                     Const(root, 2.0f),
                     Mean(root, Mul(root, error, x_placeholder), {0}));
    auto grad_b = Mul(root,
                     Const(root, 2.0f),
                     Mean(root, error, {0}));
    
    // Gradient descent update
    auto lr = Const(root, LEARNING_RATE);
    auto w_update = AssignSub(root, w_init, Mul(root, lr, grad_w));
    auto b_update = AssignSub(root, b_init, Mul(root, lr, grad_b));
    
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
        
        // Print progress every 100 epochs
        if (epoch % 100 == 0 || epoch == NUM_EPOCHS - 1) {
            float current_loss = outputs[0].scalar<float>()();
            std::cout << "Epoch " << std::setw(4) << epoch 
                      << " | Loss: " << current_loss << std::endl;
        }
    }
    
    // Get final learned parameters
    std::vector<Tensor> final_params;
    TF_CHECK_OK(session.Run({w_init, b_init}, &final_params));
    
    float learned_slope = final_params[0].matrix<float>()(0, 0);
    float learned_intercept = final_params[1].matrix<float>()(0, 0);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Learned slope:     " << learned_slope 
              << " (true: " << TRUE_SLOPE << ")" << std::endl;
    std::cout << "Learned intercept: " << learned_intercept 
              << " (true: " << TRUE_INTERCEPT << ")" << std::endl;
    
    // Calculate error
    float slope_error = std::abs(learned_slope - TRUE_SLOPE);
    float intercept_error = std::abs(learned_intercept - TRUE_INTERCEPT);
    
    std::cout << "\nError in slope:     " << slope_error << std::endl;
    std::cout << "Error in intercept: " << intercept_error << std::endl;
    
    // Make predictions on new data
    std::cout << "\n=== Predictions ===" << std::endl;
    std::cout << "Testing on new x values:" << std::endl;
    
    std::vector<float> test_x = {0.0f, 2.5f, 5.0f, 7.5f, 10.0f};
    Tensor test_tensor = createTensor(test_x);
    
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
        float true_y = TRUE_SLOPE * test_x[i] + TRUE_INTERCEPT;
        std::cout << std::setw(8) << test_x[i] << " | "
                  << std::setw(12) << pred_data(i, 0) << " | "
                  << std::setw(12) << true_y << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  Linear regression completed successfully!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
