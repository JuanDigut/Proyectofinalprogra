/**
 * @file time_series_prediction.cpp
 * @brief Example 7: Time Series Prediction using TensorFlow C++
 * 
 * This example demonstrates how to implement time series prediction using
 * a feedforward neural network with TensorFlow's C++ API.
 * 
 * The example generates synthetic time series data (sine wave with trend and noise)
 * and trains a neural network to predict the next value based on a sliding window
 * of previous values.
 * 
 * Key concepts demonstrated:
 * - Time series data preparation (sliding window / windowing)
 * - Feedforward network for sequence prediction
 * - Regression metrics (MAE, RMSE)
 * - Handling temporal dependencies with fixed-size input windows
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;
using namespace tensorflow::ops;

/**
 * @brief Generates synthetic time series data
 * 
 * Creates a time series: y(t) = sin(2*pi*t/period) + trend*t + noise
 * 
 * @param length Number of time points
 * @param period Period of the sine wave
 * @param trend Linear trend coefficient
 * @param noise_level Standard deviation of Gaussian noise
 * @param data Output vector for time series values
 */
void generateTimeSeries(int length, float period, float trend, 
                        float noise_level, std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> noise(0.0f, noise_level);
    
    data.clear();
    const float PI = 3.14159265358979323846f;
    
    for (int t = 0; t < length; ++t) {
        float value = std::sin(2.0f * PI * t / period) + trend * t / length + noise(gen);
        data.push_back(value);
    }
}

/**
 * @brief Creates sliding window features and targets from time series
 * 
 * @param series Input time series
 * @param window_size Number of previous points to use as features
 * @param x_data Output feature vectors (windows)
 * @param y_data Output target values (next value after each window)
 */
void createSlidingWindows(const std::vector<float>& series, int window_size,
                          std::vector<float>& x_data, std::vector<float>& y_data) {
    x_data.clear();
    y_data.clear();
    
    int num_windows = series.size() - window_size;
    
    for (int i = 0; i < num_windows; ++i) {
        // Window features
        for (int j = 0; j < window_size; ++j) {
            x_data.push_back(series[i + j]);
        }
        // Target: next value after window
        y_data.push_back(series[i + window_size]);
    }
}

/**
 * @brief Creates a 2D tensor from flattened data
 * @param data Flat vector of features
 * @param num_samples Number of samples
 * @param num_features Number of features per sample
 * @return TensorFlow tensor
 */
Tensor createTensor2D(const std::vector<float>& data, 
                      int num_samples, int num_features) {
    Tensor tensor(DT_FLOAT, TensorShape({num_samples, num_features}));
    auto tensor_map = tensor.matrix<float>();
    for (int i = 0; i < num_samples; ++i) {
        for (int j = 0; j < num_features; ++j) {
            tensor_map(i, j) = data[i * num_features + j];
        }
    }
    return tensor;
}

/**
 * @brief Creates a 1D column tensor from vector
 * @param data Vector of values
 * @return TensorFlow tensor (N x 1)
 */
Tensor createTensor1D(const std::vector<float>& data) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(data.size()), 1}));
    auto tensor_map = tensor.matrix<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        tensor_map(i, 0) = data[i];
    }
    return tensor;
}

/**
 * @brief Calculates Mean Absolute Error
 * @param predictions Predicted values tensor
 * @param targets Target values tensor
 * @return MAE value
 */
float calculateMAE(const Tensor& predictions, const Tensor& targets) {
    auto pred_data = predictions.matrix<float>();
    auto target_data = targets.matrix<float>();
    int n = predictions.dim_size(0);
    
    float mae = 0.0f;
    for (int i = 0; i < n; ++i) {
        mae += std::abs(pred_data(i, 0) - target_data(i, 0));
    }
    return mae / n;
}

/**
 * @brief Calculates Root Mean Squared Error
 * @param predictions Predicted values tensor
 * @param targets Target values tensor
 * @return RMSE value
 */
float calculateRMSE(const Tensor& predictions, const Tensor& targets) {
    auto pred_data = predictions.matrix<float>();
    auto target_data = targets.matrix<float>();
    int n = predictions.dim_size(0);
    
    float mse = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = pred_data(i, 0) - target_data(i, 0);
        mse += diff * diff;
    }
    return std::sqrt(mse / n);
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "       Time Series Prediction            " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Hyperparameters
    const int SERIES_LENGTH = 500;
    const int WINDOW_SIZE = 10;
    const float SINE_PERIOD = 50.0f;
    const float TREND = 0.5f;
    const float NOISE_LEVEL = 0.1f;
    
    const int NUM_EPOCHS = 1000;
    const float LEARNING_RATE = 0.01f;
    
    // Network architecture: window_size -> 32 -> 16 -> 1
    const int HIDDEN1_SIZE = 32;
    const int HIDDEN2_SIZE = 16;
    const int OUTPUT_SIZE = 1;
    
    std::cout << "\n=== Generating Time Series ===" << std::endl;
    
    // Generate time series
    std::vector<float> series;
    generateTimeSeries(SERIES_LENGTH, SINE_PERIOD, TREND, NOISE_LEVEL, series);
    
    std::cout << "Length: " << SERIES_LENGTH << " points" << std::endl;
    std::cout << "Wave period: " << SINE_PERIOD << std::endl;
    std::cout << "Trend: " << TREND << std::endl;
    std::cout << "Noise level: " << NOISE_LEVEL << std::endl;
    std::cout << "Window size: " << WINDOW_SIZE << std::endl;
    
    // Create sliding windows
    std::vector<float> x_data, y_data;
    createSlidingWindows(series, WINDOW_SIZE, x_data, y_data);
    
    int num_samples = y_data.size();
    
    // Split into train and test (80/20)
    int train_size = static_cast<int>(num_samples * 0.8);
    int test_size = num_samples - train_size;
    
    std::vector<float> x_train(x_data.begin(), x_data.begin() + train_size * WINDOW_SIZE);
    std::vector<float> y_train(y_data.begin(), y_data.begin() + train_size);
    std::vector<float> x_test(x_data.begin() + train_size * WINDOW_SIZE, x_data.end());
    std::vector<float> y_test(y_data.begin() + train_size, y_data.end());
    
    Tensor x_train_tensor = createTensor2D(x_train, train_size, WINDOW_SIZE);
    Tensor y_train_tensor = createTensor1D(y_train);
    Tensor x_test_tensor = createTensor2D(x_test, test_size, WINDOW_SIZE);
    Tensor y_test_tensor = createTensor1D(y_test);
    
    std::cout << "\nTraining samples: " << train_size << std::endl;
    std::cout << "Test samples: " << test_size << std::endl;
    
    std::cout << "\n=== Network Architecture ===" << std::endl;
    std::cout << "Input: " << WINDOW_SIZE << " (time window)" << std::endl;
    std::cout << "Hidden 1: " << HIDDEN1_SIZE << " (ReLU)" << std::endl;
    std::cout << "Hidden 2: " << HIDDEN2_SIZE << " (ReLU)" << std::endl;
    std::cout << "Output: " << OUTPUT_SIZE << " (linear)" << std::endl;
    
    // Create TensorFlow scope
    Scope root = Scope::NewRootScope();
    
    // Input placeholder
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, WINDOW_SIZE}));
    auto y_ph = Placeholder(root.WithOpName("Y"), DT_FLOAT,
                            Placeholder::Shape({-1, OUTPUT_SIZE}));
    
    // Hidden layer 1 weights and biases
    auto w1_var = Variable(root.WithOpName("W1"), 
                           {WINDOW_SIZE, HIDDEN1_SIZE}, DT_FLOAT);
    auto b1_var = Variable(root.WithOpName("b1"), 
                           {1, HIDDEN1_SIZE}, DT_FLOAT);
    
    // Hidden layer 2 weights and biases
    auto w2_var = Variable(root.WithOpName("W2"), 
                           {HIDDEN1_SIZE, HIDDEN2_SIZE}, DT_FLOAT);
    auto b2_var = Variable(root.WithOpName("b2"), 
                           {1, HIDDEN2_SIZE}, DT_FLOAT);
    
    // Output layer weights and biases
    auto w3_var = Variable(root.WithOpName("W3"), 
                           {HIDDEN2_SIZE, OUTPUT_SIZE}, DT_FLOAT);
    auto b3_var = Variable(root.WithOpName("b3"), 
                           {1, OUTPUT_SIZE}, DT_FLOAT);
    
    // Initialize weights with Xavier initialization
    float w1_scale = std::sqrt(2.0f / WINDOW_SIZE);
    float w2_scale = std::sqrt(2.0f / HIDDEN1_SIZE);
    float w3_scale = std::sqrt(2.0f / HIDDEN2_SIZE);
    
    auto w1_init = Assign(root, w1_var,
        Mul(root, RandomNormal(root, {WINDOW_SIZE, HIDDEN1_SIZE}, DT_FLOAT),
            Const(root, w1_scale)));
    auto b1_init = Assign(root, b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, HIDDEN1_SIZE}))));
    auto w2_init = Assign(root, w2_var,
        Mul(root, RandomNormal(root, {HIDDEN1_SIZE, HIDDEN2_SIZE}, DT_FLOAT),
            Const(root, w2_scale)));
    auto b2_init = Assign(root, b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, HIDDEN2_SIZE}))));
    auto w3_init = Assign(root, w3_var,
        Mul(root, RandomNormal(root, {HIDDEN2_SIZE, OUTPUT_SIZE}, DT_FLOAT),
            Const(root, w3_scale)));
    auto b3_init = Assign(root, b3_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, OUTPUT_SIZE}))));
    
    // Forward propagation
    auto z1 = Add(root, MatMul(root, x_ph, w1_var), b1_var);
    auto h1 = Relu(root.WithOpName("hidden1"), z1);
    
    auto z2 = Add(root, MatMul(root, h1, w2_var), b2_var);
    auto h2 = Relu(root.WithOpName("hidden2"), z2);
    
    auto z3 = Add(root, MatMul(root, h2, w3_var), b3_var);
    auto y_pred = Identity(root.WithOpName("output"), z3);
    
    // Loss: Mean Squared Error
    auto error = Sub(root, y_pred, y_ph);
    auto squared_error = Square(root, error);
    auto loss = Mean(root.WithOpName("loss"), squared_error, {0, 1});
    
    // Backpropagation - Manual gradient computation
    auto d_output = Mul(root, Const(root, 2.0f / train_size), error);
    
    // Output layer gradients
    auto d_w3 = MatMul(root, h2, d_output, MatMul::TransposeA(true));
    auto d_b3 = Mean(root, d_output, {0});
    
    // Hidden layer 2 gradients
    auto d_h2_pre = MatMul(root, d_output, w3_var, MatMul::TransposeB(true));
    auto relu_mask_h2 = Cast(root, Greater(root, z2, Const(root, 0.0f)), DT_FLOAT);
    auto d_h2 = Mul(root, d_h2_pre, relu_mask_h2);
    auto d_w2 = MatMul(root, h1, d_h2, MatMul::TransposeA(true));
    auto d_b2 = Mean(root, d_h2, {0});
    
    // Hidden layer 1 gradients
    auto d_h1_pre = MatMul(root, d_h2, w2_var, MatMul::TransposeB(true));
    auto relu_mask_h1 = Cast(root, Greater(root, z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_h1 = Mul(root, d_h1_pre, relu_mask_h1);
    auto d_w1 = MatMul(root, x_ph, d_h1, MatMul::TransposeA(true));
    auto d_b1 = Mean(root, d_h1, {0});
    
    // Gradient descent updates
    auto lr = Const(root, LEARNING_RATE);
    
    auto w1_update = AssignSub(root, w1_var, Mul(root, lr, d_w1));
    auto b1_update = AssignSub(root, b1_var, 
        Mul(root, lr, Reshape(root, d_b1, {1, HIDDEN1_SIZE})));
    auto w2_update = AssignSub(root, w2_var, Mul(root, lr, d_w2));
    auto b2_update = AssignSub(root, b2_var,
        Mul(root, lr, Reshape(root, d_b2, {1, HIDDEN2_SIZE})));
    auto w3_update = AssignSub(root, w3_var, Mul(root, lr, d_w3));
    auto b3_update = AssignSub(root, b3_var,
        Mul(root, lr, Reshape(root, d_b3, {1, OUTPUT_SIZE})));
    
    // Create session
    ClientSession session(root);
    
    // Initialize all variables
    TF_CHECK_OK(session.Run({w1_init, b1_init, w2_init, b2_init, w3_init, b3_init}, nullptr));
    
    std::cout << "\n=== Training Model ===" << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Epochs: " << NUM_EPOCHS << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Training loop
    std::vector<Tensor> outputs;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        TF_CHECK_OK(session.Run(
            {{x_ph, x_train_tensor}, {y_ph, y_train_tensor}},
            {loss, w1_update, b1_update, w2_update, b2_update, w3_update, b3_update},
            &outputs));
        
        if (epoch % 200 == 0 || epoch == NUM_EPOCHS - 1) {
            float current_loss = outputs[0].scalar<float>()();
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " - Loss: " << current_loss << std::endl;
        }
    }
    
    // Evaluate on test data
    std::cout << "\n=== Prediction Metrics ===" << std::endl;
    
    std::vector<Tensor> test_outputs;
    TF_CHECK_OK(session.Run(
        {{x_ph, x_test_tensor}},
        {y_pred},
        &test_outputs));
    
    float mae = calculateMAE(test_outputs[0], y_test_tensor);
    float rmse = calculateRMSE(test_outputs[0], y_test_tensor);
    
    std::cout << "MAE: " << mae << std::endl;
    std::cout << "RMSE: " << rmse << std::endl;
    
    // Show sample predictions
    std::cout << "\n=== Sample Predictions ===" << std::endl;
    std::cout << std::setw(8) << "t" << " | "
              << std::setw(12) << "Predicted" << " | "
              << std::setw(12) << "Actual" << " | "
              << std::setw(12) << "Error" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    auto pred_data = test_outputs[0].matrix<float>();
    auto target_data = y_test_tensor.matrix<float>();
    
    // Show predictions at specific time points
    std::vector<int> sample_indices = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    
    for (int idx : sample_indices) {
        if (idx < test_size) {
            int t = train_size + WINDOW_SIZE + idx;
            float predicted = pred_data(idx, 0);
            float actual = target_data(idx, 0);
            float error_val = std::abs(predicted - actual);
            
            std::cout << std::setw(8) << t << " | "
                      << std::setw(12) << predicted << " | "
                      << std::setw(12) << actual << " | "
                      << std::setw(12) << error_val << std::endl;
        }
    }
    
    // Evaluate on training data for comparison
    std::vector<Tensor> train_outputs;
    TF_CHECK_OK(session.Run(
        {{x_ph, x_train_tensor}},
        {y_pred},
        &train_outputs));
    
    float train_mae = calculateMAE(train_outputs[0], y_train_tensor);
    float train_rmse = calculateRMSE(train_outputs[0], y_train_tensor);
    
    std::cout << "\n=== Train vs Test Comparison ===" << std::endl;
    std::cout << "Training MAE:  " << train_mae << std::endl;
    std::cout << "Training RMSE: " << train_rmse << std::endl;
    std::cout << "Test MAE:      " << mae << std::endl;
    std::cout << "Test RMSE:     " << rmse << std::endl;
    
    // Show time series snippet
    std::cout << "\n=== Time Series Snippet ===" << std::endl;
    std::cout << "(First 20 values)" << std::endl;
    std::cout << std::setprecision(3);
    for (int i = 0; i < std::min(20, SERIES_LENGTH); ++i) {
        std::cout << "t=" << std::setw(3) << i << ": " << std::setw(7) << series[i];
        if ((i + 1) % 5 == 0) std::cout << std::endl;
        else std::cout << " | ";
    }
    
    std::cout << "\n\n==========================================" << std::endl;
    std::cout << "  Time series prediction completed!      " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
