/**
 * @file neural_network.cpp
 * @brief Example 3: Neural Network Classification using TensorFlow C++
 * 
 * This example demonstrates how to build and train a simple neural network
 * for binary classification using TensorFlow's C++ API.
 * 
 * The example creates a 2-layer neural network to classify synthetic data
 * into two classes based on an XOR-like decision boundary.
 * 
 * Key concepts demonstrated:
 * - Multi-layer neural network architecture
 * - Activation functions (Sigmoid)
 * - Forward propagation
 * - Loss computation (Binary Cross-Entropy)
 * - Backpropagation with gradient descent
 * - Classification accuracy metrics
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
 * @brief Generates synthetic classification data (XOR-like pattern)
 * 
 * Creates a dataset where class 1 corresponds to points where x1*x2 > 0
 * (first and third quadrants), and class 0 corresponds to points where
 * x1*x2 < 0 (second and fourth quadrants). This tests the network's 
 * ability to learn non-linear decision boundaries.
 * 
 * @param num_samples Total number of samples
 * @param x_data Output feature matrix (num_samples x 2)
 * @param y_data Output labels (0 or 1)
 */
void generateClassificationData(int num_samples, 
                                std::vector<float>& x_data,
                                std::vector<float>& y_data) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> uniform(-2.0f, 2.0f);
    std::normal_distribution<float> noise(0.0f, 0.1f);
    
    x_data.clear();
    y_data.clear();
    
    for (int i = 0; i < num_samples; ++i) {
        float x1 = uniform(gen);
        float x2 = uniform(gen);
        
        // XOR-like pattern: label = 1 if (x1*x2 > 0), else 0
        float label = (x1 * x2 > 0) ? 1.0f : 0.0f;
        
        // Add some noise to make it challenging
        x1 += noise(gen);
        x2 += noise(gen);
        
        x_data.push_back(x1);
        x_data.push_back(x2);
        y_data.push_back(label);
    }
}

/**
 * @brief Creates a 2D tensor from feature data
 * @param data Flat vector of features
 * @param num_samples Number of samples
 * @param num_features Number of features per sample
 * @return TensorFlow tensor
 */
Tensor createFeatureTensor(const std::vector<float>& data, 
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
 * @brief Creates a 1D tensor from label data
 * @param data Vector of labels
 * @return TensorFlow tensor
 */
Tensor createLabelTensor(const std::vector<float>& data) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(data.size()), 1}));
    auto tensor_map = tensor.matrix<float>();
    for (size_t i = 0; i < data.size(); ++i) {
        tensor_map(i, 0) = data[i];
    }
    return tensor;
}

/**
 * @brief Computes classification accuracy
 * @param predictions Predicted probabilities
 * @param labels True labels
 * @return Accuracy as a percentage
 */
float computeAccuracy(const Tensor& predictions, const Tensor& labels) {
    auto pred_data = predictions.matrix<float>();
    auto label_data = labels.matrix<float>();
    
    int correct = 0;
    int total = predictions.dim_size(0);
    
    for (int i = 0; i < total; ++i) {
        float pred_class = (pred_data(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        if (pred_class == label_data(i, 0)) {
            correct++;
        }
    }
    
    return 100.0f * static_cast<float>(correct) / static_cast<float>(total);
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  TensorFlow C++ Neural Network Demo     " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Hyperparameters
    const int NUM_SAMPLES = 200;
    const int NUM_EPOCHS = 2000;
    const float LEARNING_RATE = 0.5f;
    const int HIDDEN_SIZE = 8;  // Number of neurons in hidden layer
    const int INPUT_SIZE = 2;   // Number of input features
    const int OUTPUT_SIZE = 1;  // Binary classification
    
    std::cout << "\n=== Network Architecture ===" << std::endl;
    std::cout << "Input layer:  " << INPUT_SIZE << " neurons" << std::endl;
    std::cout << "Hidden layer: " << HIDDEN_SIZE << " neurons (Sigmoid activation)" << std::endl;
    std::cout << "Output layer: " << OUTPUT_SIZE << " neuron (Sigmoid activation)" << std::endl;
    
    // Generate training data
    std::vector<float> x_data, y_data;
    generateClassificationData(NUM_SAMPLES, x_data, y_data);
    
    Tensor x_tensor = createFeatureTensor(x_data, NUM_SAMPLES, INPUT_SIZE);
    Tensor y_tensor = createLabelTensor(y_data);
    
    // Count class distribution
    int class_0 = std::count(y_data.begin(), y_data.end(), 0.0f);
    int class_1 = std::count(y_data.begin(), y_data.end(), 1.0f);
    
    std::cout << "\n=== Dataset ===" << std::endl;
    std::cout << "Total samples: " << NUM_SAMPLES << std::endl;
    std::cout << "Class 0: " << class_0 << " samples" << std::endl;
    std::cout << "Class 1: " << class_1 << " samples" << std::endl;
    
    // Create TensorFlow scope
    Scope root = Scope::NewRootScope();
    
    // Input placeholders
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, INPUT_SIZE}));
    auto y_ph = Placeholder(root.WithOpName("Y"), DT_FLOAT,
                            Placeholder::Shape({-1, OUTPUT_SIZE}));
    
    // Initialize weights and biases using Xavier initialization
    // Hidden layer weights and biases
    auto w1_var = Variable(root.WithOpName("W1"), 
                           {INPUT_SIZE, HIDDEN_SIZE}, DT_FLOAT);
    auto b1_var = Variable(root.WithOpName("b1"), 
                           {1, HIDDEN_SIZE}, DT_FLOAT);
    
    // Output layer weights and biases
    auto w2_var = Variable(root.WithOpName("W2"), 
                           {HIDDEN_SIZE, OUTPUT_SIZE}, DT_FLOAT);
    auto b2_var = Variable(root.WithOpName("b2"), 
                           {1, OUTPUT_SIZE}, DT_FLOAT);
    
    // Initialize variables
    float w1_scale = std::sqrt(2.0f / INPUT_SIZE);
    float w2_scale = std::sqrt(2.0f / HIDDEN_SIZE);
    
    auto w1_init = Assign(root, w1_var,
        Mul(root, RandomNormal(root, {INPUT_SIZE, HIDDEN_SIZE}, DT_FLOAT),
            Const(root, w1_scale)));
    auto b1_init = Assign(root, b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, HIDDEN_SIZE}))));
    auto w2_init = Assign(root, w2_var,
        Mul(root, RandomNormal(root, {HIDDEN_SIZE, OUTPUT_SIZE}, DT_FLOAT),
            Const(root, w2_scale)));
    auto b2_init = Assign(root, b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, OUTPUT_SIZE}))));
    
    // Forward propagation
    // Hidden layer: h = sigmoid(X * W1 + b1)
    auto z1 = Add(root, MatMul(root, x_ph, w1_var), b1_var);
    auto h = Sigmoid(root.WithOpName("hidden"), z1);
    
    // Output layer: y_pred = sigmoid(h * W2 + b2)
    auto z2 = Add(root, MatMul(root, h, w2_var), b2_var);
    auto y_pred = Sigmoid(root.WithOpName("output"), z2);
    
    // Loss: Binary Cross-Entropy
    // loss = -mean(y * log(y_pred + epsilon) + (1-y) * log(1 - y_pred + epsilon))
    auto epsilon = Const(root, 1e-7f);
    auto one = Const(root, 1.0f);
    
    auto term1 = Mul(root, y_ph, Log(root, Add(root, y_pred, epsilon)));
    auto term2 = Mul(root, Sub(root, one, y_ph), 
                     Log(root, Add(root, Sub(root, one, y_pred), epsilon)));
    auto loss = Neg(root, Mean(root.WithOpName("loss"), 
                               Add(root, term1, term2), {0, 1}));
    
    // Backpropagation (manual gradient computation)
    // Output layer gradients
    auto d_output = Sub(root, y_pred, y_ph); // d(loss)/d(z2)
    auto d_w2 = MatMul(root, h, d_output, MatMul::TransposeA(true));
    auto d_b2 = Mean(root, d_output, {0});
    
    // Hidden layer gradients
    auto d_hidden = Mul(root,
        MatMul(root, d_output, w2_var, MatMul::TransposeB(true)),
        Mul(root, h, Sub(root, one, h))); // sigmoid derivative: h * (1-h)
    auto d_w1 = MatMul(root, x_ph, d_hidden, MatMul::TransposeA(true));
    auto d_b1 = Mean(root, d_hidden, {0});
    
    // Gradient descent updates
    auto lr = Const(root, LEARNING_RATE);
    auto lr_scaled = Const(root, LEARNING_RATE / static_cast<float>(NUM_SAMPLES));
    
    auto w2_update = AssignSub(root, w2_var, Mul(root, lr_scaled, d_w2));
    auto b2_update = AssignSub(root, b2_var, 
        Mul(root, lr, Reshape(root, d_b2, {1, OUTPUT_SIZE})));
    auto w1_update = AssignSub(root, w1_var, Mul(root, lr_scaled, d_w1));
    auto b1_update = AssignSub(root, b1_var,
        Mul(root, lr, Reshape(root, d_b1, {1, HIDDEN_SIZE})));
    
    // Create session
    ClientSession session(root);
    
    // Initialize all variables
    TF_CHECK_OK(session.Run({w1_init, b1_init, w2_init, b2_init}, nullptr));
    
    std::cout << "\n=== Training ===" << std::endl;
    std::cout << "Learning rate: " << LEARNING_RATE << std::endl;
    std::cout << "Epochs: " << NUM_EPOCHS << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Training loop
    std::vector<Tensor> outputs;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        // Forward pass and updates
        TF_CHECK_OK(session.Run(
            {{x_ph, x_tensor}, {y_ph, y_tensor}},
            {loss, y_pred, w1_update, b1_update, w2_update, b2_update},
            &outputs));
        
        // Print progress
        if (epoch % 200 == 0 || epoch == NUM_EPOCHS - 1) {
            float current_loss = outputs[0].scalar<float>()();
            float accuracy = computeAccuracy(outputs[1], y_tensor);
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << std::setw(8) << current_loss
                      << " | Accuracy: " << std::setw(6) << accuracy << "%" << std::endl;
        }
    }
    
    // Final evaluation
    std::cout << "\n=== Final Evaluation ===" << std::endl;
    
    TF_CHECK_OK(session.Run(
        {{x_ph, x_tensor}, {y_ph, y_tensor}},
        {loss, y_pred},
        &outputs));
    
    float final_loss = outputs[0].scalar<float>()();
    float final_accuracy = computeAccuracy(outputs[1], y_tensor);
    
    std::cout << "Final Loss: " << final_loss << std::endl;
    std::cout << "Final Accuracy: " << final_accuracy << "%" << std::endl;
    
    // Show some predictions
    std::cout << "\n=== Sample Predictions ===" << std::endl;
    std::cout << std::setw(8) << "X1" << " | "
              << std::setw(8) << "X2" << " | "
              << std::setw(8) << "True" << " | "
              << std::setw(8) << "Pred" << " | "
              << std::setw(8) << "Prob" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    auto pred_data = outputs[1].matrix<float>();
    auto x_mat = x_tensor.matrix<float>();
    auto y_mat = y_tensor.matrix<float>();
    
    // Show first 10 predictions
    for (int i = 0; i < std::min(10, NUM_SAMPLES); ++i) {
        float pred_class = (pred_data(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        std::cout << std::setw(8) << x_mat(i, 0) << " | "
                  << std::setw(8) << x_mat(i, 1) << " | "
                  << std::setw(8) << y_mat(i, 0) << " | "
                  << std::setw(8) << pred_class << " | "
                  << std::setw(8) << pred_data(i, 0) << std::endl;
    }
    
    // Test on new data points
    std::cout << "\n=== Predictions on New Data ===" << std::endl;
    std::vector<float> test_x = {
        1.0f, 1.0f,    // Should be class 1 (+ * + > 0)
        -1.0f, -1.0f,  // Should be class 1 (- * - > 0)
        1.0f, -1.0f,   // Should be class 0 (+ * - < 0)
        -1.0f, 1.0f,   // Should be class 0 (- * + < 0)
        0.5f, 0.5f,    // Should be class 1
        -0.5f, 0.5f    // Should be class 0
    };
    
    Tensor test_tensor = createFeatureTensor(test_x, 6, 2);
    
    std::vector<Tensor> test_outputs;
    TF_CHECK_OK(session.Run(
        {{x_ph, test_tensor}},
        {y_pred},
        &test_outputs));
    
    auto test_pred = test_outputs[0].matrix<float>();
    auto test_mat = test_tensor.matrix<float>();
    
    std::cout << std::setw(8) << "X1" << " | "
              << std::setw(8) << "X2" << " | "
              << std::setw(10) << "Expected" << " | "
              << std::setw(8) << "Pred" << " | "
              << std::setw(8) << "Prob" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<float> expected = {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    for (int i = 0; i < 6; ++i) {
        float pred_class = (test_pred(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        std::cout << std::setw(8) << test_mat(i, 0) << " | "
                  << std::setw(8) << test_mat(i, 1) << " | "
                  << std::setw(10) << expected[i] << " | "
                  << std::setw(8) << pred_class << " | "
                  << std::setw(8) << test_pred(i, 0) << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  Neural network demo completed!         " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
