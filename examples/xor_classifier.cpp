/**
 * @file xor_classifier.cpp
 * @brief Example 5: XOR Classifier using Neural Network with TensorFlow C++
 * 
 * This example demonstrates how to build and train a neural network
 * to solve the classic XOR problem using TensorFlow's C++ API.
 * 
 * The XOR problem is a non-linearly separable classification task that
 * requires at least one hidden layer to solve. This example implements
 * a multi-layer network with architecture: 2 -> 8 -> 4 -> 1
 * 
 * Key concepts demonstrated:
 * - Multi-layer neural network architecture
 * - Activation functions (ReLU for hidden layers, Sigmoid for output)
 * - The XOR problem and why it needs hidden layers
 * - Binary Cross-Entropy Loss
 * - Forward and backward propagation
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
 * @brief Creates a tensor from XOR input data
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
 * @brief Creates a tensor from label data
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
    std::cout << "  TensorFlow C++ XOR Classifier Demo     " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // XOR truth table
    // Input1 | Input2 | Output
    //   0    |   0    |   0
    //   0    |   1    |   1
    //   1    |   0    |   1
    //   1    |   1    |   0
    
    std::cout << "\n=== XOR Truth Table ===" << std::endl;
    std::cout << "Input1 | Input2 | Output" << std::endl;
    std::cout << "   0   |   0    |   0" << std::endl;
    std::cout << "   0   |   1    |   1" << std::endl;
    std::cout << "   1   |   0    |   1" << std::endl;
    std::cout << "   1   |   1    |   0" << std::endl;
    
    // XOR data
    std::vector<float> x_data = {
        0.0f, 0.0f,  // -> 0
        0.0f, 1.0f,  // -> 1
        1.0f, 0.0f,  // -> 1
        1.0f, 1.0f   // -> 0
    };
    
    std::vector<float> y_data = {0.0f, 1.0f, 1.0f, 0.0f};
    
    // Hyperparameters
    const int NUM_SAMPLES = 4;
    const int NUM_EPOCHS = 5000;
    const float LEARNING_RATE = 0.5f;
    const int INPUT_SIZE = 2;
    const int HIDDEN1_SIZE = 8;   // First hidden layer
    const int HIDDEN2_SIZE = 4;   // Second hidden layer
    const int OUTPUT_SIZE = 1;
    
    std::cout << "\n=== Network Architecture ===" << std::endl;
    std::cout << "Input layer:    " << INPUT_SIZE << " neurons" << std::endl;
    std::cout << "Hidden layer 1: " << HIDDEN1_SIZE << " neurons (ReLU activation)" << std::endl;
    std::cout << "Hidden layer 2: " << HIDDEN2_SIZE << " neurons (ReLU activation)" << std::endl;
    std::cout << "Output layer:   " << OUTPUT_SIZE << " neuron (Sigmoid activation)" << std::endl;
    
    // Create tensors
    Tensor x_tensor = createFeatureTensor(x_data, NUM_SAMPLES, INPUT_SIZE);
    Tensor y_tensor = createLabelTensor(y_data);
    
    // Create TensorFlow scope
    Scope root = Scope::NewRootScope();
    
    // Input placeholders
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, INPUT_SIZE}));
    auto y_ph = Placeholder(root.WithOpName("Y"), DT_FLOAT,
                            Placeholder::Shape({-1, OUTPUT_SIZE}));
    
    // Layer 1: Input -> Hidden1 (2 -> 8)
    auto w1_var = Variable(root.WithOpName("W1"), 
                           {INPUT_SIZE, HIDDEN1_SIZE}, DT_FLOAT);
    auto b1_var = Variable(root.WithOpName("b1"), 
                           {1, HIDDEN1_SIZE}, DT_FLOAT);
    
    // Layer 2: Hidden1 -> Hidden2 (8 -> 4)
    auto w2_var = Variable(root.WithOpName("W2"), 
                           {HIDDEN1_SIZE, HIDDEN2_SIZE}, DT_FLOAT);
    auto b2_var = Variable(root.WithOpName("b2"), 
                           {1, HIDDEN2_SIZE}, DT_FLOAT);
    
    // Layer 3: Hidden2 -> Output (4 -> 1)
    auto w3_var = Variable(root.WithOpName("W3"), 
                           {HIDDEN2_SIZE, OUTPUT_SIZE}, DT_FLOAT);
    auto b3_var = Variable(root.WithOpName("b3"), 
                           {1, OUTPUT_SIZE}, DT_FLOAT);
    
    // Initialize weights with He initialization for ReLU
    float w1_scale = std::sqrt(2.0f / INPUT_SIZE);
    float w2_scale = std::sqrt(2.0f / HIDDEN1_SIZE);
    float w3_scale = std::sqrt(2.0f / HIDDEN2_SIZE);
    
    auto w1_init = Assign(root, w1_var,
        Mul(root, RandomNormal(root, {INPUT_SIZE, HIDDEN1_SIZE}, DT_FLOAT),
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
    // Layer 1: h1 = ReLU(X * W1 + b1)
    auto z1 = Add(root, MatMul(root, x_ph, w1_var), b1_var);
    auto h1 = Relu(root.WithOpName("hidden1"), z1);
    
    // Layer 2: h2 = ReLU(h1 * W2 + b2)
    auto z2 = Add(root, MatMul(root, h1, w2_var), b2_var);
    auto h2 = Relu(root.WithOpName("hidden2"), z2);
    
    // Output layer: y_pred = Sigmoid(h2 * W3 + b3)
    auto z3 = Add(root, MatMul(root, h2, w3_var), b3_var);
    auto y_pred = Sigmoid(root.WithOpName("output"), z3);
    
    // Loss: Binary Cross-Entropy
    // loss = -mean(y * log(y_pred + epsilon) + (1-y) * log(1 - y_pred + epsilon))
    auto epsilon = Const(root, 1e-7f);
    auto one = Const(root, 1.0f);
    
    auto term1 = Mul(root, y_ph, Log(root, Add(root, y_pred, epsilon)));
    auto term2 = Mul(root, Sub(root, one, y_ph), 
                     Log(root, Add(root, Sub(root, one, y_pred), epsilon)));
    auto loss = Neg(root, Mean(root.WithOpName("loss"), 
                               Add(root, term1, term2), {0, 1}));
    
    // Backpropagation
    // Output layer gradients
    auto d_output = Sub(root, y_pred, y_ph);  // d(loss)/d(z3)
    auto d_w3 = MatMul(root, h2, d_output, MatMul::TransposeA(true));
    auto d_b3 = Mean(root, d_output, {0});
    
    // Hidden layer 2 gradients (ReLU derivative: 1 if z > 0, else 0)
    auto relu_grad2 = Cast(root, Greater(root, z2, Const(root, 0.0f)), DT_FLOAT);
    auto d_hidden2 = Mul(root,
        MatMul(root, d_output, w3_var, MatMul::TransposeB(true)),
        relu_grad2);
    auto d_w2 = MatMul(root, h1, d_hidden2, MatMul::TransposeA(true));
    auto d_b2 = Mean(root, d_hidden2, {0});
    
    // Hidden layer 1 gradients
    auto relu_grad1 = Cast(root, Greater(root, z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_hidden1 = Mul(root,
        MatMul(root, d_hidden2, w2_var, MatMul::TransposeB(true)),
        relu_grad1);
    auto d_w1 = MatMul(root, x_ph, d_hidden1, MatMul::TransposeA(true));
    auto d_b1 = Mean(root, d_hidden1, {0});
    
    // Gradient descent updates
    auto lr = Const(root, LEARNING_RATE);
    auto lr_scaled = Const(root, LEARNING_RATE / static_cast<float>(NUM_SAMPLES));
    
    auto w3_update = AssignSub(root, w3_var, Mul(root, lr_scaled, d_w3));
    auto b3_update = AssignSub(root, b3_var, 
        Mul(root, lr, Reshape(root, d_b3, {1, OUTPUT_SIZE})));
    auto w2_update = AssignSub(root, w2_var, Mul(root, lr_scaled, d_w2));
    auto b2_update = AssignSub(root, b2_var,
        Mul(root, lr, Reshape(root, d_b2, {1, HIDDEN2_SIZE})));
    auto w1_update = AssignSub(root, w1_var, Mul(root, lr_scaled, d_w1));
    auto b1_update = AssignSub(root, b1_var,
        Mul(root, lr, Reshape(root, d_b1, {1, HIDDEN1_SIZE})));
    
    // Create session
    ClientSession session(root);
    
    // Initialize all variables
    TF_CHECK_OK(session.Run({w1_init, b1_init, w2_init, b2_init, w3_init, b3_init}, nullptr));
    
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
            {loss, y_pred, w1_update, b1_update, w2_update, b2_update, w3_update, b3_update},
            &outputs));
        
        // Print progress
        if (epoch % 500 == 0 || epoch == NUM_EPOCHS - 1) {
            float current_loss = outputs[0].scalar<float>()();
            float accuracy = computeAccuracy(outputs[1], y_tensor);
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " | Loss: " << std::setw(8) << current_loss
                      << " | Accuracy: " << std::setw(6) << accuracy << "%" << std::endl;
        }
    }
    
    // Final evaluation
    std::cout << "\n=== Final Results ===" << std::endl;
    
    TF_CHECK_OK(session.Run(
        {{x_ph, x_tensor}, {y_ph, y_tensor}},
        {loss, y_pred},
        &outputs));
    
    float final_loss = outputs[0].scalar<float>()();
    float final_accuracy = computeAccuracy(outputs[1], y_tensor);
    
    std::cout << "Final Loss: " << final_loss << std::endl;
    std::cout << "Final Accuracy: " << final_accuracy << "%" << std::endl;
    
    // Show predictions for XOR truth table
    std::cout << "\n=== XOR Predictions ===" << std::endl;
    std::cout << "Input1 | Input2 | Expected | Predicted | Probability" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    auto pred_data = outputs[1].matrix<float>();
    auto x_mat = x_tensor.matrix<float>();
    auto y_mat = y_tensor.matrix<float>();
    
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float pred_class = (pred_data(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        std::cout << std::setw(6) << static_cast<int>(x_mat(i, 0)) << " | "
                  << std::setw(6) << static_cast<int>(x_mat(i, 1)) << " | "
                  << std::setw(8) << static_cast<int>(y_mat(i, 0)) << " | "
                  << std::setw(9) << static_cast<int>(pred_class) << " | "
                  << std::setw(11) << pred_data(i, 0) << std::endl;
    }
    
    // Verify all predictions are correct
    bool all_correct = true;
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        float pred_class = (pred_data(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        if (pred_class != y_mat(i, 0)) {
            all_correct = false;
            break;
        }
    }
    
    std::cout << "\n=== Summary ===" << std::endl;
    if (all_correct) {
        std::cout << "SUCCESS: Network learned XOR function perfectly!" << std::endl;
    } else {
        std::cout << "Network still learning... try more epochs or adjust hyperparameters." << std::endl;
    }
    
    std::cout << "\nThis demonstrates why XOR requires hidden layers:" << std::endl;
    std::cout << "- XOR is not linearly separable" << std::endl;
    std::cout << "- A single perceptron cannot solve XOR" << std::endl;
    std::cout << "- Hidden layers allow learning non-linear decision boundaries" << std::endl;
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  XOR classifier demo completed!         " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
