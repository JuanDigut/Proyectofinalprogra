/**
 * @file anomaly_detection.cpp
 * @brief Example 6: Anomaly Detection with Autoencoder using TensorFlow C++
 * 
 * This example demonstrates how to implement anomaly detection using an
 * Autoencoder neural network with TensorFlow's C++ API.
 * 
 * An Autoencoder learns to compress and reconstruct normal data. When presented
 * with anomalous data, the reconstruction error will be higher, allowing us
 * to detect anomalies based on a threshold.
 * 
 * Key concepts demonstrated:
 * - Autoencoder architecture (encoder-decoder)
 * - Unsupervised learning for anomaly detection
 * - Reconstruction error as anomaly score
 * - Threshold-based classification
 * - Detection metrics (precision, recall, accuracy)
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
 * @brief Generates synthetic normal data (2D Gaussian distribution)
 * 
 * @param num_samples Number of normal samples to generate
 * @param data Output vector for data points (flattened)
 * @param mean_x Mean of x coordinate
 * @param mean_y Mean of y coordinate
 * @param std_dev Standard deviation
 */
void generateNormalData(int num_samples, std::vector<float>& data,
                        float mean_x, float mean_y, float std_dev) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist_x(mean_x, std_dev);
    std::normal_distribution<float> dist_y(mean_y, std_dev);
    
    for (int i = 0; i < num_samples; ++i) {
        data.push_back(dist_x(gen));
        data.push_back(dist_y(gen));
    }
}

/**
 * @brief Generates anomaly data (points far from normal distribution)
 * 
 * @param num_samples Number of anomaly samples to generate
 * @param data Output vector for data points (flattened)
 */
void generateAnomalies(int num_samples, std::vector<float>& data) {
    std::random_device rd;
    std::mt19937 gen(123); // Different seed for anomalies
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (int i = 0; i < num_samples; ++i) {
        // Generate points far from the normal distribution center (0, 0)
        float x = dist(gen);
        float y = dist(gen);
        
        // Ensure anomalies are outside the normal region
        if (std::abs(x) < 2.0f) x += (x >= 0) ? 2.5f : -2.5f;
        if (std::abs(y) < 2.0f) y += (y >= 0) ? 2.5f : -2.5f;
        
        data.push_back(x);
        data.push_back(y);
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
 * @brief Calculates reconstruction error for each sample
 * @param original Original data tensor
 * @param reconstructed Reconstructed data tensor
 * @return Vector of reconstruction errors (MSE per sample)
 */
std::vector<float> calculateReconstructionError(const Tensor& original, 
                                                 const Tensor& reconstructed) {
    auto orig_data = original.matrix<float>();
    auto recon_data = reconstructed.matrix<float>();
    int num_samples = original.dim_size(0);
    int num_features = original.dim_size(1);
    
    std::vector<float> errors;
    for (int i = 0; i < num_samples; ++i) {
        float mse = 0.0f;
        for (int j = 0; j < num_features; ++j) {
            float diff = orig_data(i, j) - recon_data(i, j);
            mse += diff * diff;
        }
        errors.push_back(mse / num_features);
    }
    return errors;
}

/**
 * @brief Computes detection metrics
 * @param errors Reconstruction errors
 * @param labels True labels (0 = normal, 1 = anomaly)
 * @param threshold Anomaly detection threshold
 * @return Tuple of (true positives, false positives, true negatives, false negatives)
 */
void computeMetrics(const std::vector<float>& errors,
                    const std::vector<int>& labels,
                    float threshold,
                    int& tp, int& fp, int& tn, int& fn) {
    tp = fp = tn = fn = 0;
    
    for (size_t i = 0; i < errors.size(); ++i) {
        bool predicted_anomaly = errors[i] > threshold;
        bool is_anomaly = labels[i] == 1;
        
        if (predicted_anomaly && is_anomaly) tp++;
        else if (predicted_anomaly && !is_anomaly) fp++;
        else if (!predicted_anomaly && !is_anomaly) tn++;
        else fn++;
    }
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Anomaly Detection with Autoencoder     " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Hyperparameters
    const int NUM_NORMAL = 200;
    const int NUM_ANOMALIES = 20;
    const int NUM_EPOCHS = 500;
    const float LEARNING_RATE = 0.1f;
    
    // Autoencoder architecture: 2 -> 8 -> 4 -> 8 -> 2
    const int INPUT_SIZE = 2;
    const int ENCODER_H1 = 8;
    const int LATENT_SIZE = 4;
    const int DECODER_H1 = 8;
    const int OUTPUT_SIZE = 2;
    
    std::cout << "\n=== Generating Data ===" << std::endl;
    
    // Generate normal training data
    std::vector<float> normal_data;
    generateNormalData(NUM_NORMAL, normal_data, 0.0f, 0.0f, 1.0f);
    
    // Generate anomaly data
    std::vector<float> anomaly_data;
    generateAnomalies(NUM_ANOMALIES, anomaly_data);
    
    std::cout << "- Normal data: " << NUM_NORMAL << " points" << std::endl;
    std::cout << "- Anomalies: " << NUM_ANOMALIES << " points" << std::endl;
    
    // Create training tensor (only normal data for training)
    Tensor train_tensor = createTensor2D(normal_data, NUM_NORMAL, INPUT_SIZE);
    
    // Create test data (normal + anomalies)
    std::vector<float> test_data = normal_data;
    test_data.insert(test_data.end(), anomaly_data.begin(), anomaly_data.end());
    int total_test = NUM_NORMAL + NUM_ANOMALIES;
    Tensor test_tensor = createTensor2D(test_data, total_test, INPUT_SIZE);
    
    // Create labels (0 = normal, 1 = anomaly)
    std::vector<int> test_labels(NUM_NORMAL, 0);
    test_labels.insert(test_labels.end(), NUM_ANOMALIES, 1);
    
    std::cout << "\n=== Autoencoder Architecture ===" << std::endl;
    std::cout << "Encoder: " << INPUT_SIZE << " -> " << ENCODER_H1 << " -> " << LATENT_SIZE << std::endl;
    std::cout << "Decoder: " << LATENT_SIZE << " -> " << DECODER_H1 << " -> " << OUTPUT_SIZE << std::endl;
    
    // Create TensorFlow scope
    Scope root = Scope::NewRootScope();
    
    // Input placeholder
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, INPUT_SIZE}));
    
    // Encoder weights and biases
    auto enc_w1_var = Variable(root.WithOpName("enc_W1"), 
                               {INPUT_SIZE, ENCODER_H1}, DT_FLOAT);
    auto enc_b1_var = Variable(root.WithOpName("enc_b1"), 
                               {1, ENCODER_H1}, DT_FLOAT);
    auto enc_w2_var = Variable(root.WithOpName("enc_W2"), 
                               {ENCODER_H1, LATENT_SIZE}, DT_FLOAT);
    auto enc_b2_var = Variable(root.WithOpName("enc_b2"), 
                               {1, LATENT_SIZE}, DT_FLOAT);
    
    // Decoder weights and biases
    auto dec_w1_var = Variable(root.WithOpName("dec_W1"), 
                               {LATENT_SIZE, DECODER_H1}, DT_FLOAT);
    auto dec_b1_var = Variable(root.WithOpName("dec_b1"), 
                               {1, DECODER_H1}, DT_FLOAT);
    auto dec_w2_var = Variable(root.WithOpName("dec_W2"), 
                               {DECODER_H1, OUTPUT_SIZE}, DT_FLOAT);
    auto dec_b2_var = Variable(root.WithOpName("dec_b2"), 
                               {1, OUTPUT_SIZE}, DT_FLOAT);
    
    // Initialize weights with Xavier initialization
    float enc_w1_scale = std::sqrt(2.0f / INPUT_SIZE);
    float enc_w2_scale = std::sqrt(2.0f / ENCODER_H1);
    float dec_w1_scale = std::sqrt(2.0f / LATENT_SIZE);
    float dec_w2_scale = std::sqrt(2.0f / DECODER_H1);
    
    auto enc_w1_init = Assign(root, enc_w1_var,
        Mul(root, RandomNormal(root, {INPUT_SIZE, ENCODER_H1}, DT_FLOAT),
            Const(root, enc_w1_scale)));
    auto enc_b1_init = Assign(root, enc_b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, ENCODER_H1}))));
    auto enc_w2_init = Assign(root, enc_w2_var,
        Mul(root, RandomNormal(root, {ENCODER_H1, LATENT_SIZE}, DT_FLOAT),
            Const(root, enc_w2_scale)));
    auto enc_b2_init = Assign(root, enc_b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, LATENT_SIZE}))));
    
    auto dec_w1_init = Assign(root, dec_w1_var,
        Mul(root, RandomNormal(root, {LATENT_SIZE, DECODER_H1}, DT_FLOAT),
            Const(root, dec_w1_scale)));
    auto dec_b1_init = Assign(root, dec_b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, DECODER_H1}))));
    auto dec_w2_init = Assign(root, dec_w2_var,
        Mul(root, RandomNormal(root, {DECODER_H1, OUTPUT_SIZE}, DT_FLOAT),
            Const(root, dec_w2_scale)));
    auto dec_b2_init = Assign(root, dec_b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, OUTPUT_SIZE}))));
    
    // Forward propagation - Encoder
    auto enc_z1 = Add(root, MatMul(root, x_ph, enc_w1_var), enc_b1_var);
    auto enc_h1 = Relu(root.WithOpName("enc_hidden1"), enc_z1);
    
    auto enc_z2 = Add(root, MatMul(root, enc_h1, enc_w2_var), enc_b2_var);
    auto latent = Relu(root.WithOpName("latent"), enc_z2);
    
    // Forward propagation - Decoder
    auto dec_z1 = Add(root, MatMul(root, latent, dec_w1_var), dec_b1_var);
    auto dec_h1 = Relu(root.WithOpName("dec_hidden1"), dec_z1);
    
    auto dec_z2 = Add(root, MatMul(root, dec_h1, dec_w2_var), dec_b2_var);
    auto x_reconstructed = Identity(root.WithOpName("output"), dec_z2);
    
    // Loss: Mean Squared Error
    auto error = Sub(root, x_ph, x_reconstructed);
    auto squared_error = Square(root, error);
    auto loss = Mean(root.WithOpName("loss"), squared_error, {0, 1});
    
    // Backpropagation - Manual gradient computation
    auto one = Const(root, 1.0f);
    auto num_samples_f = Const(root, static_cast<float>(NUM_NORMAL));
    
    // Output layer gradients (decoder layer 2)
    auto d_output = Mul(root, Const(root, 2.0f / (NUM_NORMAL * OUTPUT_SIZE)), 
                        Sub(root, x_reconstructed, x_ph));
    auto d_dec_w2 = MatMul(root, dec_h1, d_output, MatMul::TransposeA(true));
    auto d_dec_b2 = Mean(root, d_output, {0});
    
    // Decoder hidden layer gradients (decoder layer 1)
    auto d_dec_h1_pre = MatMul(root, d_output, dec_w2_var, MatMul::TransposeB(true));
    auto relu_mask_dec = Cast(root, Greater(root, dec_z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_dec_h1 = Mul(root, d_dec_h1_pre, relu_mask_dec);
    auto d_dec_w1 = MatMul(root, latent, d_dec_h1, MatMul::TransposeA(true));
    auto d_dec_b1 = Mean(root, d_dec_h1, {0});
    
    // Latent layer gradients (encoder layer 2)
    auto d_latent_pre = MatMul(root, d_dec_h1, dec_w1_var, MatMul::TransposeB(true));
    auto relu_mask_latent = Cast(root, Greater(root, enc_z2, Const(root, 0.0f)), DT_FLOAT);
    auto d_latent = Mul(root, d_latent_pre, relu_mask_latent);
    auto d_enc_w2 = MatMul(root, enc_h1, d_latent, MatMul::TransposeA(true));
    auto d_enc_b2 = Mean(root, d_latent, {0});
    
    // Encoder hidden layer gradients (encoder layer 1)
    auto d_enc_h1_pre = MatMul(root, d_latent, enc_w2_var, MatMul::TransposeB(true));
    auto relu_mask_enc = Cast(root, Greater(root, enc_z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_enc_h1 = Mul(root, d_enc_h1_pre, relu_mask_enc);
    auto d_enc_w1 = MatMul(root, x_ph, d_enc_h1, MatMul::TransposeA(true));
    auto d_enc_b1 = Mean(root, d_enc_h1, {0});
    
    // Gradient descent updates
    auto lr = Const(root, LEARNING_RATE);
    
    auto enc_w1_update = AssignSub(root, enc_w1_var, Mul(root, lr, d_enc_w1));
    auto enc_b1_update = AssignSub(root, enc_b1_var, 
        Mul(root, lr, Reshape(root, d_enc_b1, {1, ENCODER_H1})));
    auto enc_w2_update = AssignSub(root, enc_w2_var, Mul(root, lr, d_enc_w2));
    auto enc_b2_update = AssignSub(root, enc_b2_var,
        Mul(root, lr, Reshape(root, d_enc_b2, {1, LATENT_SIZE})));
    
    auto dec_w1_update = AssignSub(root, dec_w1_var, Mul(root, lr, d_dec_w1));
    auto dec_b1_update = AssignSub(root, dec_b1_var,
        Mul(root, lr, Reshape(root, d_dec_b1, {1, DECODER_H1})));
    auto dec_w2_update = AssignSub(root, dec_w2_var, Mul(root, lr, d_dec_w2));
    auto dec_b2_update = AssignSub(root, dec_b2_var,
        Mul(root, lr, Reshape(root, d_dec_b2, {1, OUTPUT_SIZE})));
    
    // Create session
    ClientSession session(root);
    
    // Initialize all variables
    TF_CHECK_OK(session.Run({enc_w1_init, enc_b1_init, enc_w2_init, enc_b2_init,
                             dec_w1_init, dec_b1_init, dec_w2_init, dec_b2_init}, nullptr));
    
    std::cout << "\n=== Training Autoencoder ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Training loop
    std::vector<Tensor> outputs;
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        TF_CHECK_OK(session.Run(
            {{x_ph, train_tensor}},
            {loss, enc_w1_update, enc_b1_update, enc_w2_update, enc_b2_update,
             dec_w1_update, dec_b1_update, dec_w2_update, dec_b2_update},
            &outputs));
        
        if (epoch % 100 == 0 || epoch == NUM_EPOCHS - 1) {
            float current_loss = outputs[0].scalar<float>()();
            std::cout << "Epoch " << std::setw(4) << epoch
                      << " - Loss: " << current_loss << std::endl;
        }
    }
    
    // Evaluate on test data (normal + anomalies)
    std::cout << "\n=== Detection Results ===" << std::endl;
    
    std::vector<Tensor> test_outputs;
    TF_CHECK_OK(session.Run(
        {{x_ph, test_tensor}},
        {x_reconstructed},
        &test_outputs));
    
    // Calculate reconstruction errors
    std::vector<float> recon_errors = calculateReconstructionError(test_tensor, test_outputs[0]);
    
    // Calculate threshold based on normal data errors (mean + 2*std)
    float mean_error = 0.0f;
    for (int i = 0; i < NUM_NORMAL; ++i) {
        mean_error += recon_errors[i];
    }
    mean_error /= NUM_NORMAL;
    
    float std_error = 0.0f;
    for (int i = 0; i < NUM_NORMAL; ++i) {
        float diff = recon_errors[i] - mean_error;
        std_error += diff * diff;
    }
    std_error = std::sqrt(std_error / NUM_NORMAL);
    
    float threshold = mean_error + 2.0f * std_error;
    
    std::cout << "Anomaly threshold: " << threshold << std::endl;
    std::cout << "(based on mean + 2*std of reconstruction error)" << std::endl;
    
    // Compute detection metrics
    int tp, fp, tn, fn;
    computeMetrics(recon_errors, test_labels, threshold, tp, fp, tn, fn);
    
    int anomalies_detected = tp;
    int total_anomalies = NUM_ANOMALIES;
    
    std::cout << "\nAnomalies detected: " << anomalies_detected << "/" << total_anomalies << std::endl;
    std::cout << "False positives: " << fp << std::endl;
    std::cout << "False negatives: " << fn << std::endl;
    
    float precision = (tp + fp > 0) ? (100.0f * tp / (tp + fp)) : 0.0f;
    float recall = (tp + fn > 0) ? (100.0f * tp / (tp + fn)) : 0.0f;
    float accuracy = 100.0f * (tp + tn) / (tp + tn + fp + fn);
    float f1_score = (precision + recall > 0) ? (2.0f * precision * recall / (precision + recall)) : 0.0f;
    
    std::cout << "\n=== Detection Metrics ===" << std::endl;
    std::cout << "Precision: " << std::setprecision(1) << precision << "%" << std::endl;
    std::cout << "Recall: " << recall << "%" << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    std::cout << "F1-Score: " << std::setprecision(2) << f1_score << "%" << std::endl;
    
    // Show some example reconstructions
    std::cout << "\n=== Reconstruction Examples ===" << std::endl;
    std::cout << std::setprecision(3);
    std::cout << std::setw(10) << "Original X" << " | "
              << std::setw(10) << "Original Y" << " | "
              << std::setw(10) << "Recon X" << " | "
              << std::setw(10) << "Recon Y" << " | "
              << std::setw(10) << "Error" << " | "
              << std::setw(10) << "Type" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    auto test_data_mat = test_tensor.matrix<float>();
    auto recon_data_mat = test_outputs[0].matrix<float>();
    
    // Show first 5 normal and first 5 anomalies
    std::cout << "Normal data:" << std::endl;
    for (int i = 0; i < std::min(5, NUM_NORMAL); ++i) {
        std::cout << std::setw(10) << test_data_mat(i, 0) << " | "
                  << std::setw(10) << test_data_mat(i, 1) << " | "
                  << std::setw(10) << recon_data_mat(i, 0) << " | "
                  << std::setw(10) << recon_data_mat(i, 1) << " | "
                  << std::setw(10) << recon_errors[i] << " | "
                  << (recon_errors[i] > threshold ? "ANOMALY" : "Normal") << std::endl;
    }
    
    std::cout << "\nAnomalies:" << std::endl;
    for (int i = NUM_NORMAL; i < std::min(NUM_NORMAL + 5, total_test); ++i) {
        std::cout << std::setw(10) << test_data_mat(i, 0) << " | "
                  << std::setw(10) << test_data_mat(i, 1) << " | "
                  << std::setw(10) << recon_data_mat(i, 0) << " | "
                  << std::setw(10) << recon_data_mat(i, 1) << " | "
                  << std::setw(10) << recon_errors[i] << " | "
                  << (recon_errors[i] > threshold ? "ANOMALY" : "Normal") << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  Anomaly detection completed!           " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
