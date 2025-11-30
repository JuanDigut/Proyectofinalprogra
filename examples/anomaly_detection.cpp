/**
 * @file anomaly_detection.cpp
 * @brief Ejemplo 6: Detección de Anomalías con Autoencoder usando TensorFlow C++
 * 
 * Este ejemplo demuestra cómo implementar detección de anomalías usando una
 * red neuronal Autoencoder con la API de TensorFlow C++.
 * 
 * Un Autoencoder aprende a comprimir y reconstruir datos normales. Cuando se le
 * presentan datos anómalos, el error de reconstrucción será mayor, permitiéndonos
 * detectar anomalías basándonos en un umbral.
 * 
 * Conceptos clave demostrados:
 * - Arquitectura de Autoencoder (codificador-decodificador)
 * - Aprendizaje no supervisado para detección de anomalías
 * - Error de reconstrucción como puntuación de anomalía
 * - Clasificación basada en umbral
 * - Métricas de detección (precisión, recall, exactitud)
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
 * @brief Genera datos normales sintéticos (distribución Gaussiana 2D)
 * 
 * @param num_muestras Número de muestras normales a generar
 * @param datos Vector de salida para puntos de datos (aplanado)
 * @param media_x Media de la coordenada x
 * @param media_y Media de la coordenada y
 * @param desv_est Desviación estándar
 */
void generarDatosNormales(int num_muestras, std::vector<float>& datos,
                          float media_x, float media_y, float desv_est) {
    std::random_device rd;
    std::mt19937 gen(42); // Semilla fija para reproducibilidad
    std::normal_distribution<float> dist_x(media_x, desv_est);
    std::normal_distribution<float> dist_y(media_y, desv_est);
    
    for (int i = 0; i < num_muestras; ++i) {
        datos.push_back(dist_x(gen));
        datos.push_back(dist_y(gen));
    }
}

/**
 * @brief Genera datos de anomalías (puntos alejados de la distribución normal)
 * 
 * @param num_muestras Número de muestras de anomalías a generar
 * @param datos Vector de salida para puntos de datos (aplanado)
 */
void generarAnomalias(int num_muestras, std::vector<float>& datos) {
    std::random_device rd;
    std::mt19937 gen(123); // Semilla diferente para anomalías
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    
    for (int i = 0; i < num_muestras; ++i) {
        // Generar puntos alejados del centro de la distribución normal (0, 0)
        float x = dist(gen);
        float y = dist(gen);
        
        // Asegurar que las anomalías estén fuera de la región normal
        if (std::abs(x) < 2.0f) x += (x >= 0) ? 2.5f : -2.5f;
        if (std::abs(y) < 2.0f) y += (y >= 0) ? 2.5f : -2.5f;
        
        datos.push_back(x);
        datos.push_back(y);
    }
}

/**
 * @brief Crea un tensor 2D a partir de datos aplanados
 * @param datos Vector plano de características
 * @param num_muestras Número de muestras
 * @param num_caracteristicas Número de características por muestra
 * @return Tensor de TensorFlow
 */
Tensor crearTensor2D(const std::vector<float>& datos, 
                     int num_muestras, int num_caracteristicas) {
    Tensor tensor(DT_FLOAT, TensorShape({num_muestras, num_caracteristicas}));
    auto mapa_tensor = tensor.matrix<float>();
    for (int i = 0; i < num_muestras; ++i) {
        for (int j = 0; j < num_caracteristicas; ++j) {
            mapa_tensor(i, j) = datos[i * num_caracteristicas + j];
        }
    }
    return tensor;
}

/**
 * @brief Calcula el error de reconstrucción para cada muestra
 * @param original Tensor de datos originales
 * @param reconstruido Tensor de datos reconstruidos
 * @return Vector de errores de reconstrucción (ECM por muestra)
 */
std::vector<float> calcularErrorReconstruccion(const Tensor& original, 
                                               const Tensor& reconstruido) {
    auto datos_orig = original.matrix<float>();
    auto datos_recon = reconstruido.matrix<float>();
    int num_muestras = original.dim_size(0);
    int num_caracteristicas = original.dim_size(1);
    
    std::vector<float> errores;
    for (int i = 0; i < num_muestras; ++i) {
        float ecm = 0.0f;
        for (int j = 0; j < num_caracteristicas; ++j) {
            float diff = datos_orig(i, j) - datos_recon(i, j);
            ecm += diff * diff;
        }
        errores.push_back(ecm / num_caracteristicas);
    }
    return errores;
}

/**
 * @brief Calcula métricas de detección
 * @param errores Errores de reconstrucción
 * @param etiquetas Etiquetas verdaderas (0 = normal, 1 = anomalía)
 * @param umbral Umbral de detección de anomalías
 * @return Tupla de (verdaderos positivos, falsos positivos, verdaderos negativos, falsos negativos)
 */
void calcularMetricas(const std::vector<float>& errores,
                      const std::vector<int>& etiquetas,
                      float umbral,
                      int& vp, int& fp, int& vn, int& fn) {
    vp = fp = vn = fn = 0;
    
    for (size_t i = 0; i < errores.size(); ++i) {
        bool anomalia_predicha = errores[i] > umbral;
        bool es_anomalia = etiquetas[i] == 1;
        
        if (anomalia_predicha && es_anomalia) vp++;
        else if (anomalia_predicha && !es_anomalia) fp++;
        else if (!anomalia_predicha && !es_anomalia) vn++;
        else fn++;
    }
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Detección de Anomalías con Autoencoder " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Hiperparámetros
    const int NUM_NORMALES = 200;
    const int NUM_ANOMALIAS = 20;
    const int NUM_EPOCAS = 500;
    const float TASA_APRENDIZAJE = 0.1f;
    
    // Arquitectura del Autoencoder: 2 -> 8 -> 4 -> 8 -> 2
    const int TAMANO_ENTRADA = 2;
    const int CODIFICADOR_H1 = 8;
    const int TAMANO_LATENTE = 4;
    const int DECODIFICADOR_H1 = 8;
    const int TAMANO_SALIDA = 2;
    
    std::cout << "\n=== Generando Datos ===" << std::endl;
    
    // Generar datos de entrenamiento normales
    std::vector<float> datos_normales;
    generarDatosNormales(NUM_NORMALES, datos_normales, 0.0f, 0.0f, 1.0f);
    
    // Generar datos de anomalías
    std::vector<float> datos_anomalias;
    generarAnomalias(NUM_ANOMALIAS, datos_anomalias);
    
    std::cout << "- Datos normales: " << NUM_NORMALES << " puntos" << std::endl;
    std::cout << "- Anomalías: " << NUM_ANOMALIAS << " puntos" << std::endl;
    
    // Crear tensor de entrenamiento (solo datos normales para entrenar)
    Tensor tensor_entrenamiento = crearTensor2D(datos_normales, NUM_NORMALES, TAMANO_ENTRADA);
    
    // Crear datos de prueba (normales + anomalías)
    std::vector<float> datos_prueba = datos_normales;
    datos_prueba.insert(datos_prueba.end(), datos_anomalias.begin(), datos_anomalias.end());
    int total_prueba = NUM_NORMALES + NUM_ANOMALIAS;
    Tensor tensor_prueba = crearTensor2D(datos_prueba, total_prueba, TAMANO_ENTRADA);
    
    // Crear etiquetas (0 = normal, 1 = anomalía)
    std::vector<int> etiquetas_prueba(NUM_NORMALES, 0);
    etiquetas_prueba.insert(etiquetas_prueba.end(), NUM_ANOMALIAS, 1);
    
    std::cout << "\n=== Arquitectura del Autoencoder ===" << std::endl;
    std::cout << "Codificador: " << TAMANO_ENTRADA << " -> " << CODIFICADOR_H1 << " -> " << TAMANO_LATENTE << std::endl;
    std::cout << "Decodificador: " << TAMANO_LATENTE << " -> " << DECODIFICADOR_H1 << " -> " << TAMANO_SALIDA << std::endl;
    
    // Crear ámbito de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Marcador de posición de entrada
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_ENTRADA}));
    
    // Pesos y sesgos del codificador
    auto cod_w1_var = Variable(root.WithOpName("cod_W1"), 
                               {TAMANO_ENTRADA, CODIFICADOR_H1}, DT_FLOAT);
    auto cod_b1_var = Variable(root.WithOpName("cod_b1"), 
                               {1, CODIFICADOR_H1}, DT_FLOAT);
    auto cod_w2_var = Variable(root.WithOpName("cod_W2"), 
                               {CODIFICADOR_H1, TAMANO_LATENTE}, DT_FLOAT);
    auto cod_b2_var = Variable(root.WithOpName("cod_b2"), 
                               {1, TAMANO_LATENTE}, DT_FLOAT);
    
    // Pesos y sesgos del decodificador
    auto dec_w1_var = Variable(root.WithOpName("dec_W1"), 
                               {TAMANO_LATENTE, DECODIFICADOR_H1}, DT_FLOAT);
    auto dec_b1_var = Variable(root.WithOpName("dec_b1"), 
                               {1, DECODIFICADOR_H1}, DT_FLOAT);
    auto dec_w2_var = Variable(root.WithOpName("dec_W2"), 
                               {DECODIFICADOR_H1, TAMANO_SALIDA}, DT_FLOAT);
    auto dec_b2_var = Variable(root.WithOpName("dec_b2"), 
                               {1, TAMANO_SALIDA}, DT_FLOAT);
    
    // Inicializar pesos con inicialización Xavier
    float escala_cod_w1 = std::sqrt(2.0f / TAMANO_ENTRADA);
    float escala_cod_w2 = std::sqrt(2.0f / CODIFICADOR_H1);
    float escala_dec_w1 = std::sqrt(2.0f / TAMANO_LATENTE);
    float escala_dec_w2 = std::sqrt(2.0f / DECODIFICADOR_H1);
    
    auto cod_w1_init = Assign(root, cod_w1_var,
        Mul(root, RandomNormal(root, {TAMANO_ENTRADA, CODIFICADOR_H1}, DT_FLOAT),
            Const(root, escala_cod_w1)));
    auto cod_b1_init = Assign(root, cod_b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, CODIFICADOR_H1}))));
    auto cod_w2_init = Assign(root, cod_w2_var,
        Mul(root, RandomNormal(root, {CODIFICADOR_H1, TAMANO_LATENTE}, DT_FLOAT),
            Const(root, escala_cod_w2)));
    auto cod_b2_init = Assign(root, cod_b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_LATENTE}))));
    
    auto dec_w1_init = Assign(root, dec_w1_var,
        Mul(root, RandomNormal(root, {TAMANO_LATENTE, DECODIFICADOR_H1}, DT_FLOAT),
            Const(root, escala_dec_w1)));
    auto dec_b1_init = Assign(root, dec_b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, DECODIFICADOR_H1}))));
    auto dec_w2_init = Assign(root, dec_w2_var,
        Mul(root, RandomNormal(root, {DECODIFICADOR_H1, TAMANO_SALIDA}, DT_FLOAT),
            Const(root, escala_dec_w2)));
    auto dec_b2_init = Assign(root, dec_b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_SALIDA}))));
    
    // Propagación hacia adelante - Codificador
    auto cod_z1 = Add(root, MatMul(root, x_ph, cod_w1_var), cod_b1_var);
    auto cod_h1 = Relu(root.WithOpName("cod_oculta1"), cod_z1);
    
    auto cod_z2 = Add(root, MatMul(root, cod_h1, cod_w2_var), cod_b2_var);
    auto latente = Relu(root.WithOpName("latente"), cod_z2);
    
    // Propagación hacia adelante - Decodificador
    auto dec_z1 = Add(root, MatMul(root, latente, dec_w1_var), dec_b1_var);
    auto dec_h1 = Relu(root.WithOpName("dec_oculta1"), dec_z1);
    
    auto dec_z2 = Add(root, MatMul(root, dec_h1, dec_w2_var), dec_b2_var);
    auto x_reconstruido = Identity(root.WithOpName("salida"), dec_z2);
    
    // Pérdida: Error Cuadrático Medio
    auto error = Sub(root, x_ph, x_reconstruido);
    auto error_cuadrado = Square(root, error);
    auto perdida = Mean(root.WithOpName("perdida"), error_cuadrado, {0, 1});
    
    // Retropropagación - Cálculo manual de gradientes
    auto uno = Const(root, 1.0f);
    auto num_muestras_f = Const(root, static_cast<float>(NUM_NORMALES));
    
    // Gradientes de capa de salida (capa 2 del decodificador)
    auto d_salida = Mul(root, Const(root, 2.0f / (NUM_NORMALES * TAMANO_SALIDA)), 
                        Sub(root, x_reconstruido, x_ph));
    auto d_dec_w2 = MatMul(root, dec_h1, d_salida, MatMul::TransposeA(true));
    auto d_dec_b2 = Mean(root, d_salida, {0});
    
    // Gradientes de capa oculta del decodificador (capa 1 del decodificador)
    auto d_dec_h1_pre = MatMul(root, d_salida, dec_w2_var, MatMul::TransposeB(true));
    auto mascara_relu_dec = Cast(root, Greater(root, dec_z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_dec_h1 = Mul(root, d_dec_h1_pre, mascara_relu_dec);
    auto d_dec_w1 = MatMul(root, latente, d_dec_h1, MatMul::TransposeA(true));
    auto d_dec_b1 = Mean(root, d_dec_h1, {0});
    
    // Gradientes de capa latente (capa 2 del codificador)
    auto d_latente_pre = MatMul(root, d_dec_h1, dec_w1_var, MatMul::TransposeB(true));
    auto mascara_relu_latente = Cast(root, Greater(root, cod_z2, Const(root, 0.0f)), DT_FLOAT);
    auto d_latente = Mul(root, d_latente_pre, mascara_relu_latente);
    auto d_cod_w2 = MatMul(root, cod_h1, d_latente, MatMul::TransposeA(true));
    auto d_cod_b2 = Mean(root, d_latente, {0});
    
    // Gradientes de capa oculta del codificador (capa 1 del codificador)
    auto d_cod_h1_pre = MatMul(root, d_latente, cod_w2_var, MatMul::TransposeB(true));
    auto mascara_relu_cod = Cast(root, Greater(root, cod_z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_cod_h1 = Mul(root, d_cod_h1_pre, mascara_relu_cod);
    auto d_cod_w1 = MatMul(root, x_ph, d_cod_h1, MatMul::TransposeA(true));
    auto d_cod_b1 = Mean(root, d_cod_h1, {0});
    
    // Actualizaciones por descenso de gradiente
    auto lr = Const(root, TASA_APRENDIZAJE);
    
    auto cod_w1_update = AssignSub(root, cod_w1_var, Mul(root, lr, d_cod_w1));
    auto cod_b1_update = AssignSub(root, cod_b1_var, 
        Mul(root, lr, Reshape(root, d_cod_b1, {1, CODIFICADOR_H1})));
    auto cod_w2_update = AssignSub(root, cod_w2_var, Mul(root, lr, d_cod_w2));
    auto cod_b2_update = AssignSub(root, cod_b2_var,
        Mul(root, lr, Reshape(root, d_cod_b2, {1, TAMANO_LATENTE})));
    
    auto dec_w1_update = AssignSub(root, dec_w1_var, Mul(root, lr, d_dec_w1));
    auto dec_b1_update = AssignSub(root, dec_b1_var,
        Mul(root, lr, Reshape(root, d_dec_b1, {1, DECODIFICADOR_H1})));
    auto dec_w2_update = AssignSub(root, dec_w2_var, Mul(root, lr, d_dec_w2));
    auto dec_b2_update = AssignSub(root, dec_b2_var,
        Mul(root, lr, Reshape(root, d_dec_b2, {1, TAMANO_SALIDA})));
    
    // Crear sesión
    ClientSession session(root);
    
    // Inicializar todas las variables
    TF_CHECK_OK(session.Run({cod_w1_init, cod_b1_init, cod_w2_init, cod_b2_init,
                             dec_w1_init, dec_b1_init, dec_w2_init, dec_b2_init}, nullptr));
    
    std::cout << "\n=== Entrenando Autoencoder ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Bucle de entrenamiento
    std::vector<Tensor> salidas;
    for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
        TF_CHECK_OK(session.Run(
            {{x_ph, tensor_entrenamiento}},
            {perdida, cod_w1_update, cod_b1_update, cod_w2_update, cod_b2_update,
             dec_w1_update, dec_b1_update, dec_w2_update, dec_b2_update},
            &salidas));
        
        if (epoca % 100 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            std::cout << "Época " << std::setw(4) << epoca
                      << " - Pérdida: " << perdida_actual << std::endl;
        }
    }
    
    // Evaluar con datos de prueba (normales + anomalías)
    std::cout << "\n=== Resultados de Detección ===" << std::endl;
    
    std::vector<Tensor> salidas_prueba;
    TF_CHECK_OK(session.Run(
        {{x_ph, tensor_prueba}},
        {x_reconstruido},
        &salidas_prueba));
    
    // Calcular errores de reconstrucción
    std::vector<float> errores_recon = calcularErrorReconstruccion(tensor_prueba, salidas_prueba[0]);
    
    // Calcular umbral basado en errores de datos normales (media + 2*desv_est)
    float media_error = 0.0f;
    for (int i = 0; i < NUM_NORMALES; ++i) {
        media_error += errores_recon[i];
    }
    media_error /= NUM_NORMALES;
    
    float desv_error = 0.0f;
    for (int i = 0; i < NUM_NORMALES; ++i) {
        float diff = errores_recon[i] - media_error;
        desv_error += diff * diff;
    }
    desv_error = std::sqrt(desv_error / NUM_NORMALES);
    
    float umbral = media_error + 2.0f * desv_error;
    
    std::cout << "Umbral de anomalía: " << umbral << std::endl;
    std::cout << "(basado en media + 2*desv_est del error de reconstrucción)" << std::endl;
    
    // Calcular métricas de detección
    int vp, fp, vn, fn;
    calcularMetricas(errores_recon, etiquetas_prueba, umbral, vp, fp, vn, fn);
    
    int anomalias_detectadas = vp;
    int total_anomalias = NUM_ANOMALIAS;
    
    std::cout << "\nAnomalías detectadas: " << anomalias_detectadas << "/" << total_anomalias << std::endl;
    std::cout << "Falsos positivos: " << fp << std::endl;
    std::cout << "Falsos negativos: " << fn << std::endl;
    
    float precision = (vp + fp > 0) ? (100.0f * vp / (vp + fp)) : 0.0f;
    float recall = (vp + fn > 0) ? (100.0f * vp / (vp + fn)) : 0.0f;
    float exactitud = 100.0f * (vp + vn) / (vp + vn + fp + fn);
    float puntuacion_f1 = (precision + recall > 0) ? (2.0f * precision * recall / (precision + recall)) : 0.0f;
    
    std::cout << "\n=== Métricas de Detección ===" << std::endl;
    std::cout << "Precisión: " << std::setprecision(1) << precision << "%" << std::endl;
    std::cout << "Recall: " << recall << "%" << std::endl;
    std::cout << "Exactitud: " << exactitud << "%" << std::endl;
    std::cout << "Puntuación F1: " << std::setprecision(2) << puntuacion_f1 << "%" << std::endl;
    
    // Mostrar algunos ejemplos de reconstrucciones
    std::cout << "\n=== Ejemplos de Reconstrucción ===" << std::endl;
    std::cout << std::setprecision(3);
    std::cout << std::setw(10) << "Original X" << " | "
              << std::setw(10) << "Original Y" << " | "
              << std::setw(10) << "Recon X" << " | "
              << std::setw(10) << "Recon Y" << " | "
              << std::setw(10) << "Error" << " | "
              << std::setw(10) << "Tipo" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    auto mat_datos_prueba = tensor_prueba.matrix<float>();
    auto mat_datos_recon = salidas_prueba[0].matrix<float>();
    
    // Mostrar primeros 5 normales y primeras 5 anomalías
    std::cout << "Datos normales:" << std::endl;
    for (int i = 0; i < std::min(5, NUM_NORMALES); ++i) {
        std::cout << std::setw(10) << mat_datos_prueba(i, 0) << " | "
                  << std::setw(10) << mat_datos_prueba(i, 1) << " | "
                  << std::setw(10) << mat_datos_recon(i, 0) << " | "
                  << std::setw(10) << mat_datos_recon(i, 1) << " | "
                  << std::setw(10) << errores_recon[i] << " | "
                  << (errores_recon[i] > umbral ? "ANOMALÍA" : "Normal") << std::endl;
    }
    
    std::cout << "\nAnomalías:" << std::endl;
    for (int i = NUM_NORMALES; i < std::min(NUM_NORMALES + 5, total_prueba); ++i) {
        std::cout << std::setw(10) << mat_datos_prueba(i, 0) << " | "
                  << std::setw(10) << mat_datos_prueba(i, 1) << " | "
                  << std::setw(10) << mat_datos_recon(i, 0) << " | "
                  << std::setw(10) << mat_datos_recon(i, 1) << " | "
                  << std::setw(10) << errores_recon[i] << " | "
                  << (errores_recon[i] > umbral ? "ANOMALÍA" : "Normal") << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  ¡Detección de anomalías completada!    " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
