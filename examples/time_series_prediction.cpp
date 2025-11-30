/**
 * @file time_series_prediction.cpp
 * @brief Ejemplo 7: Predicción de Series Temporales usando TensorFlow C++
 * 
 * Este ejemplo demuestra cómo implementar predicción de series temporales usando
 * una red neuronal feedforward con la API de TensorFlow C++.
 * 
 * El ejemplo genera datos de series temporales sintéticas (onda sinusoidal con tendencia y ruido)
 * y entrena una red neuronal para predecir el siguiente valor basándose en una ventana
 * deslizante de valores anteriores.
 * 
 * Conceptos clave demostrados:
 * - Preparación de datos de series temporales (ventana deslizante / ventaneo)
 * - Red feedforward para predicción de secuencias
 * - Métricas de regresión (EAM, RECM)
 * - Manejo de dependencias temporales con ventanas de entrada de tamaño fijo
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
 * @brief Genera datos de series temporales sintéticas
 * 
 * Crea una serie temporal: y(t) = sin(2*pi*t/periodo) + tendencia*t + ruido
 * 
 * @param longitud Número de puntos temporales
 * @param periodo Período de la onda sinusoidal
 * @param tendencia Coeficiente de tendencia lineal
 * @param nivel_ruido Desviación estándar del ruido Gaussiano
 * @param datos Vector de salida para valores de la serie temporal
 */
void generarSerieTemporal(int longitud, float periodo, float tendencia, 
                          float nivel_ruido, std::vector<float>& datos) {
    std::random_device rd;
    std::mt19937 gen(42); // Semilla fija para reproducibilidad
    std::normal_distribution<float> ruido(0.0f, nivel_ruido);
    
    datos.clear();
    const float PI = 3.14159265358979323846f;
    
    for (int t = 0; t < longitud; ++t) {
        float valor = std::sin(2.0f * PI * t / periodo) + tendencia * t / longitud + ruido(gen);
        datos.push_back(valor);
    }
}

/**
 * @brief Crea características y objetivos de ventana deslizante a partir de la serie temporal
 * 
 * @param serie Serie temporal de entrada
 * @param tamano_ventana Número de puntos anteriores a usar como características
 * @param datos_x Vectores de características de salida (ventanas)
 * @param datos_y Valores objetivo de salida (siguiente valor después de cada ventana)
 */
void crearVentanasDeslizantes(const std::vector<float>& serie, int tamano_ventana,
                              std::vector<float>& datos_x, std::vector<float>& datos_y) {
    datos_x.clear();
    datos_y.clear();
    
    int num_ventanas = serie.size() - tamano_ventana;
    
    for (int i = 0; i < num_ventanas; ++i) {
        // Características de la ventana
        for (int j = 0; j < tamano_ventana; ++j) {
            datos_x.push_back(serie[i + j]);
        }
        // Objetivo: siguiente valor después de la ventana
        datos_y.push_back(serie[i + tamano_ventana]);
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
 * @brief Crea un tensor columna 1D a partir de un vector
 * @param datos Vector de valores
 * @return Tensor de TensorFlow (N x 1)
 */
Tensor crearTensor1D(const std::vector<float>& datos) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(datos.size()), 1}));
    auto mapa_tensor = tensor.matrix<float>();
    for (size_t i = 0; i < datos.size(); ++i) {
        mapa_tensor(i, 0) = datos[i];
    }
    return tensor;
}

/**
 * @brief Calcula el Error Absoluto Medio
 * @param predicciones Tensor de valores predichos
 * @param objetivos Tensor de valores objetivo
 * @return Valor de EAM
 */
float calcularEAM(const Tensor& predicciones, const Tensor& objetivos) {
    auto datos_pred = predicciones.matrix<float>();
    auto datos_obj = objetivos.matrix<float>();
    int n = predicciones.dim_size(0);
    
    float eam = 0.0f;
    for (int i = 0; i < n; ++i) {
        eam += std::abs(datos_pred(i, 0) - datos_obj(i, 0));
    }
    return eam / n;
}

/**
 * @brief Calcula la Raíz del Error Cuadrático Medio
 * @param predicciones Tensor de valores predichos
 * @param objetivos Tensor de valores objetivo
 * @return Valor de RECM
 */
float calcularRECM(const Tensor& predicciones, const Tensor& objetivos) {
    auto datos_pred = predicciones.matrix<float>();
    auto datos_obj = objetivos.matrix<float>();
    int n = predicciones.dim_size(0);
    
    float ecm = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = datos_pred(i, 0) - datos_obj(i, 0);
        ecm += diff * diff;
    }
    return std::sqrt(ecm / n);
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "       Predicción de Series Temporales   " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Hiperparámetros
    const int LONGITUD_SERIE = 500;
    const int TAMANO_VENTANA = 10;
    const float PERIODO_SENO = 50.0f;
    const float TENDENCIA = 0.5f;
    const float NIVEL_RUIDO = 0.1f;
    
    const int NUM_EPOCAS = 1000;
    const float TASA_APRENDIZAJE = 0.01f;
    
    // Arquitectura de la red: tamano_ventana -> 32 -> 16 -> 1
    const int TAMANO_OCULTA1 = 32;
    const int TAMANO_OCULTA2 = 16;
    const int TAMANO_SALIDA = 1;
    
    std::cout << "\n=== Generando Serie Temporal ===" << std::endl;
    
    // Generar serie temporal
    std::vector<float> serie;
    generarSerieTemporal(LONGITUD_SERIE, PERIODO_SENO, TENDENCIA, NIVEL_RUIDO, serie);
    
    std::cout << "Longitud: " << LONGITUD_SERIE << " puntos" << std::endl;
    std::cout << "Período de onda: " << PERIODO_SENO << std::endl;
    std::cout << "Tendencia: " << TENDENCIA << std::endl;
    std::cout << "Nivel de ruido: " << NIVEL_RUIDO << std::endl;
    std::cout << "Tamaño de ventana: " << TAMANO_VENTANA << std::endl;
    
    // Crear ventanas deslizantes
    std::vector<float> datos_x, datos_y;
    crearVentanasDeslizantes(serie, TAMANO_VENTANA, datos_x, datos_y);
    
    int num_muestras = datos_y.size();
    
    // Dividir en entrenamiento y prueba (80/20)
    int tamano_entrenamiento = static_cast<int>(num_muestras * 0.8);
    int tamano_prueba = num_muestras - tamano_entrenamiento;
    
    std::vector<float> x_entrenamiento(datos_x.begin(), datos_x.begin() + tamano_entrenamiento * TAMANO_VENTANA);
    std::vector<float> y_entrenamiento(datos_y.begin(), datos_y.begin() + tamano_entrenamiento);
    std::vector<float> x_prueba(datos_x.begin() + tamano_entrenamiento * TAMANO_VENTANA, datos_x.end());
    std::vector<float> y_prueba(datos_y.begin() + tamano_entrenamiento, datos_y.end());
    
    Tensor tensor_x_entrenamiento = crearTensor2D(x_entrenamiento, tamano_entrenamiento, TAMANO_VENTANA);
    Tensor tensor_y_entrenamiento = crearTensor1D(y_entrenamiento);
    Tensor tensor_x_prueba = crearTensor2D(x_prueba, tamano_prueba, TAMANO_VENTANA);
    Tensor tensor_y_prueba = crearTensor1D(y_prueba);
    
    std::cout << "\nMuestras de entrenamiento: " << tamano_entrenamiento << std::endl;
    std::cout << "Muestras de prueba: " << tamano_prueba << std::endl;
    
    std::cout << "\n=== Arquitectura de la Red ===" << std::endl;
    std::cout << "Entrada: " << TAMANO_VENTANA << " (ventana temporal)" << std::endl;
    std::cout << "Oculta 1: " << TAMANO_OCULTA1 << " (ReLU)" << std::endl;
    std::cout << "Oculta 2: " << TAMANO_OCULTA2 << " (ReLU)" << std::endl;
    std::cout << "Salida: " << TAMANO_SALIDA << " (lineal)" << std::endl;
    
    // Crear ámbito de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Marcador de posición de entrada
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_VENTANA}));
    auto y_ph = Placeholder(root.WithOpName("Y"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_SALIDA}));
    
    // Pesos y sesgos de capa oculta 1
    auto w1_var = Variable(root.WithOpName("W1"), 
                           {TAMANO_VENTANA, TAMANO_OCULTA1}, DT_FLOAT);
    auto b1_var = Variable(root.WithOpName("b1"), 
                           {1, TAMANO_OCULTA1}, DT_FLOAT);
    
    // Pesos y sesgos de capa oculta 2
    auto w2_var = Variable(root.WithOpName("W2"), 
                           {TAMANO_OCULTA1, TAMANO_OCULTA2}, DT_FLOAT);
    auto b2_var = Variable(root.WithOpName("b2"), 
                           {1, TAMANO_OCULTA2}, DT_FLOAT);
    
    // Pesos y sesgos de capa de salida
    auto w3_var = Variable(root.WithOpName("W3"), 
                           {TAMANO_OCULTA2, TAMANO_SALIDA}, DT_FLOAT);
    auto b3_var = Variable(root.WithOpName("b3"), 
                           {1, TAMANO_SALIDA}, DT_FLOAT);
    
    // Inicializar pesos con inicialización Xavier
    float escala_w1 = std::sqrt(2.0f / TAMANO_VENTANA);
    float escala_w2 = std::sqrt(2.0f / TAMANO_OCULTA1);
    float escala_w3 = std::sqrt(2.0f / TAMANO_OCULTA2);
    
    auto w1_init = Assign(root, w1_var,
        Mul(root, RandomNormal(root, {TAMANO_VENTANA, TAMANO_OCULTA1}, DT_FLOAT),
            Const(root, escala_w1)));
    auto b1_init = Assign(root, b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_OCULTA1}))));
    auto w2_init = Assign(root, w2_var,
        Mul(root, RandomNormal(root, {TAMANO_OCULTA1, TAMANO_OCULTA2}, DT_FLOAT),
            Const(root, escala_w2)));
    auto b2_init = Assign(root, b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_OCULTA2}))));
    auto w3_init = Assign(root, w3_var,
        Mul(root, RandomNormal(root, {TAMANO_OCULTA2, TAMANO_SALIDA}, DT_FLOAT),
            Const(root, escala_w3)));
    auto b3_init = Assign(root, b3_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_SALIDA}))));
    
    // Propagación hacia adelante
    auto z1 = Add(root, MatMul(root, x_ph, w1_var), b1_var);
    auto h1 = Relu(root.WithOpName("oculta1"), z1);
    
    auto z2 = Add(root, MatMul(root, h1, w2_var), b2_var);
    auto h2 = Relu(root.WithOpName("oculta2"), z2);
    
    auto z3 = Add(root, MatMul(root, h2, w3_var), b3_var);
    auto y_pred = Identity(root.WithOpName("salida"), z3);
    
    // Pérdida: Error Cuadrático Medio
    auto error = Sub(root, y_pred, y_ph);
    auto error_cuadrado = Square(root, error);
    auto perdida = Mean(root.WithOpName("perdida"), error_cuadrado, {0, 1});
    
    // Retropropagación - Cálculo manual de gradientes
    auto d_salida = Mul(root, Const(root, 2.0f / tamano_entrenamiento), error);
    
    // Gradientes de capa de salida
    auto d_w3 = MatMul(root, h2, d_salida, MatMul::TransposeA(true));
    auto d_b3 = Mean(root, d_salida, {0});
    
    // Gradientes de capa oculta 2
    auto d_h2_pre = MatMul(root, d_salida, w3_var, MatMul::TransposeB(true));
    auto mascara_relu_h2 = Cast(root, Greater(root, z2, Const(root, 0.0f)), DT_FLOAT);
    auto d_h2 = Mul(root, d_h2_pre, mascara_relu_h2);
    auto d_w2 = MatMul(root, h1, d_h2, MatMul::TransposeA(true));
    auto d_b2 = Mean(root, d_h2, {0});
    
    // Gradientes de capa oculta 1
    auto d_h1_pre = MatMul(root, d_h2, w2_var, MatMul::TransposeB(true));
    auto mascara_relu_h1 = Cast(root, Greater(root, z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_h1 = Mul(root, d_h1_pre, mascara_relu_h1);
    auto d_w1 = MatMul(root, x_ph, d_h1, MatMul::TransposeA(true));
    auto d_b1 = Mean(root, d_h1, {0});
    
    // Actualizaciones por descenso de gradiente
    auto lr = Const(root, TASA_APRENDIZAJE);
    
    auto w1_update = AssignSub(root, w1_var, Mul(root, lr, d_w1));
    auto b1_update = AssignSub(root, b1_var, 
        Mul(root, lr, Reshape(root, d_b1, {1, TAMANO_OCULTA1})));
    auto w2_update = AssignSub(root, w2_var, Mul(root, lr, d_w2));
    auto b2_update = AssignSub(root, b2_var,
        Mul(root, lr, Reshape(root, d_b2, {1, TAMANO_OCULTA2})));
    auto w3_update = AssignSub(root, w3_var, Mul(root, lr, d_w3));
    auto b3_update = AssignSub(root, b3_var,
        Mul(root, lr, Reshape(root, d_b3, {1, TAMANO_SALIDA})));
    
    // Crear sesión
    ClientSession session(root);
    
    // Inicializar todas las variables
    TF_CHECK_OK(session.Run({w1_init, b1_init, w2_init, b2_init, w3_init, b3_init}, nullptr));
    
    std::cout << "\n=== Entrenando Modelo ===" << std::endl;
    std::cout << "Tasa de aprendizaje: " << TASA_APRENDIZAJE << std::endl;
    std::cout << "Épocas: " << NUM_EPOCAS << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Bucle de entrenamiento
    std::vector<Tensor> salidas;
    for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
        TF_CHECK_OK(session.Run(
            {{x_ph, tensor_x_entrenamiento}, {y_ph, tensor_y_entrenamiento}},
            {perdida, w1_update, b1_update, w2_update, b2_update, w3_update, b3_update},
            &salidas));
        
        if (epoca % 200 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            std::cout << "Época " << std::setw(4) << epoca
                      << " - Pérdida: " << perdida_actual << std::endl;
        }
    }
    
    // Evaluar con datos de prueba
    std::cout << "\n=== Métricas de Predicción ===" << std::endl;
    
    std::vector<Tensor> salidas_prueba;
    TF_CHECK_OK(session.Run(
        {{x_ph, tensor_x_prueba}},
        {y_pred},
        &salidas_prueba));
    
    float eam = calcularEAM(salidas_prueba[0], tensor_y_prueba);
    float recm = calcularRECM(salidas_prueba[0], tensor_y_prueba);
    
    std::cout << "EAM: " << eam << std::endl;
    std::cout << "RECM: " << recm << std::endl;
    
    // Mostrar predicciones de muestra
    std::cout << "\n=== Predicciones de Muestra ===" << std::endl;
    std::cout << std::setw(8) << "t" << " | "
              << std::setw(12) << "Predicho" << " | "
              << std::setw(12) << "Real" << " | "
              << std::setw(12) << "Error" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    auto datos_pred = salidas_prueba[0].matrix<float>();
    auto datos_objetivo = tensor_y_prueba.matrix<float>();
    
    // Mostrar predicciones en puntos temporales específicos
    std::vector<int> indices_muestra = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90};
    
    for (int idx : indices_muestra) {
        if (idx < tamano_prueba) {
            int t = tamano_entrenamiento + TAMANO_VENTANA + idx;
            float predicho = datos_pred(idx, 0);
            float real = datos_objetivo(idx, 0);
            float valor_error = std::abs(predicho - real);
            
            std::cout << std::setw(8) << t << " | "
                      << std::setw(12) << predicho << " | "
                      << std::setw(12) << real << " | "
                      << std::setw(12) << valor_error << std::endl;
        }
    }
    
    // Evaluar con datos de entrenamiento para comparación
    std::vector<Tensor> salidas_entrenamiento;
    TF_CHECK_OK(session.Run(
        {{x_ph, tensor_x_entrenamiento}},
        {y_pred},
        &salidas_entrenamiento));
    
    float eam_entrenamiento = calcularEAM(salidas_entrenamiento[0], tensor_y_entrenamiento);
    float recm_entrenamiento = calcularRECM(salidas_entrenamiento[0], tensor_y_entrenamiento);
    
    std::cout << "\n=== Comparación Entrenamiento vs Prueba ===" << std::endl;
    std::cout << "EAM Entrenamiento:  " << eam_entrenamiento << std::endl;
    std::cout << "RECM Entrenamiento: " << recm_entrenamiento << std::endl;
    std::cout << "EAM Prueba:         " << eam << std::endl;
    std::cout << "RECM Prueba:        " << recm << std::endl;
    
    // Mostrar fragmento de serie temporal
    std::cout << "\n=== Fragmento de Serie Temporal ===" << std::endl;
    std::cout << "(Primeros 20 valores)" << std::endl;
    std::cout << std::setprecision(3);
    for (int i = 0; i < std::min(20, LONGITUD_SERIE); ++i) {
        std::cout << "t=" << std::setw(3) << i << ": " << std::setw(7) << serie[i];
        if ((i + 1) % 5 == 0) std::cout << std::endl;
        else std::cout << " | ";
    }
    
    std::cout << "\n\n==========================================" << std::endl;
    std::cout << "  ¡Predicción de series temporales completada!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
