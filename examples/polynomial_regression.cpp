/**
 * @file polynomial_regression.cpp
 * @brief Ejemplo 4: Regresión Polinómica usando TensorFlow C++
 * 
 * Este ejemplo demuestra cómo implementar regresión polinómica
 * usando la API de TensorFlow C++. El modelo aprende a ajustar una función
 * polinómica y = 0.5x³ - x² + 0.5x + 1 a un conjunto de puntos de datos de entrenamiento.
 * 
 * Conceptos clave demostrados:
 * - Creación de tensores y operaciones
 * - Construcción de grafos computacionales
 * - Entrenamiento con descenso de gradiente
 * - Uso de variables y marcadores de posición
 * - Ingeniería de características (características polinómicas)
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
 * @brief Genera datos de entrenamiento sintéticos para regresión polinómica
 * 
 * Genera puntos de datos siguiendo y = 0.5x³ - x² + 0.5x + 1 + ruido
 * 
 * @param num_muestras Número de puntos de datos a generar
 * @param nivel_ruido Desviación estándar del ruido Gaussiano
 * @param datos_x Vector de salida para valores x
 * @param datos_y Vector de salida para valores y
 */
void generarDatosPolinomicos(int num_muestras, float nivel_ruido,
                             std::vector<float>& datos_x,
                             std::vector<float>& datos_y) {
    std::mt19937 gen(42);  // Semilla fija para reproducibilidad
    std::normal_distribution<float> ruido(0.0f, nivel_ruido);
    std::uniform_real_distribution<float> dist_x(-2.0f, 2.0f);
    
    datos_x.resize(num_muestras);
    datos_y.resize(num_muestras);
    
    // Coeficientes reales: y = 0.5x³ - x² + 0.5x + 1
    const float c3 = 0.5f;   // coeficiente de x³
    const float c2 = -1.0f;  // coeficiente de x²
    const float c1 = 0.5f;   // coeficiente de x
    const float c0 = 1.0f;   // intercepto
    
    for (int i = 0; i < num_muestras; ++i) {
        float x = dist_x(gen);
        datos_x[i] = x;
        float x2 = x * x;
        float x3 = x2 * x;
        datos_y[i] = c3 * x3 + c2 * x2 + c1 * x + c0 + ruido(gen);
    }
}

/**
 * @brief Crea un tensor con características polinómicas [x, x², x³]
 * 
 * @param datos_x Valores de entrada x
 * @return Tensor de TensorFlow con forma (num_muestras, 3)
 */
Tensor crearCaracteristicasPolinomicas(const std::vector<float>& datos_x) {
    int num_muestras = static_cast<int>(datos_x.size());
    Tensor tensor(DT_FLOAT, TensorShape({num_muestras, 3}));
    auto mapa_tensor = tensor.matrix<float>();
    
    for (int i = 0; i < num_muestras; ++i) {
        float x = datos_x[i];
        mapa_tensor(i, 0) = x;           // x
        mapa_tensor(i, 1) = x * x;       // x²
        mapa_tensor(i, 2) = x * x * x;   // x³
    }
    return tensor;
}

/**
 * @brief Crea un tensor a partir de valores y
 * 
 * @param datos_y Valores de entrada y
 * @return Tensor de TensorFlow con forma (num_muestras, 1)
 */
Tensor crearTensorEtiquetas(const std::vector<float>& datos_y) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(datos_y.size()), 1}));
    auto mapa_tensor = tensor.matrix<float>();
    for (size_t i = 0; i < datos_y.size(); ++i) {
        mapa_tensor(i, 0) = datos_y[i];
    }
    return tensor;
}

/**
 * @brief Demostración principal de regresión polinómica
 */
int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Regresión Polinómica TensorFlow C++    " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Parámetros de verdad base (intentaremos aprender estos)
    // y = 0.5x³ - x² + 0.5x + 1
    const float C3_REAL = 0.5f;   // coeficiente de x³
    const float C2_REAL = -1.0f;  // coeficiente de x²
    const float C1_REAL = 0.5f;   // coeficiente de x
    const float C0_REAL = 1.0f;   // intercepto
    
    const float NIVEL_RUIDO = 0.1f;
    const int NUM_MUESTRAS = 100;
    const int NUM_EPOCAS = 2000;
    const float TASA_APRENDIZAJE = 0.01f;
    
    std::cout << "\n=== Función Polinómica ===" << std::endl;
    std::cout << "y = " << C3_REAL << "x³ + (" << C2_REAL << ")x² + " 
              << C1_REAL << "x + " << C0_REAL << std::endl;
    std::cout << "Nivel de ruido: " << NIVEL_RUIDO << std::endl;
    std::cout << "Número de muestras: " << NUM_MUESTRAS << std::endl;
    
    // Generar datos de entrenamiento
    std::vector<float> datos_x, datos_y;
    generarDatosPolinomicos(NUM_MUESTRAS, NIVEL_RUIDO, datos_x, datos_y);
    
    // Crear tensores a partir de los datos
    // Características: [x, x², x³]
    Tensor tensor_x = crearCaracteristicasPolinomicas(datos_x);
    Tensor tensor_y = crearTensorEtiquetas(datos_y);
    
    std::cout << "Se generaron " << NUM_MUESTRAS << " muestras de entrenamiento." << std::endl;
    std::cout << "Forma de características: [" << tensor_x.dim_size(0) << ", " 
              << tensor_x.dim_size(1) << "]" << std::endl;
    
    // Crear ámbito de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Crear marcadores de posición para datos de entrada
    // x_placeholder recibirá características polinómicas [x, x², x³]
    auto x_placeholder = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                     Placeholder::Shape({-1, 3}));
    auto y_placeholder = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    
    // Crear variables entrenables
    // Pesos para coeficientes polinómicos [c1, c2, c3] (para x, x², x³)
    auto w_init = Variable(root.WithOpName("pesos"), {3, 1}, DT_FLOAT);
    auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                          RandomNormal(root, {3, 1}, DT_FLOAT));
    
    // Sesgo (intercepto c0)
    auto b_init = Variable(root.WithOpName("sesgo"), {1, 1}, DT_FLOAT);
    auto b_assign = Assign(root.WithOpName("b_assign"), b_init,
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    // Modelo: y_pred = x * w + b
    // Donde x contiene características polinómicas [x, x², x³]
    auto y_pred = Add(root.WithOpName("prediccion"),
                     MatMul(root, x_placeholder, w_init),
                     b_init);
    
    // Función de pérdida: Error Cuadrático Medio = media((y_pred - y)²)
    auto error = Sub(root, y_pred, y_placeholder);
    auto error_cuadrado = Square(root, error);
    auto perdida = Mean(root.WithOpName("perdida"), error_cuadrado, {0, 1});
    
    // Calcular gradientes manualmente
    // d(pérdida)/dw = 2 * media(error * x)
    // d(pérdida)/db = 2 * media(error)
    auto grad_w = Mul(root, 
                     Const(root, 2.0f),
                     Mean(root, Mul(root, 
                         Reshape(root, error, {-1, 1}),
                         x_placeholder), {0}));
    auto grad_b = Mul(root,
                     Const(root, 2.0f),
                     Mean(root, error, {0}));
    
    // Actualización por descenso de gradiente
    auto lr = Const(root, TASA_APRENDIZAJE);
    auto w_update = AssignSub(root, w_init, 
                             Mul(root, lr, Reshape(root, grad_w, {3, 1})));
    auto b_update = AssignSub(root, b_init, 
                             Mul(root, lr, Reshape(root, grad_b, {1, 1})));
    
    // Crear sesión
    ClientSession session(root);
    
    // Inicializar variables
    TF_CHECK_OK(session.Run({w_assign, b_assign}, nullptr));
    
    std::cout << "\n=== Entrenamiento ===" << std::endl;
    std::cout << "Tasa de aprendizaje: " << TASA_APRENDIZAJE << std::endl;
    std::cout << "Número de épocas: " << NUM_EPOCAS << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Bucle de entrenamiento
    std::vector<Tensor> salidas;
    for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
        // Ejecutar paso de entrenamiento
        TF_CHECK_OK(session.Run(
            {{x_placeholder, tensor_x}, {y_placeholder, tensor_y}},
            {perdida, w_update, b_update},
            &salidas));
        
        // Imprimir progreso cada 200 épocas
        if (epoca % 200 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            std::cout << "Época " << std::setw(4) << epoca 
                      << " | Pérdida: " << perdida_actual << std::endl;
        }
    }
    
    // Obtener parámetros aprendidos finales
    std::vector<Tensor> params_finales;
    TF_CHECK_OK(session.Run({w_init, b_init}, &params_finales));
    
    auto pesos = params_finales[0].matrix<float>();
    float c1_aprendido = pesos(0, 0);  // coeficiente de x
    float c2_aprendido = pesos(1, 0);  // coeficiente de x²
    float c3_aprendido = pesos(2, 0);  // coeficiente de x³
    float c0_aprendido = params_finales[1].matrix<float>()(0, 0);  // intercepto
    
    std::cout << "\n=== Coeficientes Aprendidos ===" << std::endl;
    std::cout << "Coeficiente  | Aprendido   | Real" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    std::cout << "x³ (c3)      | " << std::setw(10) << c3_aprendido 
              << " | " << C3_REAL << std::endl;
    std::cout << "x² (c2)      | " << std::setw(10) << c2_aprendido 
              << " | " << C2_REAL << std::endl;
    std::cout << "x  (c1)      | " << std::setw(10) << c1_aprendido 
              << " | " << C1_REAL << std::endl;
    std::cout << "intercepto   | " << std::setw(10) << c0_aprendido 
              << " | " << C0_REAL << std::endl;
    
    // Hacer predicciones con nuevos datos
    std::cout << "\n=== Predicciones ===" << std::endl;
    std::cout << "Probando con nuevos valores de x:" << std::endl;
    
    std::vector<float> test_x = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    Tensor tensor_test = crearCaracteristicasPolinomicas(test_x);
    
    std::vector<Tensor> predicciones;
    TF_CHECK_OK(session.Run(
        {{x_placeholder, tensor_test}},
        {y_pred},
        &predicciones));
    
    auto datos_pred = predicciones[0].matrix<float>();
    std::cout << std::setw(8) << "x" << " | " 
              << std::setw(12) << "Predicho" << " | "
              << std::setw(12) << "Real" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (size_t i = 0; i < test_x.size(); ++i) {
        float x = test_x[i];
        float y_real = C3_REAL * x * x * x + C2_REAL * x * x + C1_REAL * x + C0_REAL;
        std::cout << std::setw(8) << x << " | "
                  << std::setw(12) << datos_pred(i, 0) << " | "
                  << std::setw(12) << y_real << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  ¡Regresión polinómica completada!      " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
