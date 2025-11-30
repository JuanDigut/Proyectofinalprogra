/**
 * @file neural_network.cpp
 * @brief Ejemplo 3: Clasificación con Red Neuronal usando TensorFlow C++
 * 
 * Este ejemplo demuestra cómo construir y entrenar una red neuronal simple
 * para clasificación binaria usando la API de TensorFlow C++.
 * 
 * El ejemplo crea una red neuronal de 2 capas para clasificar datos sintéticos
 * en dos clases basándose en un límite de decisión similar a XOR.
 * 
 * Conceptos clave demostrados:
 * - Arquitectura de red neuronal multicapa
 * - Funciones de activación (Sigmoide)
 * - Propagación hacia adelante
 * - Cálculo de pérdida (Entropía Cruzada Binaria)
 * - Retropropagación con descenso de gradiente
 * - Métricas de precisión de clasificación
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
 * @brief Genera datos de clasificación sintéticos (patrón similar a XOR)
 * 
 * Crea un conjunto de datos donde la clase 1 corresponde a puntos donde x1*x2 > 0
 * (primer y tercer cuadrantes), y la clase 0 corresponde a puntos donde
 * x1*x2 < 0 (segundo y cuarto cuadrantes). Esto prueba la capacidad de la red
 * para aprender límites de decisión no lineales.
 * 
 * @param num_muestras Número total de muestras
 * @param datos_x Matriz de características de salida (num_muestras x 2)
 * @param datos_y Etiquetas de salida (0 o 1)
 */
void generarDatosClasificacion(int num_muestras, 
                               std::vector<float>& datos_x,
                               std::vector<float>& datos_y) {
    std::random_device rd;
    std::mt19937 gen(42); // Semilla fija para reproducibilidad
    std::uniform_real_distribution<float> uniforme(-2.0f, 2.0f);
    std::normal_distribution<float> ruido(0.0f, 0.1f);
    
    datos_x.clear();
    datos_y.clear();
    
    for (int i = 0; i < num_muestras; ++i) {
        float x1 = uniforme(gen);
        float x2 = uniforme(gen);
        
        // Patrón similar a XOR: etiqueta = 1 si (x1*x2 > 0), sino 0
        float etiqueta = (x1 * x2 > 0) ? 1.0f : 0.0f;
        
        // Agregar algo de ruido para hacerlo desafiante
        x1 += ruido(gen);
        x2 += ruido(gen);
        
        datos_x.push_back(x1);
        datos_x.push_back(x2);
        datos_y.push_back(etiqueta);
    }
}

/**
 * @brief Crea un tensor 2D a partir de datos de características
 * @param datos Vector plano de características
 * @param num_muestras Número de muestras
 * @param num_caracteristicas Número de características por muestra
 * @return Tensor de TensorFlow
 */
Tensor crearTensorCaracteristicas(const std::vector<float>& datos, 
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
 * @brief Crea un tensor 1D a partir de datos de etiquetas
 * @param datos Vector de etiquetas
 * @return Tensor de TensorFlow
 */
Tensor crearTensorEtiquetas(const std::vector<float>& datos) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(datos.size()), 1}));
    auto mapa_tensor = tensor.matrix<float>();
    for (size_t i = 0; i < datos.size(); ++i) {
        mapa_tensor(i, 0) = datos[i];
    }
    return tensor;
}

/**
 * @brief Calcula la precisión de clasificación
 * @param predicciones Probabilidades predichas
 * @param etiquetas Etiquetas verdaderas
 * @return Precisión como porcentaje
 */
float calcularPrecision(const Tensor& predicciones, const Tensor& etiquetas) {
    auto datos_pred = predicciones.matrix<float>();
    auto datos_etiquetas = etiquetas.matrix<float>();
    
    int correctos = 0;
    int total = predicciones.dim_size(0);
    
    for (int i = 0; i < total; ++i) {
        float clase_pred = (datos_pred(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        if (clase_pred == datos_etiquetas(i, 0)) {
            correctos++;
        }
    }
    
    return 100.0f * static_cast<float>(correctos) / static_cast<float>(total);
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Demo de Red Neuronal TensorFlow C++    " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Hiperparámetros
    const int NUM_MUESTRAS = 200;
    const int NUM_EPOCAS = 2000;
    const float TASA_APRENDIZAJE = 0.5f;
    const int TAMANO_OCULTA = 8;  // Número de neuronas en capa oculta
    const int TAMANO_ENTRADA = 2;   // Número de características de entrada
    const int TAMANO_SALIDA = 1;  // Clasificación binaria
    
    std::cout << "\n=== Arquitectura de la Red ===" << std::endl;
    std::cout << "Capa de entrada:  " << TAMANO_ENTRADA << " neuronas" << std::endl;
    std::cout << "Capa oculta: " << TAMANO_OCULTA << " neuronas (activación Sigmoide)" << std::endl;
    std::cout << "Capa de salida: " << TAMANO_SALIDA << " neurona (activación Sigmoide)" << std::endl;
    
    // Generar datos de entrenamiento
    std::vector<float> datos_x, datos_y;
    generarDatosClasificacion(NUM_MUESTRAS, datos_x, datos_y);
    
    Tensor tensor_x = crearTensorCaracteristicas(datos_x, NUM_MUESTRAS, TAMANO_ENTRADA);
    Tensor tensor_y = crearTensorEtiquetas(datos_y);
    
    // Contar distribución de clases
    int clase_0 = std::count(datos_y.begin(), datos_y.end(), 0.0f);
    int clase_1 = std::count(datos_y.begin(), datos_y.end(), 1.0f);
    
    std::cout << "\n=== Conjunto de Datos ===" << std::endl;
    std::cout << "Total de muestras: " << NUM_MUESTRAS << std::endl;
    std::cout << "Clase 0: " << clase_0 << " muestras" << std::endl;
    std::cout << "Clase 1: " << clase_1 << " muestras" << std::endl;
    
    // Crear ámbito de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Marcadores de posición de entrada
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_ENTRADA}));
    auto y_ph = Placeholder(root.WithOpName("Y"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_SALIDA}));
    
    // Inicializar pesos y sesgos usando inicialización Xavier
    // Pesos y sesgos de capa oculta
    auto w1_var = Variable(root.WithOpName("W1"), 
                           {TAMANO_ENTRADA, TAMANO_OCULTA}, DT_FLOAT);
    auto b1_var = Variable(root.WithOpName("b1"), 
                           {1, TAMANO_OCULTA}, DT_FLOAT);
    
    // Pesos y sesgos de capa de salida
    auto w2_var = Variable(root.WithOpName("W2"), 
                           {TAMANO_OCULTA, TAMANO_SALIDA}, DT_FLOAT);
    auto b2_var = Variable(root.WithOpName("b2"), 
                           {1, TAMANO_SALIDA}, DT_FLOAT);
    
    // Inicializar variables
    float escala_w1 = std::sqrt(2.0f / TAMANO_ENTRADA);
    float escala_w2 = std::sqrt(2.0f / TAMANO_OCULTA);
    
    auto w1_init = Assign(root, w1_var,
        Mul(root, RandomNormal(root, {TAMANO_ENTRADA, TAMANO_OCULTA}, DT_FLOAT),
            Const(root, escala_w1)));
    auto b1_init = Assign(root, b1_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_OCULTA}))));
    auto w2_init = Assign(root, w2_var,
        Mul(root, RandomNormal(root, {TAMANO_OCULTA, TAMANO_SALIDA}, DT_FLOAT),
            Const(root, escala_w2)));
    auto b2_init = Assign(root, b2_var,
        ZerosLike(root, Const(root, Tensor(DT_FLOAT, {1, TAMANO_SALIDA}))));
    
    // Propagación hacia adelante
    // Capa oculta: h = sigmoide(X * W1 + b1)
    auto z1 = Add(root, MatMul(root, x_ph, w1_var), b1_var);
    auto h = Sigmoid(root.WithOpName("oculta"), z1);
    
    // Capa de salida: y_pred = sigmoide(h * W2 + b2)
    auto z2 = Add(root, MatMul(root, h, w2_var), b2_var);
    auto y_pred = Sigmoid(root.WithOpName("salida"), z2);
    
    // Pérdida: Entropía Cruzada Binaria
    // pérdida = -media(y * log(y_pred + epsilon) + (1-y) * log(1 - y_pred + epsilon))
    auto epsilon = Const(root, 1e-7f);
    auto uno = Const(root, 1.0f);
    
    auto termino1 = Mul(root, y_ph, Log(root, Add(root, y_pred, epsilon)));
    auto termino2 = Mul(root, Sub(root, uno, y_ph), 
                     Log(root, Add(root, Sub(root, uno, y_pred), epsilon)));
    auto perdida = Neg(root, Mean(root.WithOpName("perdida"), 
                               Add(root, termino1, termino2), {0, 1}));
    
    // Retropropagación (cálculo manual de gradientes)
    // Gradientes de capa de salida
    auto d_salida = Sub(root, y_pred, y_ph); // d(pérdida)/d(z2)
    auto d_w2 = MatMul(root, h, d_salida, MatMul::TransposeA(true));
    auto d_b2 = Mean(root, d_salida, {0});
    
    // Gradientes de capa oculta
    auto d_oculta = Mul(root,
        MatMul(root, d_salida, w2_var, MatMul::TransposeB(true)),
        Mul(root, h, Sub(root, uno, h))); // derivada sigmoide: h * (1-h)
    auto d_w1 = MatMul(root, x_ph, d_oculta, MatMul::TransposeA(true));
    auto d_b1 = Mean(root, d_oculta, {0});
    
    // Actualizaciones por descenso de gradiente
    auto lr = Const(root, TASA_APRENDIZAJE);
    auto lr_escalada = Const(root, TASA_APRENDIZAJE / static_cast<float>(NUM_MUESTRAS));
    
    auto w2_update = AssignSub(root, w2_var, Mul(root, lr_escalada, d_w2));
    auto b2_update = AssignSub(root, b2_var, 
        Mul(root, lr, Reshape(root, d_b2, {1, TAMANO_SALIDA})));
    auto w1_update = AssignSub(root, w1_var, Mul(root, lr_escalada, d_w1));
    auto b1_update = AssignSub(root, b1_var,
        Mul(root, lr, Reshape(root, d_b1, {1, TAMANO_OCULTA})));
    
    // Crear sesión
    ClientSession session(root);
    
    // Inicializar todas las variables
    TF_CHECK_OK(session.Run({w1_init, b1_init, w2_init, b2_init}, nullptr));
    
    std::cout << "\n=== Entrenamiento ===" << std::endl;
    std::cout << "Tasa de aprendizaje: " << TASA_APRENDIZAJE << std::endl;
    std::cout << "Épocas: " << NUM_EPOCAS << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    // Bucle de entrenamiento
    std::vector<Tensor> salidas;
    for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
        // Paso hacia adelante y actualizaciones
        TF_CHECK_OK(session.Run(
            {{x_ph, tensor_x}, {y_ph, tensor_y}},
            {perdida, y_pred, w1_update, b1_update, w2_update, b2_update},
            &salidas));
        
        // Imprimir progreso
        if (epoca % 200 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            float precision = calcularPrecision(salidas[1], tensor_y);
            std::cout << "Época " << std::setw(4) << epoca
                      << " | Pérdida: " << std::setw(8) << perdida_actual
                      << " | Precisión: " << std::setw(6) << precision << "%" << std::endl;
        }
    }
    
    // Evaluación final
    std::cout << "\n=== Evaluación Final ===" << std::endl;
    
    TF_CHECK_OK(session.Run(
        {{x_ph, tensor_x}, {y_ph, tensor_y}},
        {perdida, y_pred},
        &salidas));
    
    float perdida_final = salidas[0].scalar<float>()();
    float precision_final = calcularPrecision(salidas[1], tensor_y);
    
    std::cout << "Pérdida Final: " << perdida_final << std::endl;
    std::cout << "Precisión Final: " << precision_final << "%" << std::endl;
    
    // Mostrar algunas predicciones
    std::cout << "\n=== Predicciones de Muestra ===" << std::endl;
    std::cout << std::setw(8) << "X1" << " | "
              << std::setw(8) << "X2" << " | "
              << std::setw(8) << "Real" << " | "
              << std::setw(8) << "Pred" << " | "
              << std::setw(8) << "Prob" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    auto datos_pred = salidas[1].matrix<float>();
    auto mat_x = tensor_x.matrix<float>();
    auto mat_y = tensor_y.matrix<float>();
    
    // Mostrar primeras 10 predicciones
    for (int i = 0; i < std::min(10, NUM_MUESTRAS); ++i) {
        float clase_pred = (datos_pred(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        std::cout << std::setw(8) << mat_x(i, 0) << " | "
                  << std::setw(8) << mat_x(i, 1) << " | "
                  << std::setw(8) << mat_y(i, 0) << " | "
                  << std::setw(8) << clase_pred << " | "
                  << std::setw(8) << datos_pred(i, 0) << std::endl;
    }
    
    // Probar con nuevos puntos de datos
    std::cout << "\n=== Predicciones con Nuevos Datos ===" << std::endl;
    std::vector<float> test_x = {
        1.0f, 1.0f,    // Debería ser clase 1 (+ * + > 0)
        -1.0f, -1.0f,  // Debería ser clase 1 (- * - > 0)
        1.0f, -1.0f,   // Debería ser clase 0 (+ * - < 0)
        -1.0f, 1.0f,   // Debería ser clase 0 (- * + < 0)
        0.5f, 0.5f,    // Debería ser clase 1
        -0.5f, 0.5f    // Debería ser clase 0
    };
    
    Tensor tensor_test = crearTensorCaracteristicas(test_x, 6, 2);
    
    std::vector<Tensor> salidas_test;
    TF_CHECK_OK(session.Run(
        {{x_ph, tensor_test}},
        {y_pred},
        &salidas_test));
    
    auto pred_test = salidas_test[0].matrix<float>();
    auto mat_test = tensor_test.matrix<float>();
    
    std::cout << std::setw(8) << "X1" << " | "
              << std::setw(8) << "X2" << " | "
              << std::setw(10) << "Esperado" << " | "
              << std::setw(8) << "Pred" << " | "
              << std::setw(8) << "Prob" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<float> esperado = {1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    for (int i = 0; i < 6; ++i) {
        float clase_pred = (pred_test(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        std::cout << std::setw(8) << mat_test(i, 0) << " | "
                  << std::setw(8) << mat_test(i, 1) << " | "
                  << std::setw(10) << esperado[i] << " | "
                  << std::setw(8) << clase_pred << " | "
                  << std::setw(8) << pred_test(i, 0) << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  ¡Demo de red neuronal completada!      " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
