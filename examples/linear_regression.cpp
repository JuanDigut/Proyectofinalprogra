// linear_regression.cpp
// Ejemplo 2: Regresión Lineal usando TensorFlow C++
// 
// Este ejemplo demuestra cómo implementar un modelo simple de regresión
// lineal usando la API de TensorFlow C++. El modelo aprende a ajustar una línea
// y = mx + b a un conjunto de puntos de datos de entrenamiento.
// 
// Conceptos clave demostrados:
// - Creación de variables (parámetros entrenables)
// - Definición de una función de pérdida (Error Cuadrático Medio)
// - Cálculo de gradientes
// - Implementación de optimización por descenso de gradiente
// - Implementación del bucle de entrenamiento

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

// Declaraciones de funciones

// Genera datos de entrenamiento sintéticos para regresión lineal
// Genera puntos de datos siguiendo y = pendiente_real * x + intercepto_real + ruido
// num_muestras: Número de puntos de datos a generar
// pendiente_real: La pendiente real de la línea
// intercepto_real: El intercepto real en y
// nivel_ruido: Desviación estándar del ruido Gaussiano
// datos_x: Vector de salida para valores x
// datos_y: Vector de salida para valores y
void generarDatosEntrenamiento(int num_muestras, float pendiente_real, float intercepto_real,
                               float nivel_ruido, std::vector<float>& datos_x, 
                               std::vector<float>& datos_y);

// Crea un tensor a partir de un vector de flotantes
// datos: El vector de entrada
// Retorna: Un tensor de TensorFlow conteniendo los datos
Tensor crearTensor(const std::vector<float>& datos);

// Demostración principal de regresión lineal
int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Demo de Regresión Lineal TensorFlow C++" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Parámetros de verdad base (intentaremos aprender estos)
    const float PENDIENTE_REAL = 2.5f;
    const float INTERCEPTO_REAL = 1.0f;
    const float NIVEL_RUIDO = 0.5f;
    const int NUM_MUESTRAS = 100;
    const int NUM_EPOCAS = 1000;
    const float TASA_APRENDIZAJE = 0.01f;
    
    std::cout << "\n=== Generación de Datos ===" << std::endl;
    std::cout << "Pendiente real: " << PENDIENTE_REAL << std::endl;
    std::cout << "Intercepto real: " << INTERCEPTO_REAL << std::endl;
    std::cout << "Nivel de ruido: " << NIVEL_RUIDO << std::endl;
    std::cout << "Número de muestras: " << NUM_MUESTRAS << std::endl;
    
    // Generar datos de entrenamiento
    std::vector<float> datos_x, datos_y;
    generarDatosEntrenamiento(NUM_MUESTRAS, PENDIENTE_REAL, INTERCEPTO_REAL, NIVEL_RUIDO,
                              datos_x, datos_y);
    
    // Crear tensores a partir de los datos
    Tensor tensor_x = crearTensor(datos_x);
    Tensor tensor_y = crearTensor(datos_y);
    
    std::cout << "Se generaron " << NUM_MUESTRAS << " muestras de entrenamiento." << std::endl;
    
    // Crear ámbito de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Crear marcadores de posición para datos de entrada
    auto placeholder_x = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    auto placeholder_y = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    
    // Crear variables entrenables (inicializadas con valores aleatorios)
    // Peso (pendiente)
    auto w_init = Variable(root.WithOpName("w"), {1, 1}, DT_FLOAT);
    auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    // Sesgo (intercepto)
    auto b_init = Variable(root.WithOpName("b"), {1, 1}, DT_FLOAT);
    auto b_assign = Assign(root.WithOpName("b_assign"), b_init,
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    // Modelo: y_pred = x * w + b
    auto y_pred = Add(root.WithOpName("prediccion"),
                     MatMul(root, placeholder_x, w_init),
                     b_init);
    
    // Función de pérdida: Error Cuadrático Medio = media((y_pred - y)^2)
    auto error = Sub(root, y_pred, placeholder_y);
    auto error_cuadrado = Square(root, error);
    auto perdida = Mean(root.WithOpName("perdida"), error_cuadrado, {0, 1});
    
    // Calcular gradientes manualmente
    // d(perdida)/dw = 2 * media(error * x)
    // d(perdida)/db = 2 * media(error)
    auto grad_w = Mul(root, 
                     Const(root, 2.0f),
                     Mean(root, Mul(root, error, placeholder_x), {0}));
    auto grad_b = Mul(root,
                     Const(root, 2.0f),
                     Mean(root, error, {0}));
    
    // Actualización por descenso de gradiente
    auto lr = Const(root, TASA_APRENDIZAJE);
    auto w_update = AssignSub(root, w_init, Mul(root, lr, grad_w));
    auto b_update = AssignSub(root, b_init, Mul(root, lr, grad_b));
    
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
            {{placeholder_x, tensor_x}, {placeholder_y, tensor_y}},
            {perdida, w_update, b_update},
            &salidas));
        
        // Imprimir progreso cada 100 épocas
        if (epoca % 100 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            std::cout << "Época " << std::setw(4) << epoca 
                      << " | Pérdida: " << perdida_actual << std::endl;
        }
    }
    
    // Obtener parámetros aprendidos finales
    std::vector<Tensor> params_finales;
    TF_CHECK_OK(session.Run({w_init, b_init}, &params_finales));
    
    float pendiente_aprendida = params_finales[0].matrix<float>()(0, 0);
    float intercepto_aprendido = params_finales[1].matrix<float>()(0, 0);
    
    std::cout << "\n=== Resultados ===" << std::endl;
    std::cout << "Pendiente aprendida:     " << pendiente_aprendida 
              << " (real: " << PENDIENTE_REAL << ")" << std::endl;
    std::cout << "Intercepto aprendido: " << intercepto_aprendido 
              << " (real: " << INTERCEPTO_REAL << ")" << std::endl;
    
    // Calcular error
    float error_pendiente = std::abs(pendiente_aprendida - PENDIENTE_REAL);
    float error_intercepto = std::abs(intercepto_aprendido - INTERCEPTO_REAL);
    
    std::cout << "\nError en pendiente:     " << error_pendiente << std::endl;
    std::cout << "Error en intercepto: " << error_intercepto << std::endl;
    
    // Hacer predicciones con nuevos datos
    std::cout << "\n=== Predicciones ===" << std::endl;
    std::cout << "Probando con nuevos valores de x:" << std::endl;
    
    std::vector<float> test_x = {0.0f, 2.5f, 5.0f, 7.5f, 10.0f};
    Tensor tensor_test = crearTensor(test_x);
    
    std::vector<Tensor> predicciones;
    TF_CHECK_OK(session.Run(
        {{placeholder_x, tensor_test}},
        {y_pred},
        &predicciones));
    
    auto datos_pred = predicciones[0].matrix<float>();
    std::cout << std::setw(8) << "x" << " | " 
              << std::setw(12) << "Predicho" << " | "
              << std::setw(12) << "Real" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (size_t i = 0; i < test_x.size(); ++i) {
        float y_real = PENDIENTE_REAL * test_x[i] + INTERCEPTO_REAL;
        std::cout << std::setw(8) << test_x[i] << " | "
                  << std::setw(12) << datos_pred(i, 0) << " | "
                  << std::setw(12) << y_real << std::endl;
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  ¡Regresión lineal completada exitosamente!" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}

// Definiciones de funciones

// Genera datos de entrenamiento sintéticos para regresión lineal
// Genera puntos de datos siguiendo y = pendiente_real * x + intercepto_real + ruido
// num_muestras: Número de puntos de datos a generar
// pendiente_real: La pendiente real de la línea
// intercepto_real: El intercepto real en y
// nivel_ruido: Desviación estándar del ruido Gaussiano
// datos_x: Vector de salida para valores x
// datos_y: Vector de salida para valores y
void generarDatosEntrenamiento(int num_muestras, float pendiente_real, float intercepto_real,
                               float nivel_ruido, std::vector<float>& datos_x, 
                               std::vector<float>& datos_y) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> ruido(0.0f, nivel_ruido);
    std::uniform_real_distribution<float> dist_x(0.0f, 10.0f);
    
    datos_x.resize(num_muestras);
    datos_y.resize(num_muestras);
    
    for (int i = 0; i < num_muestras; ++i) {
        datos_x[i] = dist_x(gen);
        datos_y[i] = pendiente_real * datos_x[i] + intercepto_real + ruido(gen);
    }
}

// Crea un tensor a partir de un vector de flotantes
// datos: El vector de entrada
// Retorna: Un tensor de TensorFlow conteniendo los datos
Tensor crearTensor(const std::vector<float>& datos) {
    Tensor tensor(DT_FLOAT, TensorShape({static_cast<int64_t>(datos.size()), 1}));
    auto mapa_tensor = tensor.matrix<float>();
    for (size_t i = 0; i < datos.size(); ++i) {
        mapa_tensor(i, 0) = datos[i];
    }
    return tensor;
}
