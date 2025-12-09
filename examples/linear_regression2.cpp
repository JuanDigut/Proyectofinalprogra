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
    std::cout << "  Análisis de Sensibilidad al Ruido" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Parámetros de verdad base (intentaremos aprender estos)
    const float PENDIENTE_REAL = 2.5f;
    const float INTERCEPTO_REAL = 1.0f;
    const int NUM_MUESTRAS = 100;
    const int NUM_EPOCAS = 1000;  // Aumentado para convergencia
    const float TASA_APRENDIZAJE = 0.01f;
    
    // Niveles de ruido a explorar
    std::vector<float> niveles_ruido = {0.0f, 0.1f, 0.5f, 1.0f, 2.0f};
    
    std::cout << "\n=== Parámetros de Entrenamiento ===" << std::endl;
    std::cout << "Pendiente real: " << PENDIENTE_REAL << std::endl;
    std::cout << "Intercepto real: " << INTERCEPTO_REAL << std::endl;
    std::cout << "Número de muestras: " << NUM_MUESTRAS << std::endl;
    std::cout << "Épocas: " << NUM_EPOCAS << std::endl;
    std::cout << "Tasa de aprendizaje: " << TASA_APRENDIZAJE << std::endl;
    
    std::cout << "\n=== Explorando Sensibilidad al Ruido ===" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nNivel Ruido | Pendiente Aprend | Error Pend | Intercepto Aprend | Error Intercep | MSE     | RMSE    \n";
    std::cout << "------------|-----------------|-----------|------------------|----------------|---------|--------\n";
    
    // Entrenar múltiples modelos con diferentes niveles de ruido
    for (float NIVEL_RUIDO : niveles_ruido) {
        
        // Generar datos de entrenamiento
        std::vector<float> datos_x, datos_y;
        generarDatosEntrenamiento(NUM_MUESTRAS, PENDIENTE_REAL, INTERCEPTO_REAL, NIVEL_RUIDO,
                                  datos_x, datos_y);
        
        // Crear tensores a partir de los datos
        Tensor tensor_x = crearTensor(datos_x);
        Tensor tensor_y = crearTensor(datos_y);
        
        // Crear ámbito de TensorFlow
        Scope root = Scope::NewRootScope();
        
        // Crear marcadores de posición para datos de entrada
        auto placeholder_x = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                         Placeholder::Shape({-1, 1}));
        auto placeholder_y = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                         Placeholder::Shape({-1, 1}));
        
        // Crear variables entrenables (inicializadas con valores aleatorios)
        auto w_init = Variable(root.WithOpName("w"), {1, 1}, DT_FLOAT);
        auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                              RandomNormal(root, {1, 1}, DT_FLOAT));
        
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
        auto grad_w = Mul(root, 
                         Const(root, 2.0f),
                         Mean(root, Mul(root, error, placeholder_x), {0}));
        auto grad_b = Mul(root,
                         Const(root, 2.0f),
                         Mean(root, error, {0}));
        
        // Actualización por descenso de gradiente
        auto lr = Const(root, TASA_APRENDIZAJE);
        auto w_update = AssignSub(root, w_init, Mul(root, lr, Reshape(root, grad_w, {1, 1})));
        auto b_update = AssignSub(root, b_init, Mul(root, lr, Reshape(root, grad_b, {1, 1})));
        
        // Crear sesión
        ClientSession session(root);
        
        // Inicializar variables
        TF_CHECK_OK(session.Run({w_assign, b_assign}, nullptr));
        
        // Bucle de entrenamiento (silencioso)
        std::vector<Tensor> salidas;
        std::vector<Output> fetch_ops;
        fetch_ops.push_back(perdida);
        fetch_ops.push_back(w_update);
        fetch_ops.push_back(b_update);
        
        for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
            TF_CHECK_OK(session.Run(
                {{placeholder_x, tensor_x}, {placeholder_y, tensor_y}},
                fetch_ops,
                &salidas));
        }
        
        // Obtener parámetros aprendidos finales
        std::vector<Tensor> params_finales;
        TF_CHECK_OK(session.Run({w_init, b_init}, &params_finales));
        
        float pendiente_aprendida = params_finales[0].matrix<float>()(0, 0);
        float intercepto_aprendido = params_finales[1].matrix<float>()(0, 0);
        
        // Calcular errores
        float error_pendiente = std::abs(pendiente_aprendida - PENDIENTE_REAL);
        float error_intercepto = std::abs(intercepto_aprendido - INTERCEPTO_REAL);
        
        // Calcular MSE/RMSE final
        std::vector<Tensor> mse_final;
        TF_CHECK_OK(session.Run(
            {{placeholder_x, tensor_x}, {placeholder_y, tensor_y}},
            {perdida},
            &mse_final
        ));
        
        float mse = mse_final[0].scalar<float>()();
        float rmse = std::sqrt(mse);
        
        // Imprimir fila de resultados
        std::cout << std::setw(11) << NIVEL_RUIDO << " | "
                  << std::setw(16) << pendiente_aprendida << " | "
                  << std::setw(9) << error_pendiente << " | "
                  << std::setw(17) << intercepto_aprendido << " | "
                  << std::setw(15) << error_intercepto << " | "
                  << std::setw(7) << mse << " | "
                  << std::setw(7) << rmse << std::endl;
    }
    
    // Análisis detallado para nivel de ruido intermedio (0.5)
    std::cout << "\n=== Análisis Detallado con Ruido = 0.5 ===" << std::endl;
    
    std::vector<float> datos_x, datos_y;
    generarDatosEntrenamiento(NUM_MUESTRAS, PENDIENTE_REAL, INTERCEPTO_REAL, 0.5f,
                              datos_x, datos_y);
    
    Tensor tensor_x = crearTensor(datos_x);
    Tensor tensor_y = crearTensor(datos_y);
    
    Scope root = Scope::NewRootScope();
    
    auto placeholder_x = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    auto placeholder_y = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    
    auto w_init = Variable(root.WithOpName("w"), {1, 1}, DT_FLOAT);
    auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    auto b_init = Variable(root.WithOpName("b"), {1, 1}, DT_FLOAT);
    auto b_assign = Assign(root.WithOpName("b_assign"), b_init,
                          RandomNormal(root, {1, 1}, DT_FLOAT));
    
    auto y_pred = Add(root.WithOpName("prediccion"),
                     MatMul(root, placeholder_x, w_init),
                     b_init);
    
    auto error = Sub(root, y_pred, placeholder_y);
    auto error_cuadrado = Square(root, error);
    auto perdida = Mean(root.WithOpName("perdida"), error_cuadrado, {0, 1});
    
    auto grad_w = Mul(root, 
                     Const(root, 2.0f),
                     Mean(root, Mul(root, error, placeholder_x), {0}));
    auto grad_b = Mul(root,
                     Const(root, 2.0f),
                     Mean(root, error, {0}));
    
    auto lr = Const(root, TASA_APRENDIZAJE);
    auto w_update = AssignSub(root, w_init, Mul(root, lr, Reshape(root, grad_w, {1, 1})));
    auto b_update = AssignSub(root, b_init, Mul(root, lr, Reshape(root, grad_b, {1, 1})));
    
    ClientSession session(root);
    TF_CHECK_OK(session.Run({w_assign, b_assign}, nullptr));
    
    std::vector<Tensor> salidas;
    std::vector<Output> fetch_ops;
    fetch_ops.push_back(perdida);
    fetch_ops.push_back(w_update);
    fetch_ops.push_back(b_update);
    
    std::cout << "\nProgresión de pérdida durante entrenamiento:\n";
    std::cout << "Época  | Pérdida\n";
    std::cout << "-------|--------\n";
    
    for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
        TF_CHECK_OK(session.Run(
            {{placeholder_x, tensor_x}, {placeholder_y, tensor_y}},
            fetch_ops,
            &salidas));
        
        if (epoca % 200 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            std::cout << std::setw(5) << epoca << " | " << std::setw(7) << perdida_actual << std::endl;
        }
    }
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  ¡Análisis de Sensibilidad Completado!" << std::endl;
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