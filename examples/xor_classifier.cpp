/**
 * @file xor_classifier.cpp
 * @brief Ejemplo 5: Clasificador XOR usando Red Neuronal con TensorFlow C++
 * 
 * Este ejemplo demuestra cómo construir y entrenar una red neuronal
 * para resolver el clásico problema XOR usando la API de TensorFlow C++.
 * 
 * El problema XOR es una tarea de clasificación no linealmente separable que
 * requiere al menos una capa oculta para resolverse. Este ejemplo implementa
 * una red multicapa con arquitectura: 2 -> 8 -> 4 -> 1
 * 
 * Conceptos clave demostrados:
 * - Arquitectura de red neuronal multicapa
 * - Funciones de activación (ReLU para capas ocultas, Sigmoide para salida)
 * - El problema XOR y por qué necesita capas ocultas
 * - Pérdida de Entropía Cruzada Binaria
 * - Propagación hacia adelante y hacia atrás
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
 * @brief Crea un tensor a partir de datos de entrada XOR
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
 * @brief Crea un tensor a partir de datos de etiquetas
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
    std::cout << "  Demo de Clasificador XOR TensorFlow C++" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Tabla de verdad XOR
    // Entrada1 | Entrada2 | Salida
    //    0     |    0     |   0
    //    0     |    1     |   1
    //    1     |    0     |   1
    //    1     |    1     |   0
    
    std::cout << "\n=== Tabla de Verdad XOR ===" << std::endl;
    std::cout << "Entrada1 | Entrada2 | Salida" << std::endl;
    std::cout << "   0     |    0     |   0" << std::endl;
    std::cout << "   0     |    1     |   1" << std::endl;
    std::cout << "   1     |    0     |   1" << std::endl;
    std::cout << "   1     |    1     |   0" << std::endl;
    
    // Datos XOR
    std::vector<float> datos_x = {
        0.0f, 0.0f,  // -> 0
        0.0f, 1.0f,  // -> 1
        1.0f, 0.0f,  // -> 1
        1.0f, 1.0f   // -> 0
    };
    
    std::vector<float> datos_y = {0.0f, 1.0f, 1.0f, 0.0f};
    
    // Hiperparámetros
    const int NUM_MUESTRAS = 4;
    const int NUM_EPOCAS = 5000;
    const float TASA_APRENDIZAJE = 0.5f;
    const int TAMANO_ENTRADA = 2;
    const int TAMANO_OCULTA1 = 8;   // Primera capa oculta
    const int TAMANO_OCULTA2 = 4;   // Segunda capa oculta
    const int TAMANO_SALIDA = 1;
    
    std::cout << "\n=== Arquitectura de la Red ===" << std::endl;
    std::cout << "Capa de entrada:    " << TAMANO_ENTRADA << " neuronas" << std::endl;
    std::cout << "Capa oculta 1: " << TAMANO_OCULTA1 << " neuronas (activación ReLU)" << std::endl;
    std::cout << "Capa oculta 2: " << TAMANO_OCULTA2 << " neuronas (activación ReLU)" << std::endl;
    std::cout << "Capa de salida:   " << TAMANO_SALIDA << " neurona (activación Sigmoide)" << std::endl;
    
    // Crear tensores
    Tensor tensor_x = crearTensorCaracteristicas(datos_x, NUM_MUESTRAS, TAMANO_ENTRADA);
    Tensor tensor_y = crearTensorEtiquetas(datos_y);
    
    // Crear ámbito de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Marcadores de posición de entrada
    auto x_ph = Placeholder(root.WithOpName("X"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_ENTRADA}));
    auto y_ph = Placeholder(root.WithOpName("Y"), DT_FLOAT,
                            Placeholder::Shape({-1, TAMANO_SALIDA}));
    
    // Capa 1: Entrada -> Oculta1 (2 -> 8)
    auto w1_var = Variable(root.WithOpName("W1"), 
                           {TAMANO_ENTRADA, TAMANO_OCULTA1}, DT_FLOAT);
    auto b1_var = Variable(root.WithOpName("b1"), 
                           {1, TAMANO_OCULTA1}, DT_FLOAT);
    
    // Capa 2: Oculta1 -> Oculta2 (8 -> 4)
    auto w2_var = Variable(root.WithOpName("W2"), 
                           {TAMANO_OCULTA1, TAMANO_OCULTA2}, DT_FLOAT);
    auto b2_var = Variable(root.WithOpName("b2"), 
                           {1, TAMANO_OCULTA2}, DT_FLOAT);
    
    // Capa 3: Oculta2 -> Salida (4 -> 1)
    auto w3_var = Variable(root.WithOpName("W3"), 
                           {TAMANO_OCULTA2, TAMANO_SALIDA}, DT_FLOAT);
    auto b3_var = Variable(root.WithOpName("b3"), 
                           {1, TAMANO_SALIDA}, DT_FLOAT);
    
    // Inicializar pesos con inicialización He para ReLU
    float escala_w1 = std::sqrt(2.0f / TAMANO_ENTRADA);
    float escala_w2 = std::sqrt(2.0f / TAMANO_OCULTA1);
    float escala_w3 = std::sqrt(2.0f / TAMANO_OCULTA2);
    
    auto w1_init = Assign(root, w1_var,
        Mul(root, RandomNormal(root, {TAMANO_ENTRADA, TAMANO_OCULTA1}, DT_FLOAT),
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
    // Capa 1: h1 = ReLU(X * W1 + b1)
    auto z1 = Add(root, MatMul(root, x_ph, w1_var), b1_var);
    auto h1 = Relu(root.WithOpName("oculta1"), z1);
    
    // Capa 2: h2 = ReLU(h1 * W2 + b2)
    auto z2 = Add(root, MatMul(root, h1, w2_var), b2_var);
    auto h2 = Relu(root.WithOpName("oculta2"), z2);
    
    // Capa de salida: y_pred = Sigmoide(h2 * W3 + b3)
    auto z3 = Add(root, MatMul(root, h2, w3_var), b3_var);
    auto y_pred = Sigmoid(root.WithOpName("salida"), z3);
    
    // Pérdida: Entropía Cruzada Binaria
    // pérdida = -media(y * log(y_pred + epsilon) + (1-y) * log(1 - y_pred + epsilon))
    auto epsilon = Const(root, 1e-7f);
    auto uno = Const(root, 1.0f);
    
    auto termino1 = Mul(root, y_ph, Log(root, Add(root, y_pred, epsilon)));
    auto termino2 = Mul(root, Sub(root, uno, y_ph), 
                     Log(root, Add(root, Sub(root, uno, y_pred), epsilon)));
    auto perdida = Neg(root, Mean(root.WithOpName("perdida"), 
                               Add(root, termino1, termino2), {0, 1}));
    
    // Retropropagación
    // Gradientes de capa de salida
    auto d_salida = Sub(root, y_pred, y_ph);  // d(pérdida)/d(z3)
    auto d_w3 = MatMul(root, h2, d_salida, MatMul::TransposeA(true));
    auto d_b3 = Mean(root, d_salida, {0});
    
    // Gradientes de capa oculta 2 (derivada de ReLU: 1 si z > 0, sino 0)
    auto grad_relu2 = Cast(root, Greater(root, z2, Const(root, 0.0f)), DT_FLOAT);
    auto d_oculta2 = Mul(root,
        MatMul(root, d_salida, w3_var, MatMul::TransposeB(true)),
        grad_relu2);
    auto d_w2 = MatMul(root, h1, d_oculta2, MatMul::TransposeA(true));
    auto d_b2 = Mean(root, d_oculta2, {0});
    
    // Gradientes de capa oculta 1
    auto grad_relu1 = Cast(root, Greater(root, z1, Const(root, 0.0f)), DT_FLOAT);
    auto d_oculta1 = Mul(root,
        MatMul(root, d_oculta2, w2_var, MatMul::TransposeB(true)),
        grad_relu1);
    auto d_w1 = MatMul(root, x_ph, d_oculta1, MatMul::TransposeA(true));
    auto d_b1 = Mean(root, d_oculta1, {0});
    
    // Actualizaciones por descenso de gradiente
    auto lr = Const(root, TASA_APRENDIZAJE);
    auto lr_escalada = Const(root, TASA_APRENDIZAJE / static_cast<float>(NUM_MUESTRAS));
    
    auto w3_update = AssignSub(root, w3_var, Mul(root, lr_escalada, d_w3));
    auto b3_update = AssignSub(root, b3_var, 
        Mul(root, lr, Reshape(root, d_b3, {1, TAMANO_SALIDA})));
    auto w2_update = AssignSub(root, w2_var, Mul(root, lr_escalada, d_w2));
    auto b2_update = AssignSub(root, b2_var,
        Mul(root, lr, Reshape(root, d_b2, {1, TAMANO_OCULTA2})));
    auto w1_update = AssignSub(root, w1_var, Mul(root, lr_escalada, d_w1));
    auto b1_update = AssignSub(root, b1_var,
        Mul(root, lr, Reshape(root, d_b1, {1, TAMANO_OCULTA1})));
    
    // Crear sesión
    ClientSession session(root);
    
    // Inicializar todas las variables
    TF_CHECK_OK(session.Run({w1_init, b1_init, w2_init, b2_init, w3_init, b3_init}, nullptr));
    
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
            {perdida, y_pred, w1_update, b1_update, w2_update, b2_update, w3_update, b3_update},
            &salidas));
        
        // Imprimir progreso
        if (epoca % 500 == 0 || epoca == NUM_EPOCAS - 1) {
            float perdida_actual = salidas[0].scalar<float>()();
            float precision = calcularPrecision(salidas[1], tensor_y);
            std::cout << "Época " << std::setw(4) << epoca
                      << " | Pérdida: " << std::setw(8) << perdida_actual
                      << " | Precisión: " << std::setw(6) << precision << "%" << std::endl;
        }
    }
    
    // Evaluación final
    std::cout << "\n=== Resultados Finales ===" << std::endl;
    
    TF_CHECK_OK(session.Run(
        {{x_ph, tensor_x}, {y_ph, tensor_y}},
        {perdida, y_pred},
        &salidas));
    
    float perdida_final = salidas[0].scalar<float>()();
    float precision_final = calcularPrecision(salidas[1], tensor_y);
    
    std::cout << "Pérdida Final: " << perdida_final << std::endl;
    std::cout << "Precisión Final: " << precision_final << "%" << std::endl;
    
    // Mostrar predicciones para la tabla de verdad XOR
    std::cout << "\n=== Predicciones XOR ===" << std::endl;
    std::cout << "Entrada1 | Entrada2 | Esperado | Predicho | Probabilidad" << std::endl;
    std::cout << std::string(55, '-') << std::endl;
    
    auto datos_pred = salidas[1].matrix<float>();
    auto mat_x = tensor_x.matrix<float>();
    auto mat_y = tensor_y.matrix<float>();
    
    for (int i = 0; i < NUM_MUESTRAS; ++i) {
        float clase_pred = (datos_pred(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        std::cout << std::setw(8) << static_cast<int>(mat_x(i, 0)) << " | "
                  << std::setw(8) << static_cast<int>(mat_x(i, 1)) << " | "
                  << std::setw(8) << static_cast<int>(mat_y(i, 0)) << " | "
                  << std::setw(8) << static_cast<int>(clase_pred) << " | "
                  << std::setw(12) << datos_pred(i, 0) << std::endl;
    }
    
    // Verificar que todas las predicciones sean correctas
    bool todas_correctas = true;
    for (int i = 0; i < NUM_MUESTRAS; ++i) {
        float clase_pred = (datos_pred(i, 0) >= 0.5f) ? 1.0f : 0.0f;
        if (clase_pred != mat_y(i, 0)) {
            todas_correctas = false;
            break;
        }
    }
    
    std::cout << "\n=== Resumen ===" << std::endl;
    if (todas_correctas) {
        std::cout << "¡ÉXITO: La red aprendió la función XOR perfectamente!" << std::endl;
    } else {
        std::cout << "La red aún está aprendiendo... intenta con más épocas o ajusta los hiperparámetros." << std::endl;
    }
    
    std::cout << "\nEsto demuestra por qué XOR requiere capas ocultas:" << std::endl;
    std::cout << "- XOR no es linealmente separable" << std::endl;
    std::cout << "- Un perceptrón simple no puede resolver XOR" << std::endl;
    std::cout << "- Las capas ocultas permiten aprender límites de decisión no lineales" << std::endl;
    
    std::cout << "\n==========================================" << std::endl;
    std::cout << "  ¡Demo del clasificador XOR completada! " << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return 0;
}
