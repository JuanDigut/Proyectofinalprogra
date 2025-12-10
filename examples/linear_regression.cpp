
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;
using namespace tensorflow::ops;

// Declaraciones de funciones
void generarDatosEntrenamiento(int num_muestras, float pendiente_real, float intercepto_real,
                               float nivel_ruido, std::vector<float>& datos_x, 
                               std::vector<float>& datos_y);

Tensor crearTensor(const std::vector<float>& datos);

int main() {
    
    // Parámetros verdaderos base
    const float PENDIENTE_REAL = 2.5f;
    const float INTERCEPTO_REAL = 1.0f;
    const int NUM_MUESTRAS = 100;
    const int NUM_EPOCAS = 1000;
    const float TASA_APRENDIZAJE = 0.01f;
    
    // Niveles de ruido a explorar
    std::vector<float> niveles_ruido = {0.0f, 0.1f, 0.5f, 1.0f, 2.0f};
    
    // Abrir archivo CSV para resultados
    std::ofstream csv_ruido("resultados_ruido.csv");
    csv_ruido << "nivel_ruido,pendiente_aprendida,intercepto_aprendido,error_pendiente,error_intercepto,mse,rmse\n";
    
    std::cout << "\n=== Parámetros de Entrenamiento ===" << std::endl;
    std::cout << "Pendiente real: " << PENDIENTE_REAL << std::endl;
    std::cout << "Intercepto real: " << INTERCEPTO_REAL << std::endl;
    std::cout << "Número de muestras: " << NUM_MUESTRAS << std::endl;
    std::cout << "Épocas: " << NUM_EPOCAS << std::endl;
    std::cout << "Tasa de aprendizaje: " << TASA_APRENDIZAJE << std::endl;
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nNivel Ruido | Pendiente Aprend | Error Pend | Intercepto Aprend | Error Intercep | MSE     | RMSE    \n";
    std::cout << "------------|-----------------|-----------|------------------|----------------|---------|--------\n";
    
    // Entrenar múltiples modelos con diferentes niveles de ruido
    for (float NIVEL_RUIDO : niveles_ruido) {
        
        std::vector<float> datos_x, datos_y;
        generarDatosEntrenamiento(NUM_MUESTRAS, PENDIENTE_REAL, INTERCEPTO_REAL, NIVEL_RUIDO,
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
        
        // Guardar en CSV
        csv_ruido << NIVEL_RUIDO << ","
                  << pendiente_aprendida << ","
                  << intercepto_aprendido << ","
                  << error_pendiente << ","
                  << error_intercepto << ","
                  << mse << ","
                  << rmse << "\n";
        
        // Imprimir en consola
        std::cout << std::setw(11) << NIVEL_RUIDO << " | "
                  << std::setw(16) << pendiente_aprendida << " | "
                  << std::setw(9) << error_pendiente << " | "
                  << std::setw(17) << intercepto_aprendido << " | "
                  << std::setw(15) << error_intercepto << " | "
                  << std::setw(7) << mse << " | "
                  << std::setw(7) << rmse << std::endl;
    }
    
    csv_ruido.close();
    std::cout << "\nResultados guardados en: resultados_ruido.csv" << std::endl;
    
    // Análisis de sensibilidad a la tasa de aprendizaje
    
    std::ofstream csv_lr("resultados_tasa_aprendizaje.csv");
    csv_lr << "tasa_aprendizaje,pendiente_aprendida,intercepto_aprendido,error_pendiente,error_intercepto,mse,rmse,epocas_convergencia\n";
    
    std::vector<float> tasas_aprendizaje = {0.001f, 0.005f, 0.01f, 0.05f, 0.1f, 0.2f, 0.3f};
    const float RUIDO_FIJO = 0.5f;
    const float UMBRAL_CONVERGENCIA = 0.001f;  // ← Umbral más estricto
    
    // Generar datos UNA SOLA VEZ para todas las tasas de aprendizaje
    std::vector<float> datos_x_comun, datos_y_comun;
    {
        std::mt19937 gen_lr(42);  // Semilla fija
        std::normal_distribution<float> ruido(0.0f, RUIDO_FIJO);
        std::uniform_real_distribution<float> dist_x(0.0f, 10.0f);
        
        datos_x_comun.resize(NUM_MUESTRAS);
        datos_y_comun.resize(NUM_MUESTRAS);
        
        for (int i = 0; i < NUM_MUESTRAS; ++i) {
            datos_x_comun[i] = dist_x(gen_lr);
            datos_y_comun[i] = PENDIENTE_REAL * datos_x_comun[i] + INTERCEPTO_REAL + ruido(gen_lr);
        }
    }
    
    Tensor tensor_x_comun = crearTensor(datos_x_comun);
    Tensor tensor_y_comun = crearTensor(datos_y_comun);
    
    std::cout << "\n=== Análisis de Sensibilidad a Tasa de Aprendizaje ===" << std::endl;
    std::cout << "Ruido fijo: " << RUIDO_FIJO << std::endl;
    std::cout << "Umbral convergencia: " << UMBRAL_CONVERGENCIA << std::endl;
    std::cout << "\nTasa Aprend | Pendiente | Error Pend | Intercepto | Error Int | MSE     | Épocas Conv\n";
    std::cout << "------------|-----------|------------|------------|-----------|---------|------------\n";
    
    for (float lr_actual : tasas_aprendizaje) {
        
        Scope root_lr = Scope::NewRootScope();
        
        auto placeholder_x_lr = Placeholder(root_lr.WithOpName("x"), DT_FLOAT,
                                            Placeholder::Shape({-1, 1}));
        auto placeholder_y_lr = Placeholder(root_lr.WithOpName("y"), DT_FLOAT,
                                            Placeholder::Shape({-1, 1}));
        
        auto w_init_lr = Variable(root_lr.WithOpName("w"), {1, 1}, DT_FLOAT);
        auto w_assign_lr = Assign(root_lr.WithOpName("w_assign"), w_init_lr, 
                                  Const(root_lr, {{0.5f}}));
        
        auto b_init_lr = Variable(root_lr.WithOpName("b"), {1, 1}, DT_FLOAT);
        auto b_assign_lr = Assign(root_lr.WithOpName("b_assign"), b_init_lr,
                                  Const(root_lr, {{0.5f}}));
        
        auto y_pred_lr = Add(root_lr.WithOpName("prediccion"),
                             MatMul(root_lr, placeholder_x_lr, w_init_lr),
                             b_init_lr);
        
        auto error_lr = Sub(root_lr, y_pred_lr, placeholder_y_lr);
        auto error_cuadrado_lr = Square(root_lr, error_lr);
        auto perdida_lr = Mean(root_lr.WithOpName("perdida"), error_cuadrado_lr, {0, 1});
        
        auto grad_w_lr = Mul(root_lr, 
                             Const(root_lr, 2.0f),
                             Mean(root_lr, Mul(root_lr, error_lr, placeholder_x_lr), {0}));
        auto grad_b_lr = Mul(root_lr,
                             Const(root_lr, 2.0f),
                             Mean(root_lr, error_lr, {0}));
        
        auto lr_const = Const(root_lr, lr_actual);
        auto w_update_lr = AssignSub(root_lr, w_init_lr, Mul(root_lr, lr_const, Reshape(root_lr, grad_w_lr, {1, 1})));
        auto b_update_lr = AssignSub(root_lr, b_init_lr, Mul(root_lr, lr_const, Reshape(root_lr, grad_b_lr, {1, 1})));
        
        ClientSession session_lr(root_lr);
        TF_CHECK_OK(session_lr.Run({w_assign_lr, b_assign_lr}, nullptr));
        
        std::vector<Tensor> salidas_lr;
        std::vector<Output> fetch_ops_lr;
        fetch_ops_lr.push_back(perdida_lr);
        fetch_ops_lr.push_back(w_update_lr);
        fetch_ops_lr.push_back(b_update_lr);
        
        int epocas_convergencia = NUM_EPOCAS;
        float perdida_hace_50 = 999999.0f;
        float perdida_actual = 0.0f;
        bool convergido = false;
        
        for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
            TF_CHECK_OK(session_lr.Run(
                {{placeholder_x_lr, tensor_x_comun}, {placeholder_y_lr, tensor_y_comun}},
                fetch_ops_lr,
                &salidas_lr));
            
            perdida_actual = salidas_lr[0].scalar<float>()();
            
            // Guardar primera pérdida para comparar
            if (epoca == 0) {
                perdida_hace_50 = perdida_actual;
            }
            
            // Evaluar convergencia cada 50 épocas
            if (!convergido && epoca > 0 && epoca % 50 == 0) {
                float diferencia = std::abs(perdida_hace_50 - perdida_actual);
                if (diferencia < 0.01f) {
                    epocas_convergencia = epoca;
                    convergido = true;
                }
                perdida_hace_50 = perdida_actual;
            }
        }
        
        std::vector<Tensor> params_finales_lr;
        TF_CHECK_OK(session_lr.Run({w_init_lr, b_init_lr}, &params_finales_lr));
        
        float pendiente_aprendida = params_finales_lr[0].matrix<float>()(0, 0);
        float intercepto_aprendido = params_finales_lr[1].matrix<float>()(0, 0);
        float error_pendiente = std::abs(pendiente_aprendida - PENDIENTE_REAL);
        float error_intercepto = std::abs(intercepto_aprendido - INTERCEPTO_REAL);
        
        std::vector<Tensor> mse_final_lr;
        TF_CHECK_OK(session_lr.Run(
            {{placeholder_x_lr, tensor_x_comun}, {placeholder_y_lr, tensor_y_comun}},
            {perdida_lr},
            &mse_final_lr
        ));
        
        float mse = mse_final_lr[0].scalar<float>()();
        float rmse = std::sqrt(mse);
        
        // Guardar en CSV
        csv_lr << lr_actual << ","
               << pendiente_aprendida << ","
               << intercepto_aprendido << ","
               << error_pendiente << ","
               << error_intercepto << ","
               << mse << ","
               << rmse << ","
               << epocas_convergencia << "\n";
        
        // Imprimir en consola
        std::cout << std::setw(11) << lr_actual << " | "
                  << std::setw(9) << pendiente_aprendida << " | "
                  << std::setw(10) << error_pendiente << " | "
                  << std::setw(10) << intercepto_aprendido << " | "
                  << std::setw(9) << error_intercepto << " | "
                  << std::setw(7) << mse << " | "
                  << std::setw(10) << epocas_convergencia << std::endl;
    }
    
    csv_lr.close();
    std::cout << "\nResultados guardados en: resultados_tasa_aprendizaje.csv" << std::endl;
  // análisis de datos
    std::ofstream csv_progresion("progresion_perdida.csv");
    csv_progresion << "epoca,perdida\n";
    
    // Usar los mismos datos comunes
    Scope root = Scope::NewRootScope();
    
    auto placeholder_x = Placeholder(root.WithOpName("x"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    auto placeholder_y = Placeholder(root.WithOpName("y"), DT_FLOAT,
                                     Placeholder::Shape({-1, 1}));
    
    auto w_init = Variable(root.WithOpName("w"), {1, 1}, DT_FLOAT);
    auto w_assign = Assign(root.WithOpName("w_assign"), w_init, 
                          Const(root, {{0.5f}}));  // ← Mismo valor inicial
    
    auto b_init = Variable(root.WithOpName("b"), {1, 1}, DT_FLOAT);
    auto b_assign = Assign(root.WithOpName("b_assign"), b_init,
                          Const(root, {{0.5f}}));  // ← Mismo valor inicial

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
    
    std::cout << "\nProgresión de pérdida durante entrenamiento (ruido = 0.5):\n";
    std::cout << "Época  | Pérdida\n";
    std::cout << "-------|--------\n";
    
    for (int epoca = 0; epoca < NUM_EPOCAS; ++epoca) {
        TF_CHECK_OK(session.Run(
            {{placeholder_x, tensor_x_comun}, {placeholder_y, tensor_y_comun}},
            fetch_ops,
            &salidas));
        
        float perdida_actual = salidas[0].scalar<float>()();
        
        // Guardar cada 50 épocas en CSV (excluyendo época 0)
        if ((epoca > 0 && epoca % 50 == 0) || epoca == NUM_EPOCAS - 1) {
            csv_progresion << epoca << "," << perdida_actual << "\n";
        }
        
        // Imprimir en consola cada 200 épocas (excluyendo época 0)
        if ((epoca > 0 && epoca % 200 == 0) || epoca == NUM_EPOCAS - 1) {
            std::cout << std::setw(5) << epoca << " | " << std::setw(7) << perdida_actual << std::endl;
        }
    }
    
    csv_progresion.close();
    std::cout << "\nProgresión guardada en: progresion_perdida.csv" << std::endl;
    
    return 0;
}

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
