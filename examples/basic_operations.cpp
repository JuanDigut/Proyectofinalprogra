// basic_operations.cpp
// Ejemplo 1: Operaciones Básicas de TensorFlow C++
// 
// Este ejemplo demuestra operaciones fundamentales de TensorFlow C++ incluyendo:
// - Creación e inicialización de una sesión de TensorFlow
// - Creación de tensores (constantes)
// - Realización de operaciones matemáticas básicas
// - Trabajo con formas y tipos de datos de tensores
// 
// La API de TensorFlow C++ proporciona acceso de bajo nivel a la funcionalidad
// del grafo computacional de TensorFlow, permitiendo la ejecución eficiente de
// tareas de aprendizaje automático y computación numérica.

#include <iostream>
#include <vector>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/core/framework/tensor.h>

using namespace tensorflow;
using namespace tensorflow::ops;

// Declaraciones de funciones

// Demuestra la creación y manipulación de tensores escalares
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarOperacionesEscalares(Scope& root, ClientSession& session);

// Demuestra operaciones de tensores vectoriales y matriciales
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarOperacionesVectoriales(Scope& root, ClientSession& session);

// Demuestra operaciones matriciales incluyendo multiplicación
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarOperacionesMatriciales(Scope& root, ClientSession& session);

// Demuestra manipulación de formas de tensores
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarFormasTensores(Scope& root, ClientSession& session);

// Demuestra funciones matemáticas (sin, cos, exp, log)
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarFuncionesMatematicas(Scope& root, ClientSession& session);

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "  Demo de Operaciones Básicas TensorFlow C++" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    // Crear un ámbito raíz para las operaciones de TensorFlow
    Scope root = Scope::NewRootScope();
    
    // Verificar si la creación del ámbito fue exitosa
    if (!root.ok()) {
        std::cerr << "Error al crear el ámbito de TensorFlow: " << root.status().ToString() << std::endl;
        return 1;
    }
    
    // Crear una sesión de cliente para ejecutar operaciones
    ClientSession session(root);
    
    try {
        // Ejecutar todas las demostraciones
        demostrarOperacionesEscalares(root, session);
        demostrarOperacionesVectoriales(root, session);
        demostrarOperacionesMatriciales(root, session);
        demostrarFormasTensores(root, session);
        demostrarFuncionesMatematicas(root, session);
        
        std::cout << "\n==========================================" << std::endl;
        std::cout << "  ¡Todas las demostraciones completadas exitosamente!" << std::endl;
        std::cout << "==========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// Definiciones de funciones

// Demuestra la creación y manipulación de tensores escalares
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarOperacionesEscalares(Scope& root, ClientSession& session) {
    std::cout << "\n=== Operaciones Escalares ===" << std::endl;
    
    // Crear constantes escalares
    auto a = Const(root, 5.0f);
    auto b = Const(root, 3.0f);
    
    // Realizar operaciones aritméticas
    auto suma = Add(root, a, b);
    auto resta = Sub(root, a, b);
    auto producto = Mul(root, a, b);
    auto cociente = Div(root, a, b);
    
    // Ejecutar las operaciones
    std::vector<Tensor> salidas;
    
    TF_CHECK_OK(session.Run({suma, resta, producto, cociente}, &salidas));
    
    std::cout << "a = 5.0, b = 3.0" << std::endl;
    std::cout << "a + b = " << salidas[0].scalar<float>()() << std::endl;
    std::cout << "a - b = " << salidas[1].scalar<float>()() << std::endl;
    std::cout << "a * b = " << salidas[2].scalar<float>()() << std::endl;
    std::cout << "a / b = " << salidas[3].scalar<float>()() << std::endl;
}

// Demuestra operaciones de tensores vectoriales y matriciales
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarOperacionesVectoriales(Scope& root, ClientSession& session) {
    std::cout << "\n=== Operaciones Vectoriales ===" << std::endl;
    
    // Crear tensores vectoriales
    auto vec1 = Const(root, {{1.0f, 2.0f, 3.0f}});
    auto vec2 = Const(root, {{4.0f, 5.0f, 6.0f}});
    
    // Operaciones elemento a elemento
    auto suma_vec = Add(root, vec1, vec2);
    auto prod_vec = Mul(root, vec1, vec2);
    
    // Ejecutar operaciones
    std::vector<Tensor> salidas;
    TF_CHECK_OK(session.Run({suma_vec, prod_vec}, &salidas));
    
    std::cout << "vec1 = [1, 2, 3]" << std::endl;
    std::cout << "vec2 = [4, 5, 6]" << std::endl;
    
    // Imprimir resultados
    auto datos_suma = salidas[0].flat<float>();
    std::cout << "vec1 + vec2 = [";
    for (int i = 0; i < datos_suma.size(); ++i) {
        std::cout << datos_suma(i);
        if (i < datos_suma.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    auto datos_prod = salidas[1].flat<float>();
    std::cout << "vec1 * vec2 = [";
    for (int i = 0; i < datos_prod.size(); ++i) {
        std::cout << datos_prod(i);
        if (i < datos_prod.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Demuestra operaciones matriciales incluyendo multiplicación
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarOperacionesMatriciales(Scope& root, ClientSession& session) {
    std::cout << "\n=== Operaciones Matriciales ===" << std::endl;
    
    // Crear matrices 2x2
    auto mat1 = Const(root, {{1.0f, 2.0f}, {3.0f, 4.0f}});
    auto mat2 = Const(root, {{5.0f, 6.0f}, {7.0f, 8.0f}});
    
    // Multiplicación de matrices
    auto mult_mat = MatMul(root, mat1, mat2);
    
    // Suma elemento a elemento
    auto suma_mat = Add(root, mat1, mat2);
    
    // Transpuesta
    auto transpuesta_mat = Transpose(root, mat1, {1, 0});
    
    // Ejecutar operaciones
    std::vector<Tensor> salidas;
    TF_CHECK_OK(session.Run({mult_mat, suma_mat, transpuesta_mat}, &salidas));
    
    std::cout << "mat1 = [[1, 2], [3, 4]]" << std::endl;
    std::cout << "mat2 = [[5, 6], [7, 8]]" << std::endl;
    
    // Imprimir resultado de multiplicación de matrices
    std::cout << "\nmat1 @ mat2 (multiplicación de matrices) = " << std::endl;
    auto datos_mult = salidas[0].matrix<float>();
    std::cout << "  [[" << datos_mult(0, 0) << ", " << datos_mult(0, 1) << "]," << std::endl;
    std::cout << "   [" << datos_mult(1, 0) << ", " << datos_mult(1, 1) << "]]" << std::endl;
    
    // Imprimir resultado de suma elemento a elemento
    std::cout << "\nmat1 + mat2 (elemento a elemento) = " << std::endl;
    auto datos_suma = salidas[1].matrix<float>();
    std::cout << "  [[" << datos_suma(0, 0) << ", " << datos_suma(0, 1) << "]," << std::endl;
    std::cout << "   [" << datos_suma(1, 0) << ", " << datos_suma(1, 1) << "]]" << std::endl;
    
    // Imprimir resultado de transpuesta
    std::cout << "\nTranspuesta(mat1) = " << std::endl;
    auto datos_trans = salidas[2].matrix<float>();
    std::cout << "  [[" << datos_trans(0, 0) << ", " << datos_trans(0, 1) << "]," << std::endl;
    std::cout << "   [" << datos_trans(1, 0) << ", " << datos_trans(1, 1) << "]]" << std::endl;
}

// Demuestra manipulación de formas de tensores
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarFormasTensores(Scope& root, ClientSession& session) {
    std::cout << "\n=== Formas de Tensores ===" << std::endl;
    
    // Crear un tensor con forma específica
    auto tensor_3x4 = Const(root, {
        {1.0f, 2.0f, 3.0f, 4.0f},
        {5.0f, 6.0f, 7.0f, 8.0f},
        {9.0f, 10.0f, 11.0f, 12.0f}
    });
    
    // Operación de redimensionamiento
    auto redim_2x6 = Reshape(root, tensor_3x4, {2, 6});
    auto redim_6x2 = Reshape(root, tensor_3x4, {6, 2});
    auto aplanado = Reshape(root, tensor_3x4, {-1});
    
    // Obtener formas
    auto forma_original = Shape(root, tensor_3x4);
    auto forma_nueva = Shape(root, redim_2x6);
    
    std::vector<Tensor> salidas;
    TF_CHECK_OK(session.Run({tensor_3x4, redim_2x6, redim_6x2, aplanado, 
                             forma_original, forma_nueva}, &salidas));
    
    std::cout << "Tensor original (3x4):" << std::endl;
    auto datos_orig = salidas[0].matrix<float>();
    for (int i = 0; i < 3; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 4; ++j) {
            std::cout << datos_orig(i, j);
            if (j < 3) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nRedimensionado a (2x6):" << std::endl;
    auto datos_redim = salidas[1].matrix<float>();
    for (int i = 0; i < 2; ++i) {
        std::cout << "  [";
        for (int j = 0; j < 6; ++j) {
            std::cout << datos_redim(i, j);
            if (j < 5) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "\nAplanado (12 elementos):" << std::endl;
    auto datos_planos = salidas[3].flat<float>();
    std::cout << "  [";
    for (int i = 0; i < 12; ++i) {
        std::cout << datos_planos(i);
        if (i < 11) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Demuestra funciones matemáticas (sin, cos, exp, log)
// root: El ámbito de TensorFlow para las operaciones
// session: La sesión del cliente para ejecutar operaciones
void demostrarFuncionesMatematicas(Scope& root, ClientSession& session) {
    std::cout << "\n=== Funciones Matemáticas ===" << std::endl;
    
    // Crear valores de entrada
    auto x = Const(root, {{0.0f, 0.5f, 1.0f, 1.5f, 2.0f}});
    
    // Aplicar funciones matemáticas
    auto sin_x = Sin(root, x);
    auto cos_x = Cos(root, x);
    auto exp_x = Exp(root, x);
    auto sqrt_x = Sqrt(root, Add(root, x, Const(root, 1.0f))); // sqrt(x+1) para evitar sqrt(0)
    
    std::vector<Tensor> salidas;
    TF_CHECK_OK(session.Run({x, sin_x, cos_x, exp_x, sqrt_x}, &salidas));
    
    auto datos_x = salidas[0].flat<float>();
    auto datos_sin = salidas[1].flat<float>();
    auto datos_cos = salidas[2].flat<float>();
    auto datos_exp = salidas[3].flat<float>();
    auto datos_sqrt = salidas[4].flat<float>();
    
    std::cout << std::fixed;
    std::cout.precision(4);
    
    std::cout << "valores x: [";
    for (int i = 0; i < 5; ++i) {
        std::cout << datos_x(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "sin(x):    [";
    for (int i = 0; i < 5; ++i) {
        std::cout << datos_sin(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "cos(x):    [";
    for (int i = 0; i < 5; ++i) {
        std::cout << datos_cos(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "exp(x):    [";
    for (int i = 0; i < 5; ++i) {
        std::cout << datos_exp(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "sqrt(x+1): [";
    for (int i = 0; i < 5; ++i) {
        std::cout << datos_sqrt(i);
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}
