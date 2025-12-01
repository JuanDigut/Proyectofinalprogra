# Ejemplos de TensorFlow C++ - Proyecto Final

Este repositorio contiene programas de ejemplo que demuestran la utilidad de la biblioteca TensorFlow C++. Los ejemplos muestran operaciones fundamentales, regresión lineal y clasificación con redes neuronales utilizando la API de C++ de TensorFlow.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Compilación de los Ejemplos](#compilación-de-los-ejemplos)
  - [Ejecución con Apptainer](#ejecución-con-apptainer-contenedor)
- [Ejemplos](#ejemplos)
  - [Operaciones Básicas](#ejemplo-1-operaciones-básicas)
  - [Regresión Lineal](#ejemplo-2-regresión-lineal)
  - [Clasificación con Red Neuronal](#ejemplo-3-clasificación-con-red-neuronal)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Licencia](#licencia)

## Descripción General

TensorFlow proporciona una poderosa API de C++ que permite a los desarrolladores construir y desplegar modelos de aprendizaje automático en aplicaciones C++. Esto es particularmente útil para:

- **Aplicaciones de alto rendimiento** donde se requiere C++
- **Sistemas embebidos** y dispositivos IoT
- **Despliegues en producción** que requieren inferencia de baja latencia
- **Integración** con bases de código C++ existentes

Este proyecto demuestra tres casos de uso clave:
1. Operaciones básicas con tensores y funciones matemáticas
2. Entrenamiento de un modelo de regresión lineal
3. Construcción y entrenamiento de una red neuronal para clasificación

## Requisitos Previos

Antes de compilar estos ejemplos, necesitas:

- **Compilador C++** con soporte para C++17 (GCC 7+, Clang 5+, o MSVC 2019+)
- **CMake** versión 3.16 o superior
- **Biblioteca TensorFlow C++** (libtensorflow_cc)

### Instalación de la Biblioteca TensorFlow C++

#### Opción 1: Compilar desde el Código Fuente (Recomendado para funcionalidad completa)

```bash
# Clonar el repositorio de TensorFlow
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Configurar la compilación
./configure

# Compilar la biblioteca C++
bazel build //tensorflow:libtensorflow_cc.so

# La biblioteca estará en bazel-bin/tensorflow/
```

#### Opción 2: Usar Binarios Precompilados

Descarga las bibliotecas C de TensorFlow precompiladas desde:
https://www.tensorflow.org/install/lang_c

Nota: La API de C tiene funcionalidad limitada comparada con la API completa de C++.

#### Opción 3: Usar Gestores de Paquetes

**Ubuntu/Debian:**
```bash
# Agregar PPA de TensorFlow (si está disponible) o compilar desde código fuente
sudo apt-get install libtensorflow-dev
```

**macOS con Homebrew:**
```bash
brew install tensorflow
```

## Compilación de los Ejemplos

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/JuanDigut/Proyectofinalprogra.git
cd Proyectofinalprogra
```

### Paso 2: Crear el Directorio de Compilación

```bash
mkdir build
cd build
```

### Paso 3: Configurar con CMake

Si TensorFlow está instalado a nivel de sistema:
```bash
cmake ..
```

Si TensorFlow está en una ubicación personalizada:
```bash
cmake -DTENSORFLOW_DIR=/ruta/a/tensorflow ..
```

O usando una variable de entorno:
```bash
export TENSORFLOW_DIR=/ruta/a/tensorflow
cmake ..
```

### Paso 4: Compilar

```bash
make -j$(nproc)
```

### Paso 5: Ejecutar los Ejemplos

```bash
# Ejecutar el ejemplo de operaciones básicas
./examples/basic_operations

# Ejecutar el ejemplo de regresión lineal
./examples/linear_regression

# Ejecutar el ejemplo de red neuronal
./examples/neural_network
```

### Ejecución con Apptainer (Contenedor)

Para ejecutar los ejemplos en un entorno aislado y reproducible, puedes usar Apptainer (anteriormente Singularity). El repositorio incluye un archivo de definición `apptainer.def` basado en Docker.

#### Definición del Contenedor Apptainer

El archivo `apptainer.def` define un contenedor basado en Ubuntu 22.04 con todas las dependencias necesarias:

```singularity
Bootstrap: docker
From: ubuntu:22.04

%labels
    Author JuanDigut
    Version 1.0
    Description Contenedor para compilar y ejecutar ejemplos de TensorFlow C++

%post
    # Actualizar e instalar dependencias básicas
    apt-get update && apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        pkg-config \
        python3 \
        python3-pip \
        python3-numpy \
        libprotobuf-dev \
        protobuf-compiler \
        && rm -rf /var/lib/apt/lists/*

    # Instalar TensorFlow C API (base para C++)
    cd /tmp
    wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.14.0.tar.gz
    tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.14.0.tar.gz
    ldconfig
    rm libtensorflow-cpu-linux-x86_64-2.14.0.tar.gz

    # Configurar variables de entorno para TensorFlow
    echo 'export TENSORFLOW_DIR=/usr/local' >> /etc/bash.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> /etc/bash.bashrc

%environment
    export TENSORFLOW_DIR=/usr/local
    export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

%runscript
    echo "Contenedor de TensorFlow C++ listo para compilar y ejecutar ejemplos"
    echo "Uso: apptainer exec tensorflow_cpp.sif <comando>"
    exec "$@"

%help
    Este contenedor proporciona un entorno para compilar y ejecutar
    ejemplos de TensorFlow C++.
```

#### Pasos para Ejecutar con Apptainer

**Paso 1: Instalar Apptainer**

En Ubuntu/Debian:
```bash
# Agregar el repositorio de Apptainer
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt update
sudo apt install -y apptainer
```

En sistemas con módulos de ambiente (HPC):
```bash
module load apptainer
# o
module load singularity
```

**Paso 2: Clonar el Repositorio**

```bash
git clone https://github.com/JuanDigut/Proyectofinalprogra.git
cd Proyectofinalprogra
```

**Paso 3: Construir el Contenedor**

```bash
# Construir la imagen del contenedor desde la definición
sudo apptainer build tensorflow_cpp.sif apptainer.def
```

Nota: Si no tienes permisos de sudo, puedes usar el modo fakeroot:
```bash
apptainer build --fakeroot tensorflow_cpp.sif apptainer.def
```

**Paso 4: Compilar los Ejemplos Dentro del Contenedor**

```bash
# Crear directorio de compilación y compilar
apptainer exec --bind $(pwd):/proyecto tensorflow_cpp.sif bash -c "
    cd /proyecto && \
    mkdir -p build && \
    cd build && \
    cmake .. && \
    make -j\$(nproc)
"
```

**Paso 5: Ejecutar los Ejemplos**

```bash
# Ejecutar el ejemplo de operaciones básicas
apptainer exec --bind $(pwd):/proyecto tensorflow_cpp.sif /proyecto/build/examples/basic_operations

# Ejecutar el ejemplo de regresión lineal
apptainer exec --bind $(pwd):/proyecto tensorflow_cpp.sif /proyecto/build/examples/linear_regression

# Ejecutar el ejemplo de red neuronal
apptainer exec --bind $(pwd):/proyecto tensorflow_cpp.sif /proyecto/build/examples/neural_network
```

**Alternativa: Shell Interactivo**

También puedes entrar al contenedor de forma interactiva:
```bash
apptainer shell --bind $(pwd):/proyecto tensorflow_cpp.sif

# Dentro del contenedor:
cd /proyecto/build
./examples/basic_operations
./examples/linear_regression
./examples/neural_network
```

## Ejemplos

### Ejemplo 1: Operaciones Básicas

**Archivo:** `examples/basic_operations.cpp`

Este ejemplo demuestra operaciones fundamentales de TensorFlow C++:

- **Operaciones Escalares**: Suma, resta, multiplicación, división
- **Operaciones Vectoriales**: Operaciones elemento a elemento en tensores 1D
- **Operaciones Matriciales**: Multiplicación de matrices, transposición, operaciones elemento a elemento
- **Formas de Tensores**: Redimensionamiento de tensores, aplanamiento
- **Funciones Matemáticas**: sin, cos, exp, sqrt

**Salida de Ejemplo:**
```
=== Operaciones Escalares ===
a = 5.0, b = 3.0
a + b = 8
a - b = 2
a * b = 15
a / b = 1.66667

=== Operaciones Matriciales ===
mat1 @ mat2 (multiplicación de matrices) =
  [[19, 22],
   [43, 50]]
```

### Ejemplo 2: Regresión Lineal

**Archivo:** `examples/linear_regression.cpp`

Este ejemplo implementa un modelo simple de regresión lineal que aprende a ajustar una línea `y = mx + b` a datos de entrenamiento sintéticos.

**Conceptos Clave:**
- Creación de variables entrenables (pesos y sesgos)
- Definición de una función de pérdida (Error Cuadrático Medio)
- Cálculo de gradientes para retropropagación
- Optimización por descenso de gradiente
- Implementación del bucle de entrenamiento

**Salida de Ejemplo:**
```
=== Entrenamiento ===
Época    0 | Pérdida: 125.3456
Época  100 | Pérdida: 0.2541
Época  200 | Pérdida: 0.2512
...
=== Resultados ===
Pendiente aprendida:     2.4823 (real: 2.5)
Intercepto aprendido: 1.0234 (real: 1.0)
```

### Ejemplo 3: Clasificación con Red Neuronal

**Archivo:** `examples/neural_network.cpp`

Este ejemplo construye una red neuronal de 2 capas para clasificación binaria en un patrón similar a XOR.

**Arquitectura de la Red:**
- Capa de entrada: 2 neuronas (características)
- Capa oculta: 8 neuronas con activación Sigmoide
- Capa de salida: 1 neurona con activación Sigmoide

**Conceptos Clave:**
- Construcción de redes neuronales multicapa
- Funciones de activación (Sigmoide)
- Pérdida de Entropía Cruzada Binaria
- Implementación manual de retropropagación
- Métricas de precisión de clasificación

**Salida de Ejemplo:**
```
=== Entrenamiento ===
Época    0 | Pérdida: 0.6931 | Precisión: 50.00%
Época  200 | Pérdida: 0.5234 | Precisión: 72.50%
Época  400 | Pérdida: 0.3156 | Precisión: 88.00%
...
=== Evaluación Final ===
Pérdida Final: 0.0823
Precisión Final: 97.50%
```

## Estructura del Proyecto

```
Proyectofinalprogra/
├── CMakeLists.txt           # Configuración principal de CMake
├── README.md                # Este archivo
├── apptainer.def            # Definición del contenedor Apptainer
├── examples/
│   ├── CMakeLists.txt       # Configuración de CMake para ejemplos
│   ├── basic_operations.cpp # Ejemplo 1: Operaciones básicas con tensores
│   ├── linear_regression.cpp# Ejemplo 2: Modelo de regresión lineal
│   └── neural_network.cpp   # Ejemplo 3: Clasificador de red neuronal
├── include/                 # Archivos de cabecera (si es necesario)
└── src/                     # Archivos fuente (si es necesario)
```

## Conceptos Clave de la API TensorFlow C++

### Scope (Ámbito)
La clase `Scope` define un espacio de nombres para las operaciones. Ayuda a organizar el grafo computacional.

```cpp
Scope root = Scope::NewRootScope();
```

### Operaciones
Las operaciones son los nodos en el grafo computacional:

```cpp
auto a = Const(root, 5.0f);      // Constante
auto b = Placeholder(root, DT_FLOAT);  // Marcador de posición para entrada
auto c = Add(root, a, b);        // Operación de suma
```

### Sesión
El `ClientSession` ejecuta las operaciones en el grafo:

```cpp
ClientSession session(root);
std::vector<Tensor> outputs;
session.Run({c}, &outputs);
```

### Tensores
Los tensores son arreglos multidimensionales:

```cpp
Tensor tensor(DT_FLOAT, TensorShape({3, 4}));
auto matrix = tensor.matrix<float>();
matrix(0, 0) = 1.0f;
```

## Solución de Problemas

### Problemas Comunes

1. **TensorFlow no encontrado**
   - Asegúrate de que `TENSORFLOW_DIR` esté configurado correctamente
   - Verifica que `libtensorflow_cc.so` exista en el directorio lib

2. **Cabeceras faltantes**
   - Verifica que la ruta de inclusión de TensorFlow sea correcta
   - Asegúrate de tener las cabeceras completas de TensorFlow C++ (no solo la API de C)

3. **Errores de enlace**
   - Agrega la ruta de la biblioteca TensorFlow a `LD_LIBRARY_PATH`:
     ```bash
     export LD_LIBRARY_PATH=$TENSORFLOW_DIR/lib:$LD_LIBRARY_PATH
     ```

4. **Errores en tiempo de ejecución**
   - Asegúrate de usar una versión compatible de TensorFlow
   - Verifica la compatibilidad de CUDA/cuDNN si usas GPU

## Licencia

Este proyecto es con fines educativos como parte de un proyecto final de programación.