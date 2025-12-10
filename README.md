# Ejemplos de TensorFlow C++ - Proyecto Final

Este repositorio contiene programas de ejemplo que demuestran la utilidad de la biblioteca TensorFlow C++. Los ejemplos muestran operaciones fundamentales, regresión lineal y clasificación con redes neuronales utilizando la API de C++ de TensorFlow.

## Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Requisitos Previos](#requisitos-previos)
- [Construcción del Contenedor Docker](#construcción-del-contenedor-docker)
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

Para ejecutar estos ejemplos, se utiliza Docker, que incluye todas las dependencias necesarias:

- **Docker** para construir y ejecutar los ejemplos en un entorno aislado

## Construcción del Contenedor Docker

La forma de ejecutar los ejemplos es utilizando Docker. El repositorio incluye un `Dockerfile` que construye automáticamente todos los ejemplos con todas las dependencias necesarias.

### Paso 1: Instalar Docker

Si aún no tienes Docker instalado:

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo systemctl enable docker
# Agregar tu usuario al grupo docker (opcional, para evitar usar sudo)
sudo usermod -aG docker $USER
```

**Otras plataformas:** Visita https://docs.docker.com/get-docker/

### Paso 2: Construir la Imagen Docker

```bash
# Clonar el repositorio (si aún no lo has hecho)
git clone https://github.com/JuanDigut/Proyectofinalprogra.git
cd Proyectofinalprogra

# Construir la imagen Docker
docker build -t tensorflow-cpp-ejemplos .
```

Este proceso puede tomar varios minutos la primera vez, ya que descarga e instala TensorFlow y compila los ejemplos.

### Paso 3: Ejecutar los Ejemplos

Una vez construida la imagen, puedes ejecutar los ejemplos directamente:

```bash
# Ejecutar el ejemplo de operaciones básicas
docker run --rm tensorflow-cpp-ejemplos basic_operations

# Ejecutar el ejemplo de regresión lineal
docker run --rm tensorflow-cpp-ejemplos linear_regression

# Ejecutar el ejemplo de red neuronal
docker run --rm tensorflow-cpp-ejemplos neural_network

# Ejecutar el ejemplo de detección de anomalías
docker run --rm tensorflow-cpp-ejemplos anomaly_detection

# Ejecutar el ejemplo de predicción de series temporales
docker run --rm tensorflow-cpp-ejemplos time_series_prediction
```

### Paso 3.1: Guardar Archivos CSV de Regresión Lineal

El programa `linear_regression` genera tres archivos CSV con resultados del análisis. Para acceder a estos archivos desde tu sistema host, tienes dos opciones:

#### Opción 1: Montar un Volumen

Monta un directorio local como volumen para que los archivos se guarden directamente en tu sistema:

```bash
# Crear directorio para los resultados
mkdir -p resultados

# Ejecutar linear_regression con volumen montado
docker run --rm -v $(pwd)/resultados:/resultados tensorflow-cpp-ejemplos sh -c \
  "linear_regression && cp /opt/proyecto/resultados_ruido.csv /opt/proyecto/resultados_tasa_aprendizaje.csv /opt/proyecto/progresion_perdida.csv /resultados/"

# Los archivos CSV estarán en ./resultados/
```

#### Opción 2: Copiar Archivos Después de la Ejecución

Ejecuta el contenedor sin `--rm` y luego copia los archivos:

```bash
# Ejecutar sin --rm para mantener el contenedor
docker run --name tf-linear tensorflow-cpp-ejemplos linear_regression

# Copiar los archivos CSV al host (desde el directorio de trabajo del contenedor /opt/proyecto)
docker cp tf-linear:/opt/proyecto/resultados_ruido.csv .
docker cp tf-linear:/opt/proyecto/resultados_tasa_aprendizaje.csv .
docker cp tf-linear:/opt/proyecto/progresion_perdida.csv .

# Eliminar el contenedor
docker rm tf-linear
```

#### Archivos CSV Generados

Los archivos contienen los siguientes datos:

1. **`resultados_ruido.csv`** - Análisis de sensibilidad al ruido
   - Columnas: `nivel_ruido`, `pendiente_aprendida`, `intercepto_aprendido`, `error_pendiente`, `error_intercepto`, `mse`, `rmse`

2. **`resultados_tasa_aprendizaje.csv`** - Análisis de sensibilidad a la tasa de aprendizaje
   - Columnas: `tasa_aprendizaje`, `pendiente_aprendida`, `intercepto_aprendido`, `error_pendiente`, `error_intercepto`, `mse`, `rmse`, `epocas_convergencia`

3. **`progresion_perdida.csv`** - Progresión de la pérdida durante el entrenamiento
   - Columnas: `epoca`, `perdida`

### Paso 4: Shell Interactivo (Opcional)

Si deseas explorar dentro del contenedor:

```bash
docker run --rm -it tensorflow-cpp-ejemplos /bin/bash

# Dentro del contenedor puedes ejecutar:
basic_operations
linear_regression
neural_network
# etc.
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
├── dockerfile               # Definición del contenedor Docker
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

## Licencia

Este proyecto es con fines educativos como parte de un proyecto final de programación.