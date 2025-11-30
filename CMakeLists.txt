cmake_minimum_required(VERSION 3.16)
project(TensorFlowCppExamples VERSION 1.0.0 LANGUAGES CXX)

# Establecer el estándar de C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Habilitar exportación de comandos de compilación para soporte de IDE
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Buscar la biblioteca TensorFlow C++
# El usuario necesita establecer TENSORFLOW_DIR al directorio de instalación de TensorFlow
# o instalar TensorFlow a nivel de sistema
if(DEFINED ENV{TENSORFLOW_DIR})
    set(TENSORFLOW_DIR $ENV{TENSORFLOW_DIR})
endif()

if(TENSORFLOW_DIR)
    message(STATUS "Usando TensorFlow desde: ${TENSORFLOW_DIR}")
    include_directories(${TENSORFLOW_DIR}/include)
    link_directories(${TENSORFLOW_DIR}/lib)
else()
    # Intentar encontrar TensorFlow usando pkg-config
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(TENSORFLOW tensorflow)
    endif()
    
    if(NOT TENSORFLOW_FOUND)
        message(WARNING "TensorFlow no encontrado. Por favor establece la variable de entorno TENSORFLOW_DIR o instala TensorFlow.")
        message(STATUS "Ejemplo: cmake -DTENSORFLOW_DIR=/ruta/a/tensorflow ..")
    endif()
endif()

# Agregar ejemplos
add_subdirectory(examples)

# Imprimir resumen de configuración
message(STATUS "")
message(STATUS "=== Configuración de Ejemplos TensorFlow C++ ===")
message(STATUS "Versión de CMake: ${CMAKE_VERSION}")
message(STATUS "Estándar C++: ${CMAKE_CXX_STANDARD}")
message(STATUS "Tipo de compilación: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
