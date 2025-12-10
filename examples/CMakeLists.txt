# Subdirectorio de ejemplos CMakeLists.txt

# Bibliotecas de enlace para TensorFlow
set(TENSORFLOW_LIBS tensorflow_cc tensorflow_framework)

# Ejemplo 1: Operaciones Básicas de TensorFlow
add_executable(basic_operations basic_operations.cpp)
target_link_libraries(basic_operations ${TENSORFLOW_LIBS})

# Ejemplo 2: Regresión Lineal
add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression ${TENSORFLOW_LIBS})

# Ejemplo 6: Detección de Anomalías con Autoencoder
add_executable(anomaly_detection anomaly_detection.cpp)
target_link_libraries(anomaly_detection ${TENSORFLOW_LIBS})

# Instalar objetivos
install(TARGETS basic_operations linear_regression neural_network anomaly_detection time_series_prediction
    RUNTIME DESTINATION bin
)
