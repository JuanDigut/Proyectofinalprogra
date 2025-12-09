FROM tensorflow/tensorflow:devel

LABEL author="JuanDigut"
LABEL description="Contenedor para ejecutar ejemplos de TensorFlow C++ del Proyecto Final"
LABEL version="1.0"

# Instalar dependencias
RUN set -ex && \
    apt-get update && apt-get install -y \
        cmake git pkg-config rsync \
    && rm -rf /var/lib/apt/lists/*

# Clonar repositorio
RUN git clone https://github.com/JuanDigut/Proyectofinalprogra.git /opt/proyecto

# Configurar TensorFlow
RUN set -ex && \
    pip install --upgrade pip && \
    pip install tensorflow==2.13.1 && \
    mkdir -p /opt/tensorflow/lib /opt/tensorflow/include && \
    # Get TensorFlow paths
    TENSORFLOW_PATH=$(python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())") && \
    TENSORFLOW_INCLUDE=$(python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())") && \
    echo "TensorFlow path: $TENSORFLOW_PATH" && \
    echo "TensorFlow include: $TENSORFLOW_INCLUDE" && \
    # List available libraries
    echo "=== Available .so files ===" && \
    ls -la ${TENSORFLOW_PATH}/*.so* || true && \
    # Copy ALL TensorFlow libraries
    cp -P ${TENSORFLOW_PATH}/*.so* /opt/tensorflow/lib/ 2>/dev/null || true && \
    # Create symlinks for the linker
    cd /opt/tensorflow/lib && \
    echo "=== Copied libraries ===" && \
    ls -la && \
    # Create symlinks if versioned files exist
    for lib in libtensorflow_cc libtensorflow_framework; do \
        if [ -f "${lib}.so.2" ]; then \
            ln -sf ${lib}.so.2 ${lib}.so; \
        fi; \
    done && \
    echo "=== After symlinks ===" && \
    ls -la && \
    # Copy headers
    cp -r ${TENSORFLOW_INCLUDE}/* /opt/tensorflow/include/ && \
    # Update library cache
    echo "/opt/tensorflow/lib" > /etc/ld.so.conf.d/tensorflow.conf && \
    ldconfig

# Compilar el proyecto
RUN set -ex && \
    cd /opt/proyecto && \
    rm -rf build && \
    mkdir -p build && cd build && \
    export LD_LIBRARY_PATH=/opt/tensorflow/lib:$LD_LIBRARY_PATH && \
    cmake -DTENSORFLOW_DIR=/opt/tensorflow \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-I/opt/tensorflow/include" \
          -DCMAKE_EXE_LINKER_FLAGS="-L/opt/tensorflow/lib -Wl,-rpath,/opt/tensorflow/lib" \
          .. && \
    make VERBOSE=1 -j$(nproc)

# Variables de entorno
ENV TENSORFLOW_DIR=/opt/tensorflow
ENV LD_LIBRARY_PATH=/opt/tensorflow/lib
ENV PATH=/opt/proyecto/build/examples:$PATH
ENV TF_CPP_MIN_LOG_LEVEL=2

# Directorio de trabajo
WORKDIR /opt/proyecto

# Comando por defecto
CMD ["/bin/bash", "-c", "echo '=== Contenedor TensorFlow C++ - Proyecto Final ===' && \
    echo '' && \
    echo 'Programas disponibles:' && \
    echo '  - basic_operations       : Operaciones basicas con tensores' && \
    echo '  - linear_regression      : Modelo de regresion lineal' && \
    echo '  - neural_network         : Clasificacion con red neuronal' && \
    echo '  - anomaly_detection      : Deteccion de anomalias con autoencoder' && \
    echo '  - time_series_prediction : Prediccion de series temporales' && \
    echo '' && \
    exec /bin/bash"]
