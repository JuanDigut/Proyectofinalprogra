FROM tensorflow/tensorflow:devel

LABEL author="JuanDigut"
LABEL description="Contenedor para ejecutar ejemplos de TensorFlow C++ del Proyecto Final"
LABEL version="1.0"

# Copiar archivos del proyecto
COPY Proyectofinalprogra /opt/proyecto

# Instalar dependencias y configurar TensorFlow
RUN set -ex && \
    apt-get update && apt-get install -y \
        cmake git pkg-config rsync \
    && rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install tensorflow==2.13.1 && \
    mkdir -p /opt/tensorflow/lib /opt/tensorflow/include && \
    # Get TensorFlow paths and copy libraries
    TENSORFLOW_PATH=$(python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())") && \
    TENSORFLOW_INCLUDE=$(python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())") && \
    echo "TensorFlow path: $TENSORFLOW_PATH" && \
    echo "TensorFlow include: $TENSORFLOW_INCLUDE" && \
    # Copy ALL TensorFlow libraries
    cp -P ${TENSORFLOW_PATH}/*.so* /opt/tensorflow/lib/ 2>/dev/null || true && \
    # Create symlinks
    cd /opt/tensorflow/lib && \
    for lib in libtensorflow_cc libtensorflow_framework; do \
        if [ -f "${lib}.so.2" ]; then \
            ln -sf ${lib}.so.2 ${lib}.so; \
        fi \
    done && \
    # Copy headers
    cp -r ${TENSORFLOW_INCLUDE}/* /opt/tensorflow/include/ && \
    # Update library cache
    echo "/opt/tensorflow/lib" > /etc/ld.so.conf.d/tensorflow.conf && \
    ldconfig

# Compilar el proyecto
RUN cd /opt/proyecto && \
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
ENV LD_LIBRARY_PATH=/opt/tensorflow/lib:$LD_LIBRARY_PATH
ENV PATH=/opt/proyecto/build/examples:$PATH

# Directorio de trabajo
WORKDIR /opt/proyecto

# Comando por defecto
CMD ["/bin/bash", "-c", "echo '=== Contenedor TensorFlow C++ - Proyecto Final ===' && \
    echo '' && \
    echo 'Programas disponibles:' && \
    echo '  - basic_operations       : Operaciones básicas con tensores' && \
    echo '  - linear_regression      : Modelo de regresión lineal' && \
    echo '  - neural_network         : Clasificación con red neuronal' && \
    echo '  - anomaly_detection      : Detección de anomalías con autoencoder' && \
    echo '  - time_series_prediction : Predicción de series temporales' && \
    echo '' && \
    exec /bin/bash"]