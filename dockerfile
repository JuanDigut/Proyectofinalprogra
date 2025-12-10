FROM tensorflow/tensorflow:devel

LABEL author="JuanDigut"
LABEL description="Contenedor para ejecutar ejemplos de TensorFlow C++ del Proyecto Final"
LABEL version="1.0"

# Instalar dependencias (se cachea)
RUN set -ex && \
    apt-get update && apt-get install -y \
        cmake git pkg-config rsync \
    && rm -rf /var/lib/apt/lists/*

# Configurar TensorFlow (se cachea - esto es lo que más tarda)
RUN set -ex && \
    pip install --upgrade pip && \
    pip install tensorflow==2.13.1 && \
    mkdir -p /opt/tensorflow/lib /opt/tensorflow/include && \
    TENSORFLOW_PATH=$(python3 -c "import tensorflow as tf; print(tf.sysconfig.get_lib())") && \
    TENSORFLOW_INCLUDE=$(python3 -c "import tensorflow as tf; print(tf.sysconfig.get_include())") && \
    cp -P ${TENSORFLOW_PATH}/*.so* /opt/tensorflow/lib/ 2>/dev/null || true && \
    cd /opt/tensorflow/lib && \
    for lib in libtensorflow_cc libtensorflow_framework; do \
        if [ -f "${lib}.so.2" ]; then \
            ln -sf ${lib}.so.2 ${lib}.so; \
        fi; \
    done && \
    cp -r ${TENSORFLOW_INCLUDE}/* /opt/tensorflow/include/ && \
    echo "/opt/tensorflow/lib" > /etc/ld.so.conf.d/tensorflow.conf && \
    ldconfig

# Variables de entorno
ENV TENSORFLOW_DIR=/opt/tensorflow
ENV LD_LIBRARY_PATH=/opt/tensorflow/lib
ENV TF_CPP_MIN_LOG_LEVEL=2

# Copiar archivos locales (DESPUÉS de instalar TensorFlow)
COPY . /opt/proyecto

# Compilar el proyecto (solo esto se re-ejecuta al cambiar código)
RUN cd /opt/proyecto && \
    rm -rf build && \
    mkdir -p build && cd build && \
    cmake -DTENSORFLOW_DIR=/opt/tensorflow \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_FLAGS="-I/opt/tensorflow/include" \
          -DCMAKE_EXE_LINKER_FLAGS="-L/opt/tensorflow/lib -Wl,-rpath,/opt/tensorflow/lib" \
          .. && \
    make -j$(nproc)

ENV PATH=/opt/proyecto/build/examples:$PATH
WORKDIR /opt/proyecto

CMD ["/bin/bash"]
