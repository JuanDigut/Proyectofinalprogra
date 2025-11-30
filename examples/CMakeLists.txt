# Examples subdirectory CMakeLists.txt

# Link libraries for TensorFlow
set(TENSORFLOW_LIBS tensorflow_cc tensorflow_framework)

# Example 1: Basic TensorFlow Operations
add_executable(basic_operations basic_operations.cpp)
target_link_libraries(basic_operations ${TENSORFLOW_LIBS})

# Example 2: Linear Regression
add_executable(linear_regression linear_regression.cpp)
target_link_libraries(linear_regression ${TENSORFLOW_LIBS})

# Example 3: Neural Network Classification
add_executable(neural_network neural_network.cpp)
target_link_libraries(neural_network ${TENSORFLOW_LIBS})

# Install targets
install(TARGETS basic_operations linear_regression neural_network
    RUNTIME DESTINATION bin
)
