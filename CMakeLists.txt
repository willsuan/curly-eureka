@@
-cmake_minimum_required(VERSION 3.22)
-project(cuHyperDual LANGUAGES CXX CUDA)
-
-set(CMAKE_CXX_STANDARD 17)
-set(CMAKE_CUDA_STANDARD 17)
-
-# Enable reasonable defaults
-set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "")
-set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
-add_definitions(-D_FORCE_INLINES)
-
-# Build type
-if(NOT CMAKE_BUILD_TYPE)
-  set(CMAKE_BUILD_TYPE Release)
-endif()
-
-# Options
-option(CUHD_USE_DOUBLE "Build examples in double precision" ON)
-
-include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
-
-add_executable(hyperdual src/main.cu src/kernel.cu)
-
-# Avoid unsafe fast-math; enable FMA contractions where legal
-target_compile_options(hyperdual PRIVATE
-  $<$<COMPILE_LANGUAGE:CUDA>:-O3;-Xcompiler=-O3;-lineinfo>
-  $<$<COMPILE_LANGUAGE:CXX>:-O3>
-)
-
-option(BUILD_PY "Build Python module with pybind11" OFF)
-if(BUILD_PY)
-  find_package(pybind11 REQUIRED)
-  add_library(cuhd MODULE src/pybind_module.cpp)
-  set_target_properties(cuhd PROPERTIES PREFIX "" OUTPUT_NAME "cuhd")
-  target_link_libraries(cuhd PRIVATE pybind11::module)
-  target_include_directories(cuhd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
-  # Link CUDA runtime if needed
-  target_link_libraries(cuhd PRIVATE cuda)
-endif()
+cmake_minimum_required(VERSION 3.24)
+project(cuHyperDual LANGUAGES CXX CUDA)
+
+set(CMAKE_CXX_STANDARD 17)
+set(CMAKE_CUDA_STANDARD 17)
+set(CMAKE_POSITION_INDEPENDENT_CODE ON)
+set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
+
+# Be explicit about architectures; fallback if none provided by user/CI
+if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
+  # Reasonable defaults for modern toolkits; edit to taste
+  set(CMAKE_CUDA_ARCHITECTURES 60;70;75;80;86;89;90)
+endif()
+
+if(NOT CMAKE_BUILD_TYPE)
+  set(CMAKE_BUILD_TYPE Release)
+endif()
+
+option(CUHD_USE_DOUBLE "Build examples in double precision" ON)
+option(BUILD_PY "Build Python module with pybind11" OFF)
+
+# Library with CUDA code
+add_library(cuhyperdual STATIC
+  src/kernel.cu
+)
+target_include_directories(cuhyperdual PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
+target_compile_options(cuhyperdual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>)
+find_package(CUDAToolkit REQUIRED) # provides imported targets like CUDA::cudart
+target_link_libraries(cuhyperdual PUBLIC CUDA::cudart)
+
+# Demo executable
+add_executable(hyperdual src/main.cu)
+target_link_libraries(hyperdual PRIVATE cuhyperdual)
+
+# Optional Python extension
+if(BUILD_PY)
+  find_package(pybind11 CONFIG REQUIRED)
+  pybind11_add_module(cuhd src/pybind_module.cpp)
+  target_include_directories(cuhd PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
+  target_link_libraries(cuhd PRIVATE cuhyperdual CUDA::cudart)
+  # Ensure no 'lib' prefix on the Python extension
+  set_target_properties(cuhd PROPERTIES PREFIX "" OUTPUT_NAME "cuhd")
+endif()
