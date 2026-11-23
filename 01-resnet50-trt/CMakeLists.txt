cmake_minimum_required(VERSION 3.22)
project(resnet50_trt VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
# 找 CUDA
# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# sm_120 = Blackwell（RTX 50 系）。换卡要改这里！
#
# ⚠️ 必须用 CACHE ... FORCE：CMake 在 enable_language(CUDA) 时会自己
#    定义 CMAKE_CUDA_ARCHITECTURES（默认取检测到的算力，本机曾误判为 52）。
#    用 if(NOT DEFINED) 永远不成立，所以这里强制覆盖。
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA arch (120 = Blackwell)" FORCE)
message(STATUS "CUDA arch forced to: ${CMAKE_CUDA_ARCHITECTURES}")

# ---------------------------------------------------------------------------
# 找 TensorRT（C++ 开发文件：头文件 + 库）
#
# 三种可能的来源，按顺序找：
#   1. -DTRT_ROOT=/path   手动指定
#   2. pip 装的 tensorrt（~/.venvs/*/lib/python*/site-packages/tensorrt_libs）
#   3. 系统装的（/usr/lib/x86_64-linux-gnu, /usr/include）
# ---------------------------------------------------------------------------
set(TRT_HINTS "")
if(TRT_ROOT)
    list(APPEND TRT_HINTS ${TRT_ROOT})
endif()
file(GLOB _pip_trt "$ENV{HOME}/.venvs/*/lib/python*/site-packages/tensorrt*")
list(APPEND TRT_HINTS ${_pip_trt})
list(APPEND TRT_HINTS
    /usr/local/TensorRT
    /usr
    /opt/tensorrt)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${TRT_HINTS}
    PATH_SUFFIXES include
    NO_DEFAULT_PATH)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)   # 兜底：系统路径

find_library(TENSORRT_LIBRARY
    NAMES nvinfer
    HINTS ${TRT_HINTS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
find_library(TENSORRT_LIBRARY NAMES nvinfer)

find_library(TENSORRT_PLUGIN_LIBRARY
    NAMES nvinfer_plugin
    HINTS ${TRT_HINTS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
find_library(TENSORRT_PLUGIN_LIBRARY NAMES nvinfer_plugin)

find_library(TENSORRT_ONNXPARSER_LIBRARY
    NAMES nvonnxparser
    HINTS ${TRT_HINTS}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH)
find_library(TENSORRT_ONNXPARSER_LIBRARY NAMES nvonnxparser)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR
        "\n找不到 TensorRT 的 C++ 开发文件（NvInfer.h / libnvinfer.so）。\n"
        "  已尝试的路径: ${TRT_HINTS}\n"
        "  解决方式见 README「先决条件」一节：\n"
        "    下载 TensorRT tar 包，解压后 -DTRT_ROOT=/path/to/TensorRT\n")
endif()

message(STATUS "TensorRT include : ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer : ${TENSORRT_LIBRARY}")
message(STATUS "TensorRT parser  : ${TENSORRT_ONNXPARSER_LIBRARY}")

add_library(TensorRT::nvinfer SHARED IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION ${TENSORRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})

add_library(TensorRT::onnxparser SHARED IMPORTED)
set_target_properties(TensorRT::onnxparser PROPERTIES
    IMPORTED_LOCATION ${TENSORRT_ONNXPARSER_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})
if(TENSORRT_PLUGIN_LIBRARY)
    add_library(TensorRT::nvinfer_plugin SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        IMPORTED_LOCATION ${TENSORRT_PLUGIN_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})
endif()

# ---------------------------------------------------------------------------
# 库：把 TrtEngine + 前处理打包成静态库
# ---------------------------------------------------------------------------
add_library(resnet50_core STATIC
    src/trt_engine.cpp
    src/preprocess.cu        # device code（nvcc 编译，不能碰标准库容器）
    src/preprocess_cpu.cpp)  # CPU 参考实现（g++ 编译，可以用 std::vector）

target_include_directories(resnet50_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(resnet50_core PRIVATE -Wall -Wextra)

target_link_libraries(resnet50_core PUBLIC
    CUDA::cudart
    TensorRT::nvinfer
    TensorRT::onnxparser)

# CUDA 12.8 的主机编译器只认 gcc<=14，系统默认 gcc 15 会被拒。
# 已在 ~/.bashrc 里用 NVCC_PREPEND_FLAGS 全局处理，这里再兜一层。
if(EXISTS /usr/bin/g++-14)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-14)
endif()

# ---------------------------------------------------------------------------
# 可执行文件
# ---------------------------------------------------------------------------
add_executable(resnet50_demo src/main.cpp)
target_link_libraries(resnet50_demo PRIVATE resnet50_core)

message(STATUS "------------------------------------------------------")
message(STATUS "resnet50_trt ${PROJECT_VERSION}")
message(STATUS "  C++ 编译器 : ${CMAKE_CXX_COMPILER}")
message(STATUS "  CUDA 架构  : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  构建类型   : ${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------")
