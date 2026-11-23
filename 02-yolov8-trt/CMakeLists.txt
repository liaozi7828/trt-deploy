cmake_minimum_required(VERSION 3.22)
project(yolov8_trt VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ---------------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# Blackwell。CACHE FORCE：CMake 启用 CUDA 时会自己定义这个变量，必须强制覆盖。
set(CMAKE_CUDA_ARCHITECTURES "120" CACHE STRING "CUDA arch (120 = Blackwell)" FORCE)

# ---------------------------------------------------------------------------
# TensorRT（和 p2 一样的手工查找）
# ---------------------------------------------------------------------------
set(TRT_HINTS "")
if(TRT_ROOT)
    list(APPEND TRT_HINTS ${TRT_ROOT})
endif()

find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h
    HINTS ${TRT_HINTS} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_path(TENSORRT_INCLUDE_DIR NAMES NvInfer.h)

find_library(TENSORRT_LIBRARY NAMES nvinfer
    HINTS ${TRT_HINTS} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
find_library(TENSORRT_LIBRARY NAMES nvinfer)

find_library(TENSORRT_ONNXPARSER_LIBRARY NAMES nvonnxparser
    HINTS ${TRT_HINTS} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
find_library(TENSORRT_ONNXPARSER_LIBRARY NAMES nvonnxparser)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR
        "\n找不到 TensorRT 开发文件。请加 -DTRT_ROOT=/path/to/TensorRT\n")
endif()

add_library(TensorRT::nvinfer SHARED IMPORTED)
set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION ${TENSORRT_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})

add_library(TensorRT::onnxparser SHARED IMPORTED)
set_target_properties(TensorRT::onnxparser PROPERTIES
    IMPORTED_LOCATION ${TENSORRT_ONNXPARSER_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR})

message(STATUS "TensorRT: ${TENSORRT_INCLUDE_DIR}")

# ---------------------------------------------------------------------------
# 核心库
#
# ⚠️ trt_engine.* / preprocess.* / cuda_utils.h 是**从 01-resnet50-trt 复制**过来的：
#    它们是通用基础设施，与具体模型无关。等 P3 结束再考虑抽成公共库。
# ---------------------------------------------------------------------------
add_library(yolo_core STATIC
    src/trt_engine.cpp
    src/preprocess.cu             # 纯 CUDA（-U_GNU_SOURCE，不能碰 STL）
    src/preprocess_cpu.cpp        # CPU 基准（g++ 编译，可用 STL）
    src/nms_kernel.cu             # W2：GPU NMS 的 device 代码（纯 CUDA）
    src/nms_gpu.cpp               # W2：GPU NMS 的主机封装（g++ 编译）
    src/decode_kernel.cu          # P4：GPU decode 的 device 代码（纯 CUDA）
    src/decode_gpu.cpp            # P4：GPU decode 的主机封装（g++ 编译）
    src/decode_nms_kernel.cu      # P5：decode+NMS 融合（为 plugin 做准备）
    src/cuda_graph.cpp            # P4：CUDA Graph 捕获/重放（g++ 编译）
    src/multi_stream.cpp          # W7：多流并发（g++ 编译）
    src/postprocess.cpp)          # 纯 C++，g++ 编译
#
# ⚠️ 这里的分工不是随意的：.cu 由 nvcc 编译且带 -U_GNU_SOURCE，
#    不能 include <string>/<vector>/<algorithm>（<cwchar> 会缺声明）。
#    所以每个功能都拆成「.cu 只写 kernel」+「.cpp 做容器/业务」。

target_include_directories(yolo_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(yolo_core PRIVATE -Wall -Wextra)

target_link_libraries(yolo_core PUBLIC
    CUDA::cudart
    TensorRT::nvinfer
    TensorRT::onnxparser)

if(EXISTS /usr/bin/g++-14)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-14)
endif()

add_executable(yolov8_demo src/main.cpp)
target_link_libraries(yolov8_demo PRIVATE yolo_core)

# P5：推理服务（帧循环 + 延迟分位数）
add_executable(yolov8_service src/service.cpp)
target_link_libraries(yolov8_service PRIVATE yolo_core)

# P5：点云 pillarization（scatter）—— BEV 的核心难点，含 CPU 参考与自检
add_executable(pillarize_test
    src/pillarize.cpp
    src/scatter_kernel.cu)
target_include_directories(pillarize_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(pillarize_test PRIVATE CUDA::cudart)

# P5：decode+NMS 融合 kernel 的自检（对照 CPU 版，而 CPU 版已被 ultralytics 验证过）
add_executable(decode_nms_test src/decode_nms_test.cpp)
target_link_libraries(decode_nms_test PRIVATE yolo_core)

message(STATUS "------------------------------------------------------")
message(STATUS "yolov8_trt ${PROJECT_VERSION}")
message(STATUS "  CUDA arch : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  构建类型   : ${CMAKE_BUILD_TYPE}")
message(STATUS "------------------------------------------------------")
