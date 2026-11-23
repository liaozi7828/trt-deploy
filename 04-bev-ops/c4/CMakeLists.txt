cmake_minimum_required(VERSION 3.20)
project(pillar_plugin LANGUAGES CXX CUDA)

# ★★ 必须用 CACHE FORCE，不能写 if(NOT DEFINED ...)：
#    CMake **默认就会定义** CMAKE_CUDA_ARCHITECTURES（本机默认是 52），
#    所以 if(NOT DEFINED) 永远不成立 → 架构一直是 52。
#
#    架构错了的后果远不止"慢"：
#      · 只生成 compute_52 的 PTX → 运行时走 JIT
#      · **compute_52 的编程模型是 warp 锁步**（Volta 之前没有独立线程调度）
#      · kernel 里 `__nanosleep` 被 `#if __CUDA_ARCH__ >= 700` 编译掉
#      → 自旋等待会卡住同 warp 里持有 claim 的线程 → **真死锁**
#    （症状：GPU util 有几十%、SM 频率很低、程序永远不返回）
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
message(STATUS "CUDA arch (强制) = ${CMAKE_CUDA_ARCHITECTURES}")

# 和项目② 一样：.cu 由 nvcc 编译，必须去掉 _GNU_SOURCE，
# 否则在 glibc 2.41 上 <cwchar> 会缺声明、STL 头文件编不过。
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -U_GNU_SOURCE")

option(TRT_ROOT "TensorRT 根目录（含 include/ 和 lib/）" "")

find_package(CUDAToolkit REQUIRED)

# TensorRT：先试 find_package，失败再用 TRT_ROOT 手写 imported target
find_package(TensorRT QUIET)
if(NOT TensorRT_FOUND)
    if(NOT TRT_ROOT)
        message(FATAL_ERROR "找不到 TensorRT：请加 -DTRT_ROOT=/path/to/TensorRT")
    endif()
    add_library(TensorRT::nvinfer SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TRT_ROOT}/lib/libnvinfer.so"
        INTERFACE_INCLUDE_DIRECTORIES "${TRT_ROOT}/include")
    add_library(TensorRT::nvinfer_plugin SHARED IMPORTED)
    set_target_properties(TensorRT::nvinfer_plugin PROPERTIES
        IMPORTED_LOCATION "${TRT_ROOT}/lib/libnvinfer_plugin.so")
endif()
message(STATUS "TensorRT include: $<TARGET_PROPERTY:TensorRT::nvinfer,INTERFACE_INCLUDE_DIRECTORIES>")

add_executable(pillar_plugin_demo
    main.cpp
    pillar_plugin.cpp
    scatter_kernel.cu)

target_include_directories(pillar_plugin_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(pillar_plugin_demo PRIVATE
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    CUDA::cudart)
# 让 REGISTER_TENSORRT_PLUGIN 的静态初始化器不被链接器丢掉
target_link_options(pillar_plugin_demo PRIVATE -Wl,--no-as-needed)
