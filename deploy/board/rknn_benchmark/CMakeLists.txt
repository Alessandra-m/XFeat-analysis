cmake_minimum_required(VERSION 3.6)

# Copy of rknn-toolkit2/rknpu2/examples/rknn_benchmark with an optional per-layer profile
# (env RKNN_PERF=1 -> rknn_init with RKNN_FLAG_COLLECT_PERF_MASK + RKNN_QUERY_PERF_DETAIL).
project(rknn_benchmark)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# path to rknn-toolkit2/rknpu2 (headers, librknnrt.so, 3rdparty)
set(RKNPU2_DIR "$ENV{HOME}/work/rknn/rknn-toolkit2/rknpu2" CACHE PATH "rknpu2 directory")

set(RKNN_API_PATH ${RKNPU2_DIR}/runtime/Linux/librknn_api)
set(RKNN_RT_LIB ${RKNN_API_PATH}/aarch64/librknnrt.so)
include_directories(${RKNN_API_PATH}/include)
include_directories(${RKNPU2_DIR}/examples/3rdparty)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rknn_benchmark
        src/rknn_benchmark.cpp
        src/cnpy/cnpy.cpp
)
target_link_libraries(rknn_benchmark ${RKNN_RT_LIB})

set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/rknn_benchmark_Linux)
install(TARGETS rknn_benchmark DESTINATION ./)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
