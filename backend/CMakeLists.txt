cmake_minimum_required(VERSION 3.10)
project(backend)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(OpenSSL REQUIRED)

# Cari cpprest library
find_path(CPPREST_INCLUDE_DIR NAMES cpprest/http_client.h PATH_SUFFIXES cpprest)
find_library(CPPREST_LIBRARY NAMES cpprest cpprestsdk)

if(NOT CPPREST_LIBRARY)
    message(FATAL_ERROR "cpprest library not found")
endif()

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/include/onnxruntime
)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    PATHS /usr/local/lib
)

if(NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "onnxruntime library not found")
endif()

add_executable(backend 
    src/main.cpp
    src/server/server.cpp 
    src/base64/base64.cpp
    src/detector/face_detector.cpp
    src/embedder/face_embedder.cpp
    src/db/face_db.cpp
    src/anti_spoof/anti_spoof.cpp
    src/anti_spoof/depth_anything.cpp
)

target_include_directories(backend PRIVATE 
    ${CPPREST_INCLUDE_DIR} 
    ${ONNXRUNTIME_INCLUDE_DIR}
    src
)

target_link_libraries(backend 
    ${OpenCV_LIBS} 
    ${CPPREST_LIBRARY}
    ${ONNXRUNTIME_LIBRARY}
    OpenSSL::SSL 
    OpenSSL::Crypto 
    pthread
)