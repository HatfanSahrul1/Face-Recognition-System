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

add_executable(backend 
    src/main.cpp
    src/server/server.cpp 
    src/base64/base64.cpp
    src/detector/face_detector.cpp
)

target_include_directories(backend PRIVATE ${CPPREST_INCLUDE_DIR} src)
target_link_libraries(backend ${OpenCV_LIBS} ${CPPREST_LIBRARY} OpenSSL::SSL OpenSSL::Crypto pthread)