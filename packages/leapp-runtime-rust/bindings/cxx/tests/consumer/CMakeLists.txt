cmake_minimum_required(VERSION 3.22)
project(leapp_bundle_consumer LANGUAGES CXX)

find_package(leapp_runtime CONFIG REQUIRED)

add_executable(consumer main.cc)
target_link_libraries(consumer PRIVATE leapp::runtime)
