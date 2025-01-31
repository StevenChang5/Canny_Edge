include_directories(${OpenCV_INCLUDE_DIRECTORIES})
add_executable(Main main.cpp)
target_include_directories(
  Main
  PUBLIC
  ${PROJECT_SOURCE_DIR}
)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm72")
  message(STATUS "Configuring for Raspberry Pi")
  target_compile_options(Main PRIVATE -I. $<SHELL_COMMAND:pkg-config --cflags opencv4>)
  target_link_libraries(Main PRIVATE utils $<SHELL_COMMAND:pkg-config --libs opencv4>)
else()
  target_link_libraries(Main PRIVATE utils ${OpenCV_LIBS})
endif()
