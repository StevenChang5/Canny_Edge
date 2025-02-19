include_directories(${OpenCV_INCLUDE_DIRECTORIES})

add_executable(main main.cpp)

add_library(utils utils.cpp)

target_link_libraries(utils PRIVATE ${OpenCV_LIBS})

if(ENABLE_CUDA)
  add_library(cuda cuda.cu)
  target_link_libraries(cuda PRIVATE ${OpenCV_LIBS} utils)
  target_link_libraries(main PRIVATE ${OpenCV_LIBS} utils cuda)
else()
  target_link_libraries(main PRIVATE ${OpenCV_LIBS} utils)
endif()


target_include_directories(
  main
  PUBLIC
  ${PROJECT_SOURCE_DIR}
)




