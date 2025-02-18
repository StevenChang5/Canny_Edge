add_executable(
  test_cuda
  test_cuda.cpp
)

target_include_directories(
    test_cuda 
    PUBLIC 
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(
  test_cuda
  GTest::gtest_main
  cuda
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

include(GoogleTest)
gtest_discover_tests(test_cuda)