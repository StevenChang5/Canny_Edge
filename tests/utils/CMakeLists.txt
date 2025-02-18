add_executable(
  test_utils
  test_utils.cpp
)

target_include_directories(
    test_utils 
    PUBLIC 
    ${PROJECT_SOURCE_DIR}
)

target_link_libraries(
  test_utils
  GTest::gtest_main
  utils
  ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(test_utils)