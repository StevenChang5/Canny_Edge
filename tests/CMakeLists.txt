include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

include_directories(${OpenCV_INCLUDE_DIRECTORIES})

add_executable(
  test_utils
  test_utils.cpp
)

target_include_directories(
    test_utils 
    PUBLIC 
    ${PROJECT_SOURCE_DIR}
)

add_library(
    utils ../src/utils.cpp
)

target_link_libraries(
  test_utils
  GTest::gtest_main
  utils
  ${OpenCV_LIBS}
)

include(GoogleTest)
gtest_discover_tests(test_utils)