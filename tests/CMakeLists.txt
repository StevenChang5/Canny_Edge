include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

include_directories(${OpenCV_INCLUDE_DIRECTORIES})

add_subdirectory(utils)

if(ENABLE_CUDA)
  add_subdirectory(cuda)
endif()