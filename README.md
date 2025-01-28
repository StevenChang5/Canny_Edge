# Canny_Edge
Canny edge detection on Raspberry Pi

## Usage
To run the program on Raspberry Pi, compile using the command:

```bash
g++ main.cpp -o main -I. `pkg-config --cflags --libs opencv4`
```

To run tests,
```bash
cmake -S . -B build
cmake --build build
cd build && ctest
```