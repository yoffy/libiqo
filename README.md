# libiqo

## About This

Image processing library for C++.

## Features

* Intel SSE4.1, AVX2, AVX512, and ARM NEON implementations.
* Area image resize
  * only down sampling
  * (implemented only one-channel U8 image)
* Lanczos image resize
  * up sampling, and down sampling
  * (implemented only one-channel U8 image)

## Requirements

* C++98 (C++03) compiler

## Build

Run `cmake` and `make` in this directory.

```
$ cmake -D CMAKE_BUILD_TYPE=Release .
$ make
```

or

```
C:\libiqo> cmake -G "Visual Studio 15 2017 Win64" .
C:\libiqo> cmake --build . --config Release
```

It outputs the library into `lib` directory.

### make Options

| option                         | description                 |
|--------------------------------|-----------------------------|
| VERBOSE=1                      | print build commands        |

### CMake Options

| option                         | description                                      |
|--------------------------------|--------------------------------------------------|
| WITH_OPENMP=ON                 | use OpenMP multi threading library (default:OFF) |
| CMAKE_CXX_COMPILER=compiler    | use `compiler` to compile                        |
| CMAKE_BUILD_TYPE=type          | Debug or Release                                 |
