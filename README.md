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
$ cmake .
$ make
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
| TARGET_ARCH=arch               | optimize for `arch`. ex. `armv7-a` on ARM.       |
| CMAKE_CXX_COMPILER=compiler    | use `compiler` to compile                        |

