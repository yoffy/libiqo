# libiqo

## About This

Image processing library for C++.

## Features

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

| option                         | description                 |
|--------------------------------|-----------------------------|
| WITH_OPENMP=ON                 | use OpenMP (default:OFF)    |
| CMAKE_CXX_COMPILER=compiler    | use `compiler` to compile   |

