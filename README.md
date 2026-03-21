# Simple-mpc

**Simple-mpc** is a C++ implementation of multiple predictive control schemes for locomotion based on the Aligator optimization solver.

It can be used with quadrupeds and bipeds to generate whole-body walking motions based on a pre-defined contact plan.

## Features

The **Simple-mpc** library provides:

* an interface to generate different locomotion gaits in a MPC-like fashion
* Python bindings to enable fast prototyping
* three different kinds of locomotion dynamics (centroidal, kinodynamics and full dynamics)

## Installation

### Build from source (local conda workflow)
0. Install Miniconda or Anaconda.

1. Clone repo.
```bash
git clone git@github.com:Simple-Robotics/simple-mpc.git
cd simple-mpc
```

2. Create and activate the development environment.
```bash
conda env create -f environment.yml
conda activate simple-mpc
```

3. Configure and build.
```bash
cmake -G Ninja -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_INTERFACE=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_BENCHMARK=OFF
cmake --build build
```

4. Run an example.
```bash
PYTHONPATH=bindings python examples/go2_kinodynamics.py
```

`ndcurves` is expected to be installed in the active conda environment.

Enable `-DBUILD_TESTING=ON` to build tests and `-DBUILD_BENCHMARK=ON` to build benchmarks.
