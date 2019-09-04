# Discontinuous Galerkin Solver

[![Documentation](https://codedocs.xyz/dougshidong/PHiLiP.svg)](https://codedocs.xyz/dougshidong/PHiLiP/)


## Code Description
- Code uses deal.II library as the backbone (https://www.dealii.org/)
- Supported Partial Differential Equations: Convection-diffusion, Euler, TODO: Navier-Stokes.
- Supported convective numerical fluxes: Lax-Friedrichs, Roe (Harten's entropy fix) for Euler
- Supported diffusive numerical fluxes: Symmetric Interior Penalty
- Supported elements: LINEs, QUADs, HEXs
- Supported refinements: h (size) or p (order).

## Building/Running the Code

The code has been succesfully built in the following environments:
- linux (ubuntu 18.04 and later);

Please consult the [installation instructions](INSTALL.md) for details concerning required software.

### Build using CMake

This must be build out-of-source. An in-source build is likely to fail. For example, to configure for the debug build:
```sh
$ ROOT$ export PHILIP_DIR=/path_to_PHiLiP/
$ ROOT$ mkdir build_debug
$ ROOT$ cd build_debug
$ ROOT$ cmake -DDEAL_II_DIR=/path_to_dealii_install/ $PHILIP_DIR
```

### Compile using Make

Once the code has been successfully built, the available `make` targets can be seen using
```sh
ROOT$ make help
```

Of primary interest are the following:
```sh
ROOT$ make -j2     // Compile the entire code, 1D, 2D, and 3D
ROOT$ make -j2 doc // Generate the Doxygen documentation.
ROOT$ make -j2 1D  // Compile the 1D version of the code
ROOT$ make -j2 2D  // Compile the 2D version of the code
ROOT$ make -j2 3D  // Compile the 3D version of the code
```

Based on past experiences, you might want to limit the number of processor to 2 or 3 (make -j 3) if you have 8GB of RAM.

The html documentation can be accessed by pointing a browser at `ROOT/doc/html/index.html`. e.g. `google-chrome ROOT/doc/html/index.html`.

## Testing

Testing can be performed using CMake's `ctest` functionality. After successfully compiling the project, all tests can be
run by executing:
```sh
$ ROOT$ make test (which is equivalent to ROOT$ make test)
```

An alternative make target is provided to run tests with --output-on-failure:
```sh
ROOT$ make check
```

Additional useful commands are:
```sh
ROOT$ ctest -N (List the tests that would be run but not actually run them)
ROOT$ ctest -R <regex> (Run tests matching regular expression)
ROOT$ ctest -V (Enable verbose output from tests)
```

# License

The code is licensed under the [GNU LGPLv2.1](LICENSE.md) due to the dependence on the deal.II library.

