# Discontinuous Galerkin Solver

[![Documentation](https://codedocs.xyz/dougshidong/PHiLiP.svg)](https://codedocs.xyz/dougshidong/PHiLiP/)


## Code Description
- Code uses deal.II library as the backbone (https://www.dealii.org/)
- Physics implementation and finite element operator will follow Philip Zwanenburg's derivations and snippets of DPGSolver (https://github.com/PhilipZwanenburg/DPGSolver)
- Methods to be implemented
	- Discontinuous Galerkin (DG);
- Supported Partial Differential Equations: Advection, (TODO): Diffusion, Euler, Navier-Stokes.
- Supported elements: LINEs, QUADs, HEXs
- (TODO) Supported refinements: isotropic h (size) or p (order).

[//]: # (**It is recommended** to follow the [Coding Style Guidelines](STYLE.md) when making modifications to the code.)

## Building/Running the Code

The code has been succesfully built in the following environments:
- linux (ubuntu 18.04 and later);

Please consult the [installation instructions](INSTALL.md) for details concerning required software.

### Build using CMake

An out-of-source build must be performed using the [sample scripts](cmake/run) by executing the
appropriate bash script. For example, to configure for the debug build:
```sh
$ ROOT$ mkdir build
$ ROOT$ cd build
$ ROOT$ cmake ../
```

### Compile using Make

Once the code has been successfully built, the available `make` targets can be seen using
```sh
ROOT$ make help
```

Of primary interest are the following:
```sh
ROOT$ make -j     // Compile the code.
ROOT$ make -j doc // Generate the Doxygen documentation.
```

Based on past experiences, you might want to limit the number of processor to 2 or 3 (make -j 3) if you have 8GB of RAM.

The html documentation can be accessed by pointing a browser at `ROOT/doc/html/index.html`.

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

