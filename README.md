# Discontinuous Galerkin Solver


## Code Description
- Code uses deal.II library as the backbone (https://www.dealii.org/)
- Physics implementation and finite element operator will follow Philip Zwanenburg's derivations and snippets of DPGSolver (https://github.com/PhilipZwanenburg/DPGSolver)
- Methods to be implemented
	- Discontinuous Galerkin (DG);
- Supported Partial Differential Equations: Advection, (TODO): Diffusion, Euler, Navier-Stokes.
- Supported elements: LINEs, QUADs, HEXs
- (TODO) Supported refinements: isotropic h (size) or p (order).

**It is recommended** to follow the [Coding Style Guidelines](STYLE.md) when making modifications to
the code.

## Building/Running the Code

The code has been succesfully built in the following environments:
- linux (ubuntu 18.04 and later);

Please consult the [installation instructions](INSTALL.md) for details concerning required software.

### Build using CMake

An out-of-source build must be performed using the [sample scripts](cmake/run) by executing the
appropriate bash script. For example, to configure for the debug build:
```sh
$ ROOT$ cmake .
$ ROOT$ make
$ ROOT$ make test
```

### Compile using Make

Once the code has been successfully built, the available `make` targets can be seen using
```sh
BUILD$ make help
```

Of primary interest are the following:
```sh
BUILD$ make -j     // Compile the code.
BUILD$ make meshes // Generate the meshes.
BUILD$ make -j doc // Generate the Doxygen documentation.
```

The html documentation can be accessed by pointing a browser at `BUILD/doc/html/index.html`.

### Running the Code

Executable files running various configurations of the code are placed in `BUILD/bin`, and should be executed following
the example in `$BUILD/script_files/quick.sh`. To run an executable with valgrind's memory leak detector enabled:
```sh
BUILD/script_files$ ./memcheck.sh
```

## Testing

**Note: In order to run all of the available tests, the code must be compiled in all of the dimension-dependent build
directories.**

Testing can be performed using CMake's `ctest` functionality. After successfully compiling the project, all tests can be
run by executing:
```sh
BUILD$ ctest (which is equivalent to BUILD$ make test)
```

An alternative make target is provided to run tests with --output-on-failure:
```sh
BUILD$ make check
```

All tests should be passing other than those listed in the [documented failing tests file](FAILING_TESTS.md). Additional
useful commands are:
```sh
BUILD$ ctest -N (List the tests that would be run but not actually run them)
BUILD$ ctest -R <regex> (Run tests matching regular expression)
BUILD$ ctest -V (Enable verbose output from tests)
```

## Contributors

- Manmeet Bhabra
- Cem Gormezano
- Siva Nadarajah, siva.nadarajah (at) mcgill.ca
- Doug Shi-Dong
- Philip Zwanenburg, philip.zwanenburg (at) mail.mcgill.ca

If you would like to make your own contributions to the project, the best place to start is the
[getting started page](https://codedocs.xyz/PhilipZwanenburg/DPGSolver/md_doc_GETTING_STARTED.html). It is highly
recommended to use Ctags to enable easy navigation through the code. Please see the
[Code Navigation using Ctags README](external/ctags/README.md) for further details.

# License

The code is licensed under the [GNU LGPLv2.1](LICENSE.md) due to the dependence on the deal.II library.

