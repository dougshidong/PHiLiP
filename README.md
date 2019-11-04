# Parallel High-Order Library for PDEs

[![Documentation](https://codedocs.xyz/dougshidong/PHiLiP.svg)](https://codedocs.xyz/dougshidong/PHiLiP/)

The [P]arallel [Hi]gh-Order [Li]brary for [P]DEs (PHiLiP) uses Discontinuous Galerkin methods to solve convection-diffusion problems.

The math supporting this code can be viewed in this **very rough draft in progress** [Overleaf document](https://www.overleaf.com/read/mytvbbbbyqnj).

## Code Description
- Code uses deal.II library as the backbone (https://www.dealii.org/)
- Parallelized through MPI
- Supports weak and strong (InProgress) form of discontinuous Galerkin (DG), and flux reconstruction (FR) (InProgress)
- Supported Partial Differential Equations: Linear advection, diffusion, convection-diffusion, Burgers, Euler, TODO: Navier-Stokes.
- Supported convective numerical fluxes: Lax-Friedrichs, Roe (Harten's entropy fix) for Euler, InProgress: Split-Form
- Supported diffusive numerical fluxes: Symmetric Interior Penalty
- Supported elements: LINEs, QUADs, HEXs since it uses deal.II
- Supported refinements: h (size) or p (order).

## Documentation

The code itself is documented using Doxygen, and the latest documentation is hosted on [codedocs.xyz](https://codedocs.xyz/dougshidong/PHiLiP/). 

Since deal.II is heavily used, their [documentation](https://www.dealii.org/developer/doxygen/deal.II/index.html) is probably the most useful.

Another great ressource is the [deal.II Google Groups](https://groups.google.com/forum/#!forum/dealii), where developers are actively answering questions.

Finally, I am also always available to answer questions regarding the code by e-mail at doug.shi-dong@mail.mcgill.ca

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

A list of currently known failing tests is kept in the [GitHub issues](https://github.com/dougshidong/PHiLiP/issues?q=is%3Aissue+is%3Aopen+label%3Atestfail) with `testfail` tags.

Testing can be performed using CMake's `ctest` functionality. After successfully compiling the project, all tests can be
run by executing:
```sh
$ ROOT$ ctest (which is equivalent to ROOT$ make test)
```

An alternative make target is provided to run tests with --output-on-failure:
```sh
ROOT$ make check
```

Additional useful commands are:
```sh
ROOT$ ctest -N (List the tests that would be run but not actually run them)
ROOT$ ctest -R <regex> (Run tests matching regular expression)
ROOT$ ctest -E <regex> (Exclude tests matching regular expression)
ROOT$ ctest -V (Enable verbose output from tests)
```
Note that running `ctest` in `Debug` will take forever since some integration tests fully solve nonlinear problems with multiple orders and multiple meshes. It is suggested to perform `ctest` in `Release` mode, and only use `Debug` mode for debugging purposes.

## Debugging

Here is a quickstart guide to debugging. It is highly suggested to use gdb and/or valgrind when the program crashes unexpectedly.
The first step is to compile the program in `DEBUG` mode through `CMAKE_BUILD_TYPE=Debug`.

If ctest fails, using `ctest -V -R failing_test_name` will show the command being run.

For a serial run, you may simply use gdb as intended
```sh
ROOT$ gdb --args commmand_to_launch_test 
GDB$ run (Executes the program. Can re-launch the program if you forgot to put breakpoints.)
```
For example 
```
gdb --args /home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/bin/PHiLiP_2D "-i" "/home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/tests/advection_implicit/2d_advection_implicit_strong.prm
```


Additional useful commands are:
```sh
GDB$ break dg.cpp:89 (Add a breakpoint in a filename at a line number. Those breakpoints can be added before launching the program.)
GDB$ continue (Continue the program until the next breakpoint or to the end)
GDB$ step (Execute the next step of instructions. It will go into the functions being called)
GDB$ next (Execute the next line of code in the function. Will NOT go into the functions being called)
GDB$ quit
```

### Memory

Memory leaks can be detected using Valgrind's tool `memcheck`. The application must be compiled in `Debug` mode. For example

```
valgrind --leak-check=full --track-origins=yes /home/ddong/Codes/PHiLiP/build_debug/bin/2D_HighOrder_MappingFEField
```

### Parallel debugging

If the error only occurs when using parallelism, you can use the following example command
```sh
mpirun -np 2 xterm -hold -e gdb -ex 'break MPI_Abort' -ex run --args /home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/bin/PHiLiP_2D "-i" "/home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/tests/advection_implicit/2d_advection_implicit_strong.prm"
```
This launches 2 xterm processes, each of which will launch gdb processes that will run the code and will have a breakpoint when MPI_Abort is encountered.

## Performance

Problems tend to show up in the 3D version if an algorithm has been implemented inefficiently. It is therefore highly recommended that a 3D test accompanies the implemented features.

### Computational

Computational bottlenecks can be inspected using Valgrind's tool `callgrind`. It is used as such

```
valgrind --tool=callgrind /home/ddong/Codes/PHiLiP/build_release/bin/2D_RBF_mesh_movement
```

This will result in a `callgrind.out.#####`. A visualizer such as `kcachegrind` (available through `apt`) can then be used to sort through the results. For example:

```kcachegrind callgrind.out.24250```

### Memory

Apart from memory leaks, it is possible that some required allocations demand too much memory. Valgrind also has a tool for this called `massif`. For example

```valgrind --tool=massif /home/ddong/Codes/PHiLiP/build_debug/bin/3D_RBF_mesh_movement```

will generate a `massif.out.#####` file that can be visualized using `massif-visualizer` (available through `apt`) as

```massif-visualizer massif.out.18580```


## Contributing checklist

1. A unit test, integration test, or regression test accompanies the feature. Tests longer than a few seconds should be tagged as with the suffix MEDIUM, and tests a minute or longer should be tagged with LONG.
  * A unit test is often most appropriate, and is aimed at testing a single component of the code. See the test on [Euler's primitive to conservative conversion](https://github.com/dougshidong/PHiLiP/blob/master/tests/unit_tests/euler_unit_test/euler_convert_primitive_conservative.cpp)
  * An integration test runs the entire main program by taking an input file and calling PHiLiP_1/2/3D. It should be derived from the [`TestBase` class](https://github.com/dougshidong/PHiLiP/blob/master/src/testing/tests.h), and have a control file located in the [integration test directory](https://github.com/dougshidong/PHiLiP/tree/master/tests/integration_tests_control_files). Since integrations tests uses multiple components, they usually take longer. Furthermore, the cause of failure is sometimes less obvious. A good suggestion is to use an existing test control file, and only change 1 parameter to help pinpoint issues when it fails.
   * A regression test stores previously computed data to validate future results. Note that this type of test is rarely appropriate since valid changes in the code can fail this type of test. If implemented, a script/code should be made available such that newly computed results can replace the old results. See [file1](https://github.com/dougshidong/PHiLiP/blob/master/tests/unit_tests/regression/jacobian_matrix_regression.cpp) and [file2](https://github.com/dougshidong/PHiLiP/blob/master/tests/unit_tests/regression/matrix_data/copy_matrices.sh)
2. The feature has been documented.
  * Function and member variable documentation should be presented in the associated header file. `make doc` should generate a html file in the `/path_to_build/doc/html/index.html` that can be opened used your browser of choice.
  * Comments in the code as appropriate.
3. The `master` branch of `https://github.com/dougshidong/PHiLiP` has been merged into your fork and merge conflicts have been resolved.
4. The entire `ctest` suite has been run in `Release` mode and the short/medium length tests have been run in `Debug` mode (using `ctest -E LONG`). Please save the log for the next point.
5. Submit a pull request with a log of the tests.

# License

The code is licensed under the [GNU LGPLv2.1](LICENSE.md) due to the dependence on the deal.II library.

