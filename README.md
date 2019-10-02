# Discontinuous Galerkin Solver

[![Documentation](https://codedocs.xyz/dougshidong/PHiLiP.svg)](https://codedocs.xyz/dougshidong/PHiLiP/)


## Code Description
- Code uses deal.II library as the backbone (https://www.dealii.org/)
- Math supporting this code can be viewed in this **very rough draft in progress** [Overleaf document](https://www.overleaf.com/read/mytvbbbbyqnj).
- Supports weak and strong (InProgress) form of discontinuous Galerkin (DG), and flux reconstruction (FR) (InProgress)
- Supported Partial Differential Equations: Linear advection, diffusion, convection-diffusion, Burgers, Euler, TODO: Navier-Stokes.
- Supported convective numerical fluxes: Lax-Friedrichs, Roe (Harten's entropy fix) for Euler, InProgress: Split-Form
- Supported diffusive numerical fluxes: Symmetric Interior Penalty
- Supported elements: LINEs, QUADs, HEXs since it uses deal.II
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
For example `--args /home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/bin/PHiLiP_2D "-i" "/home/ddong/Codes/PHiLiP_temp/PHiLiP/build_de    bug/tests/adv    ection_implicit/2d_advection_implicit_strong.prm`.


Additional useful commands are:
```sh
GDB$ break dg.cpp:89 (Add a breakpoint in a filename at a line number. Those breakpoints can be added before launching the program.)
GDB$ continue (Continue the program until the next breakpoint or to the end)
GDB$ step (Execute the next step of instructions. It will go into the functions being called)
GDB$ next (Execute the next line of code in the function. Will NOT go into the functions being called)
GDB$ quit
```

### Parallel debugging

If the error only occurs when using parallelism, you can use the following example command
```sh
mpirun -np 2 xterm -hold -e gdb -ex 'break MPI_Abort' -ex run --args /home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/bin/PHiLiP_2D "-i" "/home/ddong/Codes/PHiLiP_temp/PHiLiP/build_debug/tests/advection_implicit/2d_advection_implicit_strong.prm"
```
This launches 2 xterm processes, each of which will launch gdb processes that will run the code and will have a breakpoint when MPI_Abort is encountered.


# License

The code is licensed under the [GNU LGPLv2.1](LICENSE.md) due to the dependence on the deal.II library.

