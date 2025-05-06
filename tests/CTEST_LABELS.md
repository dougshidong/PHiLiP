## Labels
To add labels use the custom ctest function, defined in ./tests/CMakeLists.txt,  `set_tests_labels`.
This function takes one input and as many arguments as needed.
The one input is the test name and the arguments are the labels wanted. See below for an example use.
~~~
ADD_TEST(NAME NNLS_multi_core
COMMAND mpirun -n ${MPIMAX} $<TARGET_FILE:Tests.exe> multiCore)
set_tests_labels(NNLS_multi_core    LINEAR_SOLVER
                                    PARALLEL
                                    QUICK
                                    UNIT_TEST)
~~~
To use a label to run a specific set of tests use the -L flag with ctest. Multiple -L flags are counted as an
`AND` statement. One may also use -LE to exclude certain tests from running.
~~~
ctest -L UNIT_TEST -L QUICK 
~~~
will run all quick unit tests.
These `-L` flags can also use regex. For example, you can use 
 ~~~
 ctest -L "NAVIER\\|LES\\|RANS" 
 ~~~
to run all tests with NAVIER or LES or RANS in the labels.
The `-LE` option will run all tests excluding the specified regex.


Test labels should be added for the following categories (USE ALL CAPS)
- DIRECTORY NAME
   - NNLS, TGV_SCALING, ETC
- Dimension
   - 1D, 2D, 3D
- Parallel vs Serial
   - PARALLEL, SERIAL
- PDE Type
   - EULER, NAVIER_STOKES, etc
- ODE Solver Type
   - RUNGE-KUTTA, IMPLICIT, etc
- DG Type
   - STRONG, STRONG-SPLIT, WEAK
- Quadrature Type
   - COLLOCATED, UNCOLLOCATED
- OTHER (if needed)
	- MEMORY_INTENSIVE, MANUFACTURED_SOLUTION, EXPECTED_FAILURE, CONVERGENCE, CURVILINEAR, LIMITER,
     ADJOINT, GMSH, ARTIFICIAL_VISCOSITY, MESH_ADAPTATION, RELAXATION, RESTART, LES,
     CONVECTIVE JACOBIAN
- Speed of Test
   - QUICK (<~10s), MODERATE(<~180s), LONG(<~1hr), EXTRA-LONG(>~1hr)
- Type of Test
   - UNIT_TEST, INTEGRATION_TEST

## Custom Functions
There are custom functions added that make it easier to run complex ctest label combinations. They do not require to be
targeted when compiling and will not generate a new script/executable. Below are a list of functions and a description 
of what they do
- `make VISCOUS_TESTS`
  - Runs all ctest viscous tests using `ctest -L "NAVIER\\|LES\\|RANS"` 
- `make INVISCID_TESTS`
  - Runs all ctest inviscid tests using `ctest -L EULER` 