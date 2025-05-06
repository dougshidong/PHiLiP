## Function to set test labels
## How to add labels
# To add labels use the custom ctest function, defined in ./tests/CMakeLists.txt,  `set_tests_labels`.
# This function takes one input and as many arguments as needed.
# The one input is the test name and the arguments are the labels wanted. See below for an example use.
# ADD_TEST(NAME NNLS_multi_core
# COMMAND mpirun -n ${MPIMAX} $<TARGET_FILE:Tests.exe> multiCore)
# set_tests_labels(NNLS_multi_core    LINEAR_SOLVER
#                                     PARALLEL
#                                     QUICK
#                                     UNIT_TEST)

## How to use labels
# To use a label to run a specific set of tests use the -L flag with ctest. Multiple -L flags are counted as an
# `AND` statement. One may also use -LE to exclude certain tests from running.
# ctest -L UNIT_TEST -L QUICK will run all quick unit tests.
# These `-L` flags can also use regex. For example, you can use ctest -L "NAVIER\\|LES\\|RANS" to run all
# tests with NAVIER or LES or RANS in the labels.

## Test labels should be added for the following categories (USE ALL CAPS)
#- DIRECTORY NAME
#   - NNLS, TGV_SCALING, ETC
#- Dimension
#   - 1D, 2D, 3D
#- Parallel vs Serial
#   - PARALLEL, SERIAL
#- PDE Type
#   - EULER, NAVIER_STOKES, etc
#- ODE Solver Type
#   - RUNGE-KUTTA, IMPLICIT, etc
#- DG Type
#   - STRONG, STRONG-SPLIT, WEAK
#- Quadrature Type
#   - COLLOCATED, UNCOLLOCATED
#- OTHER (if needed)
#	- MEMORY_INTENSIVE, MANUFACTURED_SOLUTION, EXPECTED_FAILURE, CONVERGENCE, CURVILINEAR, LIMITER,
#     ADJOINT, GMSH, ARTIFICIAL_VISCOSITY, MESH_ADAPTATION, RELAXATION, RESTART, LES,
#     CONVECTIVE JACOBIAN
#- Speed of Test
#   - QUICK (<~10s), MODERATE(<~180s), LONG(<~1hr), EXTRA-LONG(>~1hr)
#- Type of Test
#   - UNIT_TEST, INTEGRATION_TEST

function(set_tests_labels test_name)
    set(labels ${ARGN})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

## CUSTOM CMAKE FUNCTION TO RUN VISCOUS TESTS
add_custom_target(VISCOUS_TESTS
COMMAND ${CMAKE_CTEST_COMMAND} -L "NAVIER\\|LES\\|RANS" #| is a special character, \\ is the escape character
COMMENT "Running tests labeled as NAVIER_STOKES, LES, or RANS"
)

## CUSTOM CMAKE FUNCTION TO RUN INVISCID TESTS
add_custom_target(INVISCID_TESTS
        COMMAND ${CMAKE_CTEST_COMMAND} -L "EULER" #| is a special character, \\ is the escape character
        COMMENT "Running tests labeled as EULER"
)

add_subdirectory(unit_tests)
add_subdirectory(integration_tests_control_files)
add_subdirectory(meshes)
