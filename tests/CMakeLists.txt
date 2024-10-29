## Function to set test labels
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
#   - QUICK (<~10s), MODERATE(<~180s), LONG(>~180s)
#- Type of Test
#   - UNIT_TEST, INTEGRATION_TEST

function(set_tests_labels test_name)
    set(labels ${ARGN})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

add_subdirectory(unit_tests)
add_subdirectory(integration_tests_control_files)
add_subdirectory(meshes)
