## Function to set test labels
## Test labels should be added for the following categories (USE ALL CAPS)
## - Type of Test
#	- UNIT_TEST, INTEGRATION_TEST
#- Speed of Test
#	- QUICK, SLOW
#- Test Group
#	- NNLS, REDUCED ORDER, ETC
#- Usage (if needed)
#	- MEMORY_INTENSIVE
#- Dimension
#   - 1D, 2D, 3D
#- PDE Type
#   - EULER, NAVIER-STOKES
#- Quadrature Type
#   - COLOCATED, UNCOLOCATED
#- DG Type
#   - STRONG, WEAK
#- ODE Solver Type
#   - RUNGE-KUTTA, IMPLICIT, ETC
#- Parallel vs Serial
#   - PARALLEL, SERIAL
function(set_tests_labels test_name)
    set(labels ${ARGN})
    set_tests_properties(${test_name} PROPERTIES LABELS "${labels}")
endfunction()

add_subdirectory(unit_tests)
add_subdirectory(integration_tests_control_files)
add_subdirectory(meshes)
