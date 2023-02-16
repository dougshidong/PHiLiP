set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_advection_explicit_periodic_energy.prm 2D_advection_explicit_periodic_energy.prm COPYONLY)
add_test(
 NAME MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_explicit_periodic_energy.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_advection_explicit_periodic_OOA.prm 2D_advection_explicit_periodic_OOA.prm COPYONLY)
add_test(
 NAME MPI_2D_ADVECTION_EXPLICIT_PERIODIC_OOA_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_explicit_periodic_OOA.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_advection_explicit_periodic_energy_weight_adjusted.prm 2D_advection_explicit_periodic_energy_weight_adjusted.prm COPYONLY)
add_test(
 NAME MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_WEIGHT_ADJUSTED_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_explicit_periodic_energy_weight_adjusted.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(2D_advection_explicit_periodic_energy_weight_adjusted_on_the_fly.prm 2D_advection_explicit_periodic_energy_weight_adjusted_on_the_fly.prm COPYONLY)
add_test(
 NAME MPI_2D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_WEIGHT_ADJUSTED_ON_THE_FLY_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_advection_explicit_periodic_energy_weight_adjusted_on_the_fly.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
configure_file(3D_advection_explicit_periodic_energy_weight_adjusted.prm 3D_advection_explicit_periodic_energy_weight_adjusted.prm COPYONLY)
add_test(
 NAME MPI_3D_ADVECTION_EXPLICIT_PERIODIC_ENERGY_WEIGHT_ADJUSTED_LONG
COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3D_advection_explicit_periodic_energy_weight_adjusted.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
