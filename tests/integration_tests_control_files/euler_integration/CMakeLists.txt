add_subdirectory(naca0012)

set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Manufactured solutions
set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(1d_euler_laxfriedrichs_manufactured.prm 1d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_euler_laxfriedrichs_manufactured_collocated.prm 1d_euler_laxfriedrichs_manufactured_collocated.prm COPYONLY)
add_test(
 NAME 1D_EULER_LAXFRIEDRICHS_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_laxfriedrichs_manufactured_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured.prm 2d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME 2D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured_collocated.prm 2d_euler_laxfriedrichs_manufactured_collocated.prm COPYONLY)
add_test(
 NAME 2D_EULER_LAXFRIEDRICHS_COLLOCATED_MANUFACTURED_SOLUTION
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured_collocated.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured.prm 2d_euler_laxfriedrichs_manufactured.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_laxfriedrichs_manufactured_navah.prm 2d_euler_laxfriedrichs_manufactured_navah.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION_NAVAH_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_laxfriedrichs_manufactured_navah.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_euler_roe_manufactured.prm 1d_euler_roe_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_ROE_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(1d_euler_l2roe_manufactured.prm 1d_euler_l2roe_manufactured.prm COPYONLY)
add_test(
  NAME 1D_EULER_L2ROE_MANUFACTURED_SOLUTION_LONG
  COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_euler_l2roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_roe_manufactured.prm 2d_euler_roe_manufactured.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_ROE_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_roe_manufactured_navah.prm 2d_euler_roe_manufactured_navah.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_ROE_MANUFACTURED_SOLUTION_NAVAH_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_roe_manufactured_navah.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_l2roe_manufactured.prm 2d_euler_l2roe_manufactured.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_L2ROE_MANUFACTURED_SOLUTION_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_l2roe_manufactured.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_l2roe_manufactured_navah.prm 2d_euler_l2roe_manufactured_navah.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_L2ROE_MANUFACTURED_SOLUTION_NAVAH_MEDIUM
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_l2roe_manufactured_navah.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Exact solutions

configure_file(2d_euler_cylinder.prm 2d_euler_cylinder.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_CYLINDER_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_cylinder.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_gaussian_bump.prm 2d_euler_gaussian_bump.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

##Artificial dissipation tests
configure_file(2d_euler_gaussian_bump_with_artificial_dissipation_laplacian_residual_convergence_test.prm 2d_euler_gaussian_bump_with_artificial_dissipation_laplacian_residual_convergence_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_WITH_ARTIFICIAL_VISCOSITY_LAPLACIAN_RESIDUAL_CONVERGENCE
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_with_artificial_dissipation_laplacian_residual_convergence_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_gaussian_bump_with_artificial_dissipation_enthalpy_laplacian_residual_convergence_test.prm 2d_euler_gaussian_bump_with_artificial_dissipation_enthalpy_laplacian_residual_convergence_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_WITH_ARTIFICIAL_VISCOSITY_ENTHALPY_LAPLACIAN_RESIDUAL_CONVERGENCE
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_with_artificial_dissipation_enthalpy_laplacian_residual_convergence_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_gaussian_bump_with_artificial_dissipation_physical_residual_convergence_test.prm 2d_euler_gaussian_bump_with_artificial_dissipation_physical_residual_convergence_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_WITH_ARTIFICIAL_VISCOSITY_PHYSICAL_RESIDUAL_CONVERGENCE
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_with_artificial_dissipation_physical_residual_convergence_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_gaussian_bump_with_artificial_dissipation_laplacian_discontinuity_sensor_test.prm 2d_euler_gaussian_bump_with_artificial_dissipation_laplacian_discontinuity_sensor_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_WITH_ARTIFICIAL_VISCOSITY_LAPLACIAN_DISCONTINUITY_SENSOR
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_with_artificial_dissipation_laplacian_discontinuity_sensor_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_gaussian_bump_with_artificial_dissipation_enthalpy_laplacian_discontinuity_sensor_test.prm 2d_euler_gaussian_bump_with_artificial_dissipation_enthalpy_laplacian_discontinuity_sensor_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_WITH_ARTIFICIAL_VISCOSITY_ENTHALPY_LAPLACIAN_DISCONTINUITY_SENSOR
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_with_artificial_dissipation_enthalpy_laplacian_discontinuity_sensor_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_gaussian_bump_with_artificial_dissipation_physical_discontinuity_sensor_test.prm 2d_euler_gaussian_bump_with_artificial_dissipation_physical_discontinuity_sensor_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_WITH_ARTIFICIAL_VISCOSITY_PHYSICAL_DISCONTINUITY_SENSOR
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_with_artificial_dissipation_physical_discontinuity_sensor_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
configure_file(2d_euler_gaussian_bump_enthalpy_test.prm 2d_euler_gaussian_bump_enthalpy_test.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_ENTHALPY_TEST
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_enthalpy_test.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

# Mesh Adaptation test cases
configure_file(2d_euler_gaussian_bump_residual_mesh_adaptation.prm 2d_euler_gaussian_bump_residual_mesh_adaptation.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_RESIDUAL_MESH_ADAPTATION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_residual_mesh_adaptation.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
# Adjoint test cases

# configure_file(2d_euler_gaussian_bump_adjoint.prm 2d_euler_gaussian_bump_adjoint.prm COPYONLY)
# add_test(
#  NAME MPI_2D_EULER_INTEGRATION_GAUSSIAN_BUMP_ADJOINT_LONG
#  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_adjoint.prm
#  # for testing in serial
#  #COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_gaussian_bump_adjoint.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )

configure_file(2d_euler_cylinder_adjoint.prm 2d_euler_cylinder_adjoint.prm COPYONLY)
add_test(
  NAME MPI_2D_EULER_INTEGRATION_CYLINDER_ADJOINT_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_cylinder_adjoint.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


# Vortex test case takes wayyy too much time. It works, so uncomment below if you want to wait.
# configure_file(2d_euler_vortex.prm 2d_euler_vortex.prm COPYONLY)
# add_test(
#   NAME 2D_EULER_INTEGRATION_VORTEX
#   mpirun -n 1 COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_vortex.prm
#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
# )


#configure_file(2d_euler_entropy_waves.prm 2d_euler_entropy_waves.prm COPYONLY)
#add_test(
#  NAME 2D_EULER_INTEGRATION_ENTROPY_WAVES
#  mpirun -n 1 COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_entropy_waves.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)

#configure_file(3d_euler_laxfriedrichs_manufactured.prm 3d_euler_laxfriedrichs_manufactured.prm COPYONLY)
#add_test(
#  NAME 3D_EULER_LAXFRIEDRICHS_MANUFACTURED_SOLUTION
#  mpirun -n 1 COMMAND ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_euler_laxfriedrichs_manufactured.prm
#  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#)
