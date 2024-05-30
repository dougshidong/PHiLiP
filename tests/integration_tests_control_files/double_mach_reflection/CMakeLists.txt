set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

# =======================================
# 1D Sod Shock Tube test
# =======================================
configure_file(2D_double_mach_reflection.prm 2D_double_mach_reflection.prm COPYONLY)
add_test(
  NAME 2D_DOUBLE_MACH_REFLECTION
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_double_mach_reflection.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2D_double_mach_reflection_gmsh.prm 2D_double_mach_reflection_gmsh.prm COPYONLY)
add_test(
  NAME 2D_DOUBLE_MACH_REFLECTION_GMSH
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2D_double_mach_reflection_gmsh.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
