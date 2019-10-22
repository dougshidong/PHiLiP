set(TEST_SRC
    high_order_grid_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_HighOrder_MappingFEField)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(DiscontinuousGalerkinLib)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    RBF_mesh_movement.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_RBF_mesh_movement)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(DiscontinuousGalerkinLib)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    make_cells_valid.cpp
    )

set(dim 2)
# Output executable
string(CONCAT TEST_TARGET ${dim}D_make_cells_valid)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
# Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

# Library dependency
set(ParametersLib ParametersLibrary)
string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
target_link_libraries(${TEST_TARGET} ${ParametersLib})
target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
# Setup target with deal.II
if (NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

add_test(
  NAME ${TEST_TARGET}
  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

unset(TEST_TARGET)
unset(DiscontinuousGalerkinLib)
unset(ParametersLib)

