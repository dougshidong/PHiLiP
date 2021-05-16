add_subdirectory(gmsh_reader)

set(TEST_SRC
    ffd_test.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_FFD)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})

    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT GridsLib Grids_${dim}D)
    target_link_libraries(${TEST_TARGET} ${GridsLib})
    unset(GridsLib)

    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
    unset(HighOrderGridLib)

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

endforeach()

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
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
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
    unset(HighOrderGridLib)

endforeach()

# FFD deformation test
set(TEST_SRC
    ffd_deform_mesh.cpp
    )

set(dim 2)
# Output executable
string(CONCAT TEST_TARGET ${dim}D_FFD_deform_mesh)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
# Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

# Library dependency
string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
# Setup target with deal.II
if (NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

set (NMPI ${MPIMAX})
set (LENGTH SHORT)

add_test(
  NAME ${TEST_TARGET}_${LENGTH}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)
unset(HighOrderGridLib)

# FFD deformation test
set(TEST_SRC
    ffd_linear.cpp
    )

set(dim 2)
# Output executable
string(CONCAT TEST_TARGET ${dim}D_FFD_linear_sensitivity)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
# Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

# Library dependency
string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
# Setup target with deal.II
if (NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

set (NMPI ${MPIMAX})
set (LENGTH SHORT)

add_test(
  NAME ${TEST_TARGET}_${LENGTH}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)
unset(HighOrderGridLib)

# FFD deformation test
set(TEST_SRC
    ffd_sensitivity.cpp
    )

set(dim 2)
# Output executable
string(CONCAT TEST_TARGET ${dim}D_FFD_sensitivity)
message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
add_executable(${TEST_TARGET} ${TEST_SRC})
# Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

# Compile this executable when 'make unit_tests'
add_dependencies(unit_tests ${TEST_TARGET})
add_dependencies(${dim}D ${TEST_TARGET})

# Library dependency
string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
# Setup target with deal.II
if (NOT DOC_ONLY)
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
endif()

set (NMPI ${MPIMAX})
set (LENGTH SHORT)

add_test(
  NAME ${TEST_TARGET}_${LENGTH}
  COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)
unset(TEST_TARGET)
unset(HighOrderGridLib)


# Test linear elasticity mesh movement
set(TEST_SRC
    LinearElasticity_mesh_movement.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_LinearElasticity_mesh_movement)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
    target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1)
        set (NMPI 1)
    else()
        set (NMPI ${MPIMAX})
    endif()

    if (dim EQUAL 3)
        set (LENGTH LONG)
    else()
        set (LENGTH SHORT)
    endif()

    add_test(
      NAME ${TEST_TARGET}_${LENGTH}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(HighOrderGridLib)

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
string(CONCAT HighOrderGridLib HighOrderGrid_${dim}D)
target_link_libraries(${TEST_TARGET} ${HighOrderGridLib})
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
unset(HighOrderGridLib)


unset(ParametersLib)
