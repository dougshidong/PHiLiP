set(TEST_SRC
    navier_stokes_convert_primitive_conservative.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_navier_stokes_convert_primitive_conservative)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    navier_stokes_manufactured_solution_source.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_navier_stokes_manufactured_solution_source)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    navier_stokes_convective_jacobian.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_navier_stokes_convective_jacobian)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    navier_stokes_convective_jacobian_via_dfad.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_navier_stokes_convective_jacobian_via_dfad)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    navier_stokes_viscosity_derivative_wrt_temperature.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_navier_stokes_viscosity_derivative_wrt_temperature)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

