set(TEST_SRC
    functional_dw_finiteDifferences.cpp
    )

foreach(dim RANGE 2 3)
    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_functional_dw_finiteDifferences)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT FunctionalLib Functional_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    target_link_libraries(${TEST_TARGET} ${NumericalFluxLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${FunctionalLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(dim)
    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParametersLib)
    unset(DiscontinuousGalerkinLib)
    unset(FunctionalLib)
    unset(ODESolverLib)
endforeach()

set(TEST_SRC
    target_functional_dw_finiteDifferences.cpp
    )

foreach(dim RANGE 2 3)
    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_target_functional_dw_finiteDifferences)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParametersLib ParametersLibrary)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT FunctionalLib Functional_${dim}D)
    string(CONCAT ODESolverLib ODESolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParametersLib})
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    target_link_libraries(${TEST_TARGET} ${NumericalFluxLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${FunctionalLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(dim)
    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParametersLib)
    unset(DiscontinuousGalerkinLib)
    unset(FunctionalLib)
    unset(ODESolverLib)
endforeach()
