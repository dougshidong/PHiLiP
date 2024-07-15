set(TEST_SRC
    tke_spectra_calculation_fix.cpp
    )

foreach(dim RANGE 3 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_tke_spectra_calculation_fix)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    set(ParameterLib ParametersLibrary)
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT FlowSolverLib FlowSolver_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParameterLib})
    target_link_libraries(${TEST_TARGET} ${FlowSolverLib})
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    target_link_libraries(${TEST_TARGET} m mpi)
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    unset(TEST_TARGET)
    unset(ParametersLib)
    unset(FlowSolverLib)
    unset(PhysicsLib)

endforeach()
