set(TEST_SRC
    numerical_flux_conservation.cpp
    )

foreach(dim RANGE 1 3)

    
    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_numerical_flux_conservation)
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
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ${ParameterLib})
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    target_link_libraries(${TEST_TARGET} ${NumericalFluxLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(dim)
    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParameterLib)
    unset(DiscontinuousGalerkinLib)

endforeach()
