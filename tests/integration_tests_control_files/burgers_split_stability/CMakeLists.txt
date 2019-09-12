set(TEST_SRC
    burgers_stability.cpp
    )

foreach(dim RANGE 1 1)

    
    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_burgers_energy_stability)
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
    string(CONCAT ODESolverLib ODESolver_${dim}D)

    target_link_libraries(${TEST_TARGET} ${ParameterLib})
    target_link_libraries(${TEST_TARGET} ${PhysicsLib})
    target_link_libraries(${TEST_TARGET} ${NumericalFluxLib})
    target_link_libraries(${TEST_TARGET} ${DiscontinuousGalerkinLib})
    target_link_libraries(${TEST_TARGET} ${ODESolverLib})

    target_link_libraries(${TEST_TARGET} m mpi mpi_cxx)

    # Setup target with deal.II
    DEAL_II_SETUP_TARGET(${TEST_TARGET})
    
    configure_file(1D_burgers_stability.prm 1D_burgers_stability.prm COPYONLY)
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -np 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET} -i ${CMAKE_CURRENT_BINARY_DIR}/1D_burgers_stability.prm
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(dim)
    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParameterLib)
    unset(DiscontinuousGalerkinLib)
    unset(ODESolverLib)

endforeach()
