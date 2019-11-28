set(TEST_SRC
    dRdX_fd_vs_ad.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_dRdX_fd_vs_ad)
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
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else()
        set(NMPI ${MPIMAX})
    endif()
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    compare_rhs.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_compare_rhs)
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
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else ()
        set(NMPI ${MPIMAX})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParametersLib)

endforeach()

set(TEST_SRC
    check_symmetric_hessian.cpp
    )

foreach(dim RANGE 1 2)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_check_symmetric_hessian)
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
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    if (dim EQUAL 1) 
        set(NMPI 1)
    else ()
        set(NMPI ${MPIMAX})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${NMPI} ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(ParametersLib)

endforeach()
