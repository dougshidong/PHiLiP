set(TEST_SRC
    euler_bump_residual_assembly.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_euler_bump_residual_assembly)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    target_link_libraries(${TEST_TARGET} PHiLiP_${dim}D_lib)
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    configure_file(${dim}d_euler_gaussian_bump.prm ${dim}d_euler_gaussian_bump.prm COPYONLY)
    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -np 3 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET} -i ${CMAKE_CURRENT_BINARY_DIR}/${dim}d_euler_gaussian_bump.prm
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)

endforeach()
