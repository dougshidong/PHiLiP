set(TEST_SRC
    gmsh_reader.cpp
    )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
foreach(dim RANGE 2 3)
    message(STATUS "Generating ${dim}D Gmsh grids")
    string(CONCAT GMSH_GEO ${dim}D_square.geo)
    configure_file(${GMSH_GEO} ${GMSH_GEO} COPYONLY)

    execute_process(COMMAND gmsh ${GMSH_GEO} -
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GMSH_RESULT
                    OUTPUT_QUIET)
    if(NOT GMSH_RESULT EQUAL "0")
        message(FATAL_ERROR
                "gmsh ${GMSH_RESULT}, please check ${GMSH_GEO} file")
    endif()

    string(CONCAT GMSH_MSH ${dim}D_square.msh)
    configure_file(${GMSH_MSH} ${GMSH_MSH} COPYONLY)
endforeach()

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_GMSH_READER)
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
    unset(HighOrderGridLib)

    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n ${MPIMAX} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TEST_TARGET)

endforeach()

