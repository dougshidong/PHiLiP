set(TEST_SRC
    euler_convert_primitive_conservative.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_euler_convert_primitive_conservative)
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
    set_tests_labels(${TEST_TARGET} EULER_UNIT_TEST
                                    ${dim}D
                                    SERIAL
                                    QUICK
                                    UNIT_TEST)

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    euler_manufactured_solution_source.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_euler_manufactured_solution_source)
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
    set_tests_labels(${TEST_TARGET} EULER_UNIT_TEST
                                    ${dim}D
                                    SERIAL
                                    MANUFACTURED_SOLUTION
                                    QUICK
                                    UNIT_TEST)

    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    euler_convective_jacobian.cpp
    )

foreach(dim RANGE 1 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_euler_convective_jacobian)
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
    set_tests_labels(${TEST_TARGET} EULER_UNIT_TEST
                                    ${dim}D
                                    SERIAL
                                    CONVECTIVE_JACOBIAN
                                    QUICK
                                    UNIT_TEST)
    unset(TEST_TARGET)
    unset(PhysicsLib)

endforeach()

set(TEST_SRC
    freestream_preservation.cpp
    )

foreach(dim RANGE 2 3)

    # Output executable
    string(CONCAT TEST_TARGET ${dim}D_euler_freestream_preservation)
    message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
    add_executable(${TEST_TARGET} ${TEST_SRC})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

    # Compile this executable when 'make unit_tests'
    add_dependencies(unit_tests ${TEST_TARGET})
    add_dependencies(${dim}D ${TEST_TARGET})

    # Library dependency
    target_link_libraries(${TEST_TARGET} ParametersLibrary)
    target_link_libraries(${TEST_TARGET} Grids_${dim}D)
    target_link_libraries(${TEST_TARGET} Physics_${dim}D)
    target_link_libraries(${TEST_TARGET} InitialConditions_${dim}D)
    target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TEST_TARGET} ODESolver_${dim}D)
    # Setup target with deal.II
    if (NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${TEST_TARGET})
    endif()

    add_test(
      NAME ${TEST_TARGET}
      COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )
    set_tests_labels(${TEST_TARGET} EULER_UNIT_TEST
                                    ${dim}D
                                    SERIAL
                                    MODERATE
                                    UNIT_TEST)

    unset(TEST_TARGET)

endforeach()

# set(TEST_SRC
#     3D_gmsh_euler_test.cpp
#     )

# foreach(dim 3)

#     # Output executable
#     string(CONCAT TEST_TARGET 3D_gmsh_euler_test)
#     message("Adding executable " ${TEST_TARGET} " with files " ${TEST_SRC} "\n")
#     add_executable(${TEST_TARGET} ${TEST_SRC})
#     # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
#     target_compile_definitions(${TEST_TARGET} PRIVATE PHILIP_DIM=${dim})

#     # Compile this executable when 'make unit_tests'
#     add_dependencies(unit_tests ${TEST_TARGET})
#     add_dependencies(${dim}D ${TEST_TARGET})

#     # Library dependency
#     target_link_libraries(${TEST_TARGET} ParametersLibrary)
#     target_link_libraries(${TEST_TARGET} Grids_${dim}D)
#     target_link_libraries(${TEST_TARGET} Physics_${dim}D)
#     target_link_libraries(${TEST_TARGET} InitialConditions_${dim}D)
#     target_link_libraries(${TEST_TARGET} DiscontinuousGalerkin_${dim}D)
#     target_link_libraries(${TEST_TARGET} ODESolver_${dim}D)
#     # Setup target with deal.II
#     if (NOT DOC_ONLY)
#         DEAL_II_SETUP_TARGET(${TEST_TARGET})
#     endif()

#     set (filename "NACA0012_Coarse_Slip_Wall.msh")
#     if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
#     message(SEND_ERROR
#     "Missing 3D GMSH file named ${filename}. Please download it from
#         https://drive.google.com/drive/folders/1xaQRTCd8VXDeaV1alQ5oHI_YMm2X0X5R?usp=sharing
#     and place them in
#         ${CMAKE_CURRENT_SOURCE_DIR}, or simply run get_gmsh_mesh_files_local.sh"
#         )
#     endif()
#     configure_file(${filename} ${filename} COPYONLY)

#     add_test(
#       NAME ${TEST_TARGET}
#       COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/${TEST_TARGET}
#       WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
#     )

#     unset(TEST_TARGET)

# endforeach()
