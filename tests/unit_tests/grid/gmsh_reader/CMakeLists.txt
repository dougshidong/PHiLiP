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

    unset(TEST_TARGET)

endforeach()

foreach(dim RANGE 2 3)
    add_test(
      NAME ${dim}D_GMSH_READER_SQUARE
      COMMAND mpirun -n ${MPIMAX} ${CMAKE_CURRENT_BINARY_DIR}/${dim}D_GMSH_READER --input=${dim}D_square.msh
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )
endforeach()

set (filename "airfoil.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(SEND_ERROR
"Missing NACA0012 mesh file named ${filename}. Please download them from
    https://drive.google.com/drive/folders/1EeN4ooMK2awre2O_WBWJNMyO4FGDX_PN?usp=sharing
and place them in
      ${CMAKE_CURRENT_SOURCE_DIR}, or simply run get_gmsh_mesh_files_local.sh"
      )
endif()
configure_file(${filename} ${filename} COPYONLY)
add_test(
  NAME 2D_GMSH_READER_NACA0012
  COMMAND mpirun -n ${MPIMAX} ${CMAKE_CURRENT_BINARY_DIR}/2D_GMSH_READER --input=${filename}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set (filename "SD7003.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(SEND_ERROR
"Missing 3D GMSH file named ${filename}. Please download them from
    https://drive.google.com/drive/folders/1ZjwSvbrRK5pHvMULfC69n6oierTej2Cv?usp=sharing
and place them in
      ${CMAKE_CURRENT_SOURCE_DIR}, or simply run get_gmsh_mesh_files_local.sh"
      )
endif()
configure_file(${filename} ${filename} COPYONLY)
add_test(
  NAME 3D_GMSH_READER_SD7003
  COMMAND mpirun -n ${MPIMAX} ${CMAKE_CURRENT_BINARY_DIR}/3D_GMSH_READER --input=${filename}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set (filename "naca0012_hopw_ref2.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(SEND_ERROR
"Missing NACA0012 mesh file named ${filename}. Please download them from
    https://drive.google.com/drive/folders/1EeN4ooMK2awre2O_WBWJNMyO4FGDX_PN?usp=sharing
and place them in
      ${CMAKE_CURRENT_SOURCE_DIR}, or simply run get_gmsh_mesh_files_local.sh"
      )
endif()
configure_file(${filename} ${filename} COPYONLY)
add_test(
  NAME 2D_GMSH_READER_NACA0012_HOPW
  COMMAND mpirun -n ${MPIMAX} ${CMAKE_CURRENT_BINARY_DIR}/2D_GMSH_READER --input=${filename}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

set (filename "3D_CUBE_2ndOrder.msh")
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${filename})
  message(SEND_ERROR
"Missing 3D GMSH file named ${filename}. Please download them from
    https://drive.google.com/drive/folders/1ZjwSvbrRK5pHvMULfC69n6oierTej2Cv?usp=sharing
and place them in
      ${CMAKE_CURRENT_SOURCE_DIR}, or simply run get_gmsh_mesh_files_local.sh"
      )
endif()
configure_file(${filename} ${filename} COPYONLY)
add_test(
  NAME 3D_GMSH_READER_3D_CUBE_2ndOrder
  COMMAND mpirun -n ${MPIMAX} ${CMAKE_CURRENT_BINARY_DIR}/3D_GMSH_READER --input=${filename}
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)