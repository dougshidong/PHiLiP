configure_file(2d_euler_naca0012_05_200.prm 2d_euler_naca0012_05_200.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_SUBSONIC_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)

configure_file(2d_euler_naca0012_transonic_08_125.prm 2d_euler_naca0012_transonic_08_125.prm COPYONLY)
add_test(
  NAME 2D_EULER_INTEGRATION_NACA0012_TRANSONIC_LONG
  COMMAND mpirun -np ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_euler_naca0012_transonic_08_125.prm
  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
)


foreach(REFINE RANGE 0 5)
    set (NACAFILE naca0012_hopw_ref${REFINE}.msh)
    add_custom_target(
        download_${NACAFILE}
        if (EXISTS NACAFILE)
        else()
            file(DOWNLOAD ${NACAFILE} https://drive.google.com/file/d/1rzjT9DH4Cpe92SRTMig9aAKQNbDlQG5c/view?usp=sharing)
        endif()
    )
endforeach()

#wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt

configure_file(naca0012.geo naca0012.geo COPYONLY)
configure_file(naca0012_noTE.geo naca0012_noTE.geo COPYONLY)
set(TAR_NAME naca0012_airfoil_Q3)
add_custom_target(${TAR_NAME}
    #WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski
    WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    COMMAND gmsh naca0012_noTE.geo -
    DEPENDS naca0012_noTE.geo
    BYPRODUCTS naca0012.msh
    #COMMAND mv
    #     "${CMAKE_BINARY_DIR}/gmsh_meshes/joukowski/new_msh41.msh"
    #     "${CMAKE_BINARY_DIR}/tests/integration_tests_control_files/euler_integration/naca0012/joukowski_R${ref}_Q${order}.msh"
)
add_dependencies(grids ${TAR_NAME})
add_dependencies(2D ${TAR_NAME})
