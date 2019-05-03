set(TARGET_SRC
    numerical_flux_conservation.cpp
    )

foreach(dim RANGE 1 3)
    set(PHILIP_DIM ${dim}) 

    string(CONCAT TARGETNAME ${PHILIP_DIM}D_numerical_flux_conservation)
    
    set(TARGET ${TARGETNAME})

    add_executable(${TARGETNAME} ${TARGET_SRC})

    # Weird thing where we need to add more libraries
    # Otherwise, we get the following error:
    # /usr/bin/ld: ********.o: undefined reference to symbol '_ZN3MPI8Datatype4FreeEv'
    # //usr/lib/libmpi_cxx.so.1: error adding symbols: DSO missing from command line
    # See one of the following links (translate chinese to english using Google)
    # https://zhuanlan.zhihu.com/p/21385662
    # https://www.getit01.com/p2018020821385662/
    target_link_libraries(${TARGETNAME} tbb m mpi mpi_cxx)

    string(CONCAT PhysicsLib Physics_${PHILIP_DIM}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${PHILIP_DIM}D)
    target_link_libraries(${TARGETNAME} ${PhysicsLib})
    target_link_libraries(${TARGETNAME} ${NumericalFluxLib})


    DEAL_II_SETUP_TARGET(${TARGETNAME})
    message("Adding executable " ${TARGETNAME} " with files " ${TARGET_SRC} "\n")

    target_compile_definitions(${TARGETNAME} PRIVATE PHILIP_DIM=${dim})
    

    add_test(
      NAME ${TARGETNAME}
      COMMAND ${EXECUTABLE_OUTPUT_PATH}/${TARGETNAME}
      WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
    )

    unset(TARGET)
    unset(PHILIP_DIM)
    unset(TARGETNAME)
    unset(PhysicsLib)
    unset(NumericalFluxLib)

endforeach()
