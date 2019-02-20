foreach(dim RANGE 1 3)
    set(PHILIP_DIM ${dim})

    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_${PHILIP_DIM}D)
    set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_${PHILIP_DIM}D)
    
    string(CONCAT TNAME PHiLiP_${PHILIP_DIM}D)
    
    set(TARGET ${TNAME})
    set(TARGET_SRC 
        assemble_explicit.cpp 
        assemble_implicit.cpp 
        parameters.cpp
        dg.cpp
        main.cpp)

    add_executable(${TNAME} ${TARGET_SRC})

    # Weird thing where we need to add more libraries
    # Otherwise, we get the following error:
    # /usr/bin/ld: ********.o: undefined reference to symbol '_ZN3MPI8Datatype4FreeEv'
    # //usr/lib/libmpi_cxx.so.1: error adding symbols: DSO missing from command line
    # See one of the following links (translate chinese to english using Google)
    # https://zhuanlan.zhihu.com/p/21385662
    # https://www.getit01.com/p2018020821385662/
    #target_link_libraries(${TNAME} tbb m mpi mpi_cxx)
    target_link_libraries(${TNAME} tbb m mpi mpi_cxx)

    DEAL_II_SETUP_TARGET(${TNAME})
    message("Adding executable " ${TNAME} " with files " ${TARGET_SRC} "\n")

    target_compile_definitions(${TNAME} PRIVATE PHILIP_DIM=${dim})
    
    unset(PHILIP_DIM)
    unset(TNAME)
    unset(CMAKE_BINARY_DIR)
    unset(CMAKE_CURRENT_BINARY_DIR)
    unset(TARGET ${TNAME})

endforeach()

