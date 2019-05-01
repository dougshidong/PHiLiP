set(TARGET_SRC 
    linear_solver.cpp
    grid_study.cpp
    assemble_implicit.cpp 
    parameters.cpp
    dg.cpp
    ode_solver.cpp
    main.cpp)
add_subdirectory(physics)
add_subdirectory(numerical_flux)

foreach(dim RANGE 1 3)
    set(PHILIP_DIM ${dim}) 

    string(CONCAT TNAME PHiLiP_${PHILIP_DIM}D)
    
    set(TARGET ${TNAME})

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

    string(CONCAT PhysicsLib Physics_${PHILIP_DIM}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${PHILIP_DIM}D)
    target_link_libraries(${TNAME} ${PhysicsLib})
    target_link_libraries(${TNAME} ${NumericalFluxLib})


    DEAL_II_SETUP_TARGET(${TNAME})
    message("Adding executable " ${TNAME} " with files " ${TARGET_SRC} "\n")

    target_compile_definitions(${TNAME} PRIVATE PHILIP_DIM=${dim})
    
    unset(PHILIP_DIM)
    unset(TNAME)
    unset(PhysicsLib)
    unset(NumericalFluxLib)
    unset(TARGET ${TNAME})

endforeach()

set(TNAME testrun)
set(TARGET ${TNAME})
set(TARGET_SRC 
    test.cpp)

add_executable(${TNAME} ${TARGET_SRC})
    target_link_libraries(${TNAME} tbb m mpi mpi_cxx)

DEAL_II_SETUP_TARGET(${TNAME})
message("Adding executable " ${TNAME} " with files " ${TARGET_SRC} "\n")

unset(TNAME)
unset(TARGET ${TNAME})
