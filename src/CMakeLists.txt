set(PHILIP_SRC 
    linear_solver.cpp
    grid_study.cpp
    assemble_implicit.cpp 
    dg.cpp
    ode_solver.cpp)
set(MAIN_SRC 
    main.cpp)
add_subdirectory(parameters)
add_subdirectory(physics)
add_subdirectory(numerical_flux)

foreach(dim RANGE 1 3)
    set(PHILIP_DIM ${dim}) 

    string(CONCAT PHiLiPLib PHiLiPLib_${PHILIP_DIM}D)
    add_library(${PHiLiPLib} STATIC ${PHILIP_SRC})

    # Weird thing where we need to add more libraries
    # Otherwise, we get the following error:
    # /usr/bin/ld: ********.o: undefined reference to symbol '_ZN3MPI8Datatype4FreeEv'
    # //usr/lib/libmpi_cxx.so.1: error adding symbols: DSO missing from command line
    # See one of the following links (translate chinese to english using Google)
    # https://zhuanlan.zhihu.com/p/21385662
    # https://www.getit01.com/p2018020821385662/
    target_link_libraries(${PHiLiPLib} tbb m mpi mpi_cxx)

    set(ParameterLib ParametersLibrary)
    string(CONCAT PhysicsLib Physics_${PHILIP_DIM}D)
    string(CONCAT NumericalFluxLib NumericalFlux_${PHILIP_DIM}D)
    target_link_libraries(${PHiLiPLib} ${ParameterLib})
    target_link_libraries(${PHiLiPLib} ${PhysicsLib})
    target_link_libraries(${PHiLiPLib} ${NumericalFluxLib})

    string(CONCAT MAIN_TARGET PHiLiP_${PHILIP_DIM}D)
    add_executable(${MAIN_TARGET} ${MAIN_SRC})
    target_link_libraries(${MAIN_TARGET} ${PHiLiPLib})


    target_compile_definitions(${PHiLiPLib} PRIVATE PHILIP_DIM=${dim})
    target_compile_definitions(${MAIN_TARGET} PRIVATE PHILIP_DIM=${dim})


    DEAL_II_SETUP_TARGET(${PHiLiPLib})
    DEAL_II_SETUP_TARGET(${MAIN_TARGET})

    message("Adding PHiLiP library " ${PHiLiPLib} " with files " ${PHILIP_SRC} "\n")
    message("Adding executable " ${MAIN_TARGET} " with files " ${MAIN_SRC} "\n")

    
    unset(PHILIP_DIM)
    unset(MAIN_TARGET)
    unset(PhysicsLib)
    unset(NumericalFluxLib)

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
