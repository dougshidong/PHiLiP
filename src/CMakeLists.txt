#set(PHILIP_SRC 
#    grid_study.cpp)

set(MAIN_SRC 
    main.cpp)
add_subdirectory(linear_solver)
add_subdirectory(parameters)
add_subdirectory(physics)
add_subdirectory(numerical_flux)
add_subdirectory(dg)
add_subdirectory(ode_solver)
add_subdirectory(post_processor)

add_subdirectory(testing)

foreach(dim RANGE 1 3)

    # #####################################################
#    # Main PHiLiP library
#    string(CONCAT PHiLiPLib PHiLiPLib_${dim}D)
#    message("Adding PHiLiP library " ${PHiLiPLib} " with files " ${PHILIP_SRC} "\n")
#    add_library(${PHiLiPLib} STATIC ${PHILIP_SRC})
#    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
#    target_compile_definitions(${PHiLiPLib} PRIVATE PHILIP_DIM=${dim})
#
#    # Library dependency
#    set(ParameterLib ParametersLibrary)
#    string(CONCAT ODESolverLib ODESolver_${dim}D)
#    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
#    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
#    target_link_libraries(${PHiLiPLib} ${ParameterLib})
#    target_link_libraries(${PHiLiPLib} ${ODESolverLib})
#    target_link_libraries(${PHiLiPLib} ${DiscontinuousGalerkinLib})
#    target_link_libraries(${PHiLiPLib} ${NumericalFluxLib})
#    # Setup target with deal.II
#    DEAL_II_SETUP_TARGET(${PHiLiPLib})
#
#    # Weird thing where we need to add more libraries
#    # Otherwise, we get the following error:
#    # /usr/bin/ld: ********.o: undefined reference to symbol '_ZN3MPI8Datatype4FreeEv'
#    # //usr/lib/libmpi_cxx.so.1: error adding symbols: DSO missing from command line
#    # See one of the following links (translate chinese to english using Google)
#    # https://zhuanlan.zhihu.com/p/21385662
#    # https://www.getit01.com/p2018020821385662/
#    target_link_libraries(${PHiLiPLib} m mpi)


    # #####################################################
    # Main executable
    # Simply calls the case manager given an input file
    string(CONCAT MAIN_TARGET PHiLiP_${dim}D)
    message("Adding executable " ${MAIN_TARGET} " with files " ${MAIN_SRC} "\n")
    add_executable(${MAIN_TARGET} ${MAIN_SRC})
    set_target_properties(${MAIN_TARGET} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)

    # Compile when calling 'make 1D', 'make 2D', 'make 3D'
    add_dependencies(${dim}D ${MAIN_TARGET})

    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${MAIN_TARGET} PRIVATE PHILIP_DIM=${dim})
    # Library dependency
    set(ParameterLib ParametersLibrary)
    string(CONCAT TestsLib Tests_${dim}D)
    target_link_libraries(${MAIN_TARGET} ${ParameterLib})
    target_link_libraries(${MAIN_TARGET} ${TestsLib})
    target_link_libraries(${MAIN_TARGET} m mpi)
    DEAL_II_SETUP_TARGET(${MAIN_TARGET})
    
    unset(MAIN_TARGET)
    unset(ParameterLib)
    unset(TestsLib)

endforeach()

set(SMALL_TEST testrun)
set(TARGET ${SMALL_TEST})
set(SMALL_TEST_SRC 
    temporary_test.cpp)
add_executable(${SMALL_TEST} ${SMALL_TEST_SRC})
target_link_libraries(${SMALL_TEST} mpi )
DEAL_II_SETUP_TARGET(${SMALL_TEST})
message("Adding executable " ${SMALL_TEST} " with files " ${SMALL_TEST_SRC} "\n")
unset(SMALL_TEST)
unset(TARGET ${SMALL_TEST})

#set(SMALL_TEST instantiate_vector_ad)
#set(TARGET ${SMALL_TEST})
#set(SMALL_TEST_SRC 
#    instantiate_vector_ad.cpp)
#add_executable(${SMALL_TEST} ${SMALL_TEST_SRC})
#target_link_libraries(${SMALL_TEST} mpi )
#DEAL_II_SETUP_TARGET(${SMALL_TEST})
#message("Adding executable " ${SMALL_TEST} " with files " ${SMALL_TEST_SRC} "\n")
#unset(SMALL_TEST)
#unset(TARGET ${SMALL_TEST})

set(SMALL_TEST manifoldrun)
set(TARGET ${SMALL_TEST})
set(SMALL_TEST_SRC 
    test_manifold.cpp)

add_executable(${SMALL_TEST} ${SMALL_TEST_SRC})
#    target_link_libraries(${SMALL_TEST} m mpi)

DEAL_II_SETUP_TARGET(${SMALL_TEST})
message("Adding executable " ${SMALL_TEST} " with files " ${SMALL_TEST_SRC} "\n")

unset(SMALL_TEST)
unset(TARGET ${SMALL_TEST})
