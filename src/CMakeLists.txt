#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#add_library(advection_lib
#    advection_boundary.h
#    advection_explicit.cpp 
#    advection_explicit.h)
#target_include_directories(advection_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})


#string(CONCAT directory build_${BUILD_DIM}D)

#set(RUNTIME_OUTPUT_DIRECTORY ${directory})


#string(CONCAT LIBNAME advlib_${BUILD_DIM}D)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#add_library(${LIBNAME}
#    parameters.cpp
#    assemble_explicit.cpp 
#    dg.cpp)
#
#target_include_directories(${LIBNAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#
##DEAL_II_SETUP_TARGET(${LIBNAME})
#set(TARGET ${LIBNAME})
#set(TARGET_SRC main.cpp)
#DEAL_II_INVOKE_AUTOPILOT()
#DEAL_II_SETUP_TARGET(${TARGET})

# The main program
#add_executable(PHiLiP main.cpp)
foreach(dim RANGE 1 3)
    set(BUILD_DIM ${dim})

    set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_${BUILD_DIM}D)
    set(CMAKE_CURRENT_BINARY_DIR ${CMAKE_SOURCE_DIR}/build_${BUILD_DIM}D)
    
    string(CONCAT TNAME PHiLiP_${BUILD_DIM}D)
    
    set(TARGET ${TNAME})
    set(TARGET_SRC 
        assemble_explicit.cpp 
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

    target_compile_definitions(${TNAME} PRIVATE BUILD_DIM=${dim})
    
    unset(BUILD_DIM)
    unset(TNAME)
    unset(CMAKE_BINARY_DIR)
    unset(CMAKE_CURRENT_BINARY_DIR)
    unset(TARGET ${TNAME})

endforeach()

#set(BUILD_DIM 1)
#add_definitions(-DDIM=${BUILD_DIM})
#
#string(CONCAT TNAME PHiLiP_${BUILD_DIM}D)
#
#set(TARGET ${TNAME})
#set(TARGET_SRC 
#    parameters.cpp
#    assemble_explicit.cpp 
#    dg.cpp
#    main.cpp)
#DEAL_II_INVOKE_AUTOPILOT()
#DEAL_II_SETUP_TARGET(${TARGET})
#
#add_definitions(-DDIM=${BUILD_DIM})
#unset(BUILD_DIM)
#
#
#
#
#
#
#
#
#
#set(BUILD_DIM 2)
#add_definitions(-DDIM=${BUILD_DIM})
#
#string(CONCAT TNAME PHiLiP_${BUILD_DIM}D)
#
#set(TARGET ${TNAME})
#set(TARGET_SRC 
#    parameters.cpp
#    assemble_explicit.cpp 
#    dg.cpp
#    main.cpp)
#DEAL_II_SETUP_TARGET(${TARGET})
#
#add_definitions(-DDIM=${BUILD_DIM})
#unset(BUILD_DIM)
#
#
#
#
#
#
#
#
#set(BUILD_DIM 3)
#add_definitions(-DDIM=${BUILD_DIM})
#
#string(CONCAT TNAME PHiLiP_${BUILD_DIM}D)
#
#set(TARGET ${TNAME})
#set(TARGET_SRC 
#    parameters.cpp
#    assemble_explicit.cpp 
#    dg.cpp
#    main.cpp)
#DEAL_II_SETUP_TARGET(${TARGET})
#
#add_definitions(-DDIM=${BUILD_DIM})
#unset(BUILD_DIM)
