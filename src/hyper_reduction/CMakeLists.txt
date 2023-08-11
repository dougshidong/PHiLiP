set(SOURCE
assemble_problem_ECSW.cpp)

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT ECSWLib ECSW_${dim}D)
    add_library(${ECSWLib} STATIC ${SOURCE})

    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${ECSWLib} ${ParameterLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ECSWLib})
    endif()

    target_compile_definitions(${ECSWLib} PRIVATE PHILIP_DIM=${dim})

    # Setup target with deal.II
    unset(ECSWLib)

endforeach()
