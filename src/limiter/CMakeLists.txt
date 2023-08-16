SET(LIMSOURCE
    bound_preserving_limiter.cpp
#    bound_preserving_limiter_factory.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT LimiterLib Limiter_Lib_${dim}D)
    add_library(${LimiterLib} STATIC ${LIMSOURCE})

    target_compile_definitions(${LimiterLib} PRIVATE PHILIP_DIM=${dim})
    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${ParameterLib})
    target_link_libraries(${LimiterLib} ${PhysicsLib})
    target_link_libraries(${LimiterLib} ${OperatorsLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${LimiterLib})
    endif()

    unset(PhysicsLib)
    unset(OperatorsLib)
    unset(LimiterLib)

endforeach()
