SET(LIMSOURCE
    bound_preserving_limiter.cpp
    tvb_limiter.cpp
    maximum_principle_limiter.cpp
    positivity_preserving_limiter.cpp
    bound_preserving_limiter_factory.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT LimiterLib Limiter_${dim}D)
    add_library(${LimiterLib} STATIC ${LIMSOURCE})

    target_compile_definitions(${LimiterLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1 in the code
    # NOTE: PHILIP_SPECIES will become user defined in a future PR
    target_compile_definitions(${LimiterLib} PRIVATE PHILIP_SPECIES=1)

    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${ParameterLib})
    target_link_libraries(${LimiterLib} ${PhysicsLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${LimiterLib})
    endif()

    unset(PhysicsLib)
    unset(LimiterLib)

endforeach()
