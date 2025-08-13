SET(LIMSOURCE
    bound_preserving_limiter.cpp
    tvb_limiter.cpp
    maximum_principle_limiter.cpp
    positivity_preserving_limiter.cpp
    bound_preserving_limiter_factory.cpp
    )

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT LimiterLib Limiter ${lib_spec})
    add_library(${LimiterLib} STATIC ${LIMSOURCE})

    target_compile_definitions(${LimiterLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})
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
endforeach()
