set(FLUX_SOURCE
    viscous_numerical_flux.cpp
    convective_numerical_flux.cpp
    split_form_numerical_flux.cpp
    numerical_flux_factory.cpp
    central_numerical_flux.cpp
    entropy_cons_numerical_flux.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT NumericalFluxLib NumericalFlux_${dim}D)
    add_library(${NumericalFluxLib} STATIC ${FLUX_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${NumericalFluxLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${NumericalFluxLib} ${PhysicsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${NumericalFluxLib})
    endif()


    unset(NumericalFluxLib)
    unset(PhysicsLib)

endforeach()
