set(SOURCE
    viscous_numerical_flux.cpp
    numerical_flux.cpp
    )

foreach(dim RANGE 1 3)
    set(PHILIP_DIM ${dim}) 
    string(CONCAT NumericalFluxLib NumericalFlux_${PHILIP_DIM}D)
    string(CONCAT PhysicsLib Physics_${PHILIP_DIM}D)

    add_library(${NumericalFluxLib} STATIC ${SOURCE})

    # Numerical flux requires the physics library
    target_link_libraries(${TNAME} ${PhysicsLib})

    DEAL_II_SETUP_TARGET(${NumericalFluxLib})

    target_compile_definitions(${NumericalFluxLib} PRIVATE PHILIP_DIM=${dim})

    unset(NumericalFluxLib)
    unset(PhysicsLib)

endforeach()
