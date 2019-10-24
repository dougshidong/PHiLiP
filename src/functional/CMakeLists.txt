SET(SOURCE
    functional.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT FunctionalLib Functional_${dim}D)
    add_library(${FunctionalLib} STATIC ${SOURCE})
    target_compile_definitions(${FunctionalLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${TestsLib} ${PhysicsLib})
    target_link_libraries(${TestsLib} ${DiscontinuousGalerkinLib})

    # Setup target with deal.II
    DEAL_II_SETUP_TARGET(${FunctionalLib})

    # Setup target with deal.II
    unset(FunctionalLib)
    unset(DiscontinuousGalerkinLib)
    unset(PhysicsLib)

endforeach()