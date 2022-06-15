set(INITIAL_CONDITIONS_SOURCE
    set_initial_condition.cpp
    initial_condition_function.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT InitialConditionsLib InitialConditions_${dim}D)
    add_library(${InitialConditionsLib} STATIC ${INITIAL_CONDITIONS_SOURCE})
    target_compile_definitions(${InitialConditionsLib} PRIVATE PHILIP_DIM=${dim})

    # Link with Libraries
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    target_link_libraries(${InitialConditionsLib} ${PhysicsLib})
    target_link_libraries(${InitialConditionsLib} ${DiscontinuousGalerkinLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${InitialConditionsLib})
    endif()

    unset(PhysicsLib)
    unset(DiscontinuousGalerkinLib)
    unset(InitialConditionsLib)
endforeach()
