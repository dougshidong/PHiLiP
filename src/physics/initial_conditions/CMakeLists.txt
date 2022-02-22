set(INITIAL_CONDITIONS_SOURCE
    initial_condition.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT InitialConditionsLib InitialConditions_${dim}D)
    add_library(${InitialConditionsLib} STATIC ${INITIAL_CONDITIONS_SOURCE})
    # Link with PhysicsLib
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${InitialConditionsLib} ${PhysicsLib})

    target_compile_definitions(${InitialConditionsLib} PRIVATE PHILIP_DIM=${dim})
    unset(PhysicsLib)

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${InitialConditionsLib})
    endif()

    unset(InitialConditionsLib)
endforeach()