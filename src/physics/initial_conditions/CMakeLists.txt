set(INITIAL_CONDITIONS_SOURCE
    set_initial_condition.cpp
    initial_condition_function.cpp
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
    string(CONCAT InitialConditionsLib InitialConditions ${lib_spec})
    add_library(${InitialConditionsLib} STATIC ${INITIAL_CONDITIONS_SOURCE})
    target_compile_definitions(${InitialConditionsLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Link with Libraries
    string(CONCAT PhysicsLib Physics ${lib_spec})
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
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
endforeach()
