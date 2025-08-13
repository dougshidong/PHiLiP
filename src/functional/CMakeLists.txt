SET(SOURCE
    functional.cpp
    target_functional.cpp
    target_boundary_functional.cpp
    adjoint.cpp
    lift_drag.cpp
    target_wall_pressure.cpp
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
    string(CONCAT FunctionalLib Functional ${lib_spec})
    add_library(${FunctionalLib} STATIC ${SOURCE})
    target_compile_definitions(${FunctionalLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT PhysicsLib Physics ${lib_spec})
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin ${lib_spec})
    string(CONCAT LinearSolverLib LinearSolver)
    string(CONCAT PostprocessingLib Postprocessing ${lib_spec})
    target_link_libraries(${FunctionalLib} ${PhysicsLib})
    target_link_libraries(${FunctionalLib} ${LinearSolverLib})
    target_link_libraries(${FunctionalLib} ${DiscontinuousGalerkinLib})
    target_link_libraries(${FunctionalLib} ${PostprocessingLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${FunctionalLib})
    endif()

    # Setup target with deal.II
    unset(FunctionalLib)
    unset(DiscontinuousGalerkinLib)
    unset(PhysicsLib)
    unset(LinearSolverLib)
    unset(PostprocessingLib)

endforeach()
endforeach()
