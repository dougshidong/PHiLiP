SET(SOURCE
    functional.cpp
    target_functional.cpp
    target_boundary_functional.cpp
    adjoint.cpp
    lift_drag.cpp
    target_wall_pressure.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT FunctionalLib Functional_${dim}D)
    add_library(${FunctionalLib} STATIC ${SOURCE})
    target_compile_definitions(${FunctionalLib} PRIVATE PHILIP_DIM=${dim})

    # Library dependency
    string(CONCAT PhysicsLib Physics_${dim}D)
    string(CONCAT DiscontinuousGalerkinLib DiscontinuousGalerkin_${dim}D)
    string(CONCAT LinearSolverLib LinearSolver)
    string(CONCAT PostprocessingLib Postprocessing_${dim}D)
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
