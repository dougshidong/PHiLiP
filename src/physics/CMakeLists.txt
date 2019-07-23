SET(SOURCE
    physics_factory.cpp
    physics.cpp
    convection_diffusion.cpp
    burgers.cpp
    euler.cpp
    manufactured_solution.cpp
    split_form.cpp
    split_form_burgers.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT PhysicsLib Physics_${dim}D)
    add_library(${PhysicsLib} STATIC ${SOURCE})

    # Setup target with deal.II
    DEAL_II_SETUP_TARGET(${PhysicsLib})

    target_compile_definitions(${PhysicsLib} PRIVATE PHILIP_DIM=${dim})
    target_link_libraries(${PhysicsLib} m mpi mpi_cxx)

    # Setup target with deal.II
    unset(PhysicsLib)

endforeach()
