set(SOURCE
    #linear_advection.cpp
    #diffusion.cpp
    #convection_diffusion.cpp
    physics.cpp
    euler.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT PhysicsLib Physics_${dim}D)
    add_library(${PhysicsLib} STATIC ${SOURCE})

    # Setup target with deal.II
    DEAL_II_SETUP_TARGET(${PhysicsLib})

    target_compile_definitions(${PhysicsLib} PRIVATE PHILIP_DIM=${dim})
    # Setup target with deal.II
    unset(PhysicsLib)

endforeach()
