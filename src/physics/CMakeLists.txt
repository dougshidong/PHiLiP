set(SOURCE
    linear_advection.cpp
    diffusion.cpp
    convection_diffusion.cpp
    physics.cpp
    )

foreach(dim RANGE 1 3)
    set(PHILIP_DIM ${dim}) 
    string(CONCAT PhysicsLib Physics_${PHILIP_DIM}D)

    add_library(${PhysicsLib} STATIC ${SOURCE})

    DEAL_II_SETUP_TARGET(${PhysicsLib})

    target_compile_definitions(${PhysicsLib} PRIVATE PHILIP_DIM=${dim})
    unset(PhysicsLib)

endforeach()
