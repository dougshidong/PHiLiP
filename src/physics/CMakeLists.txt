add_subdirectory(initial_conditions)
add_subdirectory(exact_solutions)
add_subdirectory(real_gas_file_reader_and_variables)

SET(SOURCE
    physics_factory.cpp
    physics.cpp
    convection_diffusion.cpp
    burgers.cpp
    euler.cpp
    manufactured_solution.cpp
    mhd.cpp
    navier_stokes.cpp
    burgers_rewienski.cpp
    physics_model.cpp
    model.cpp
    large_eddy_simulation.cpp
    reynolds_averaged_navier_stokes.cpp
    negative_spalart_allmaras_rans_model.cpp
    inviscid_real_gas.cpp
    real_gas.cpp
    model_factory.cpp)

set(POSSIBLE_N_SPECIES "1;11")

foreach(nspecies ${POSSIBLE_N_SPECIES})
foreach(dim RANGE 1 3)
    # Output library
    if(${nspecies} EQUAL "1")
        string(CONCAT PhysicsLib Physics_${dim}D)
    else()
        string(CONCAT PhysicsLib Physics_${dim}D_${nspecies}-SPECIES)
    endif()
    add_library(${PhysicsLib} STATIC ${SOURCE})

    # Library dependency
    string(CONCAT ParameterLib ParametersLibrary)
    target_link_libraries(${PhysicsLib} ${ParameterLib})

    # Library dependency
    string(CONCAT AllRealGasConstantsLib AllRealGasConstantsLibrary)
    target_link_libraries(${PhysicsLib} ${AllRealGasConstantsLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${PhysicsLib})
    endif()

    target_compile_definitions(${PhysicsLib} PRIVATE PHILIP_DIM=${dim})
    target_compile_definitions(${PhysicsLib} PRIVATE N_SPECIES=${nspecies})

    # Setup target with deal.II
    unset(PhysicsLib)

endforeach()
endforeach()
