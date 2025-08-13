set(FLUX_SOURCE
    viscous_numerical_flux.cpp
    convective_numerical_flux.cpp
    numerical_flux_factory.cpp)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    # Output library
    string(CONCAT NumericalFluxLib NumericalFlux ${lib_spec})
    add_library(${NumericalFluxLib} STATIC ${FLUX_SOURCE})
    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${NumericalFluxLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Library dependency
    string(CONCAT PhysicsLib Physics ${lib_spec})
    target_link_libraries(${NumericalFluxLib} ${PhysicsLib})
    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${NumericalFluxLib})
    endif()

    unset(NumericalFluxLib)
    unset(PhysicsLib)
endforeach()
endforeach()
