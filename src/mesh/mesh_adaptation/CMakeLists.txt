set(MESH_ADAPTATION_SRC
    mesh_error_factory.cpp
    mesh_error_estimate.cpp
    mesh_adaptation.cpp
    anisotropic_mesh_adaptation.cpp
    metric_to_mesh_generator.cpp
    fe_values_shape_hessian.cpp)

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    # String to append for library names
    if(species MATCHES 1)
        string(CONCAT lib_spec _${dim}D)
    else()
        string(CONCAT lib_spec _${dim}D_${species}species)
    endif()

    string(CONCAT MeshAdaptationLib MeshAdaptation ${lib_spec})
    add_library(${MeshAdaptationLib} STATIC ${MESH_ADAPTATION_SRC})

    # Replace occurences of PHILIP_DIM with 1, 2, or 3 in the code
    target_compile_definitions(${MeshAdaptationLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})

    # Link with HighOrderGridLib
    string(CONCAT HighOrderGridLib HighOrderGrid ${lib_spec})
    target_link_libraries(${MeshAdaptationLib} ${HighOrderGridLib})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${MeshAdaptationLib})
    endif()

    unset(HighOrderGridLib)
    unset(MeshAdaptationLib)
endforeach()
endforeach()
