set(SOURCE
    exact_solution.cpp
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
    string(CONCAT ExactSolutionsLib ExactSolutions ${lib_spec})
    add_library(${ExactSolutionsLib} STATIC ${SOURCE})
    # Link with PhysicsLib
    string(CONCAT PhysicsLib Physics ${lib_spec})
    target_link_libraries(${ExactSolutionsLib} ${PhysicsLib})

    target_compile_definitions(${ExactSolutionsLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})
    unset(PhysicsLib)

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ExactSolutionsLib})
    endif()

    unset(ExactSolutionsLib)
endforeach()
endforeach()
