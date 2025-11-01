set(SOURCE
    exact_solution.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT ExactSolutionsLib ExactSolutions_${dim}D)
    add_library(${ExactSolutionsLib} STATIC ${SOURCE})
    # Link with PhysicsLib
    string(CONCAT PhysicsLib Physics_${dim}D)
    target_link_libraries(${ExactSolutionsLib} ${PhysicsLib})

    target_compile_definitions(${ExactSolutionsLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1 in the code
    # NOTE: PHILIP_SPECIES will become user defined in a future PR
    target_compile_definitions(${ExactSolutionsLib} PRIVATE PHILIP_SPECIES=1)

    unset(PhysicsLib)

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${ExactSolutionsLib})
    endif()

    unset(ExactSolutionsLib)
endforeach()
