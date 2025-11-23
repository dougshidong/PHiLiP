SET(SOURCE
    physics_post_processor.cpp
    )

foreach(dim RANGE 1 3)
    # Output library
    string(CONCAT PostprocessingLib Postprocessing_${dim}D)
    add_library(${PostprocessingLib} STATIC ${SOURCE})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${PostprocessingLib})
    endif()

    target_compile_definitions(${PostprocessingLib} PRIVATE PHILIP_DIM=${dim})
    # Replace occurences of PHILIP_SPECIES with 1 in the code
    # NOTE: PHILIP_SPECIES will become user defined in a future PR
    target_compile_definitions(${PostprocessingLib} PRIVATE PHILIP_SPECIES=1)
    # target_link_libraries(${PostprocessingLib} m mpi mpi_cxx)

    # Setup target with deal.II
    unset(PostprocessingLib)

endforeach()

