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
    # target_link_libraries(${PostprocessingLib} m mpi mpi_cxx)

    # Setup target with deal.II
    unset(PostprocessingLib)

endforeach()

