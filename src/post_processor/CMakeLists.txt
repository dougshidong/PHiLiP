SET(SOURCE
    physics_post_processor.cpp
    )

foreach(dim RANGE 1 3)
foreach(species RANGE 1 3)
    #  Output library
    if(species MATCHES 1)
        string(CONCAT PostprocessingLib Postprocessing_${dim}D)
    else()
        string(CONCAT PostprocessingLib Postprocessing_${dim}D_${species}species)
    endif()

    add_library(${PostprocessingLib} STATIC ${SOURCE})

    # Setup target with deal.II
    if(NOT DOC_ONLY)
        DEAL_II_SETUP_TARGET(${PostprocessingLib})
    endif()

    target_compile_definitions(${PostprocessingLib} PRIVATE PHILIP_DIM=${dim} PHILIP_SPECIES=${species})
    # target_link_libraries(${PostprocessingLib} m mpi mpi_cxx)

    # Setup target with deal.II
    unset(PostprocessingLib)

endforeach()
endforeach()
