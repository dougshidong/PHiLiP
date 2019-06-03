foreach(DIM RANGE 1 3)
foreach(POLYDEGREE RANGE 1 2)
foreach(GRIDSIZE RANGE 2 4)
    configure_file(
        ${DIM}d_advection_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        ${DIM}d_advection_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        COPYONLY)
    configure_file(
        ${DIM}d_diffusion_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        ${DIM}d_diffusion_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        COPYONLY)
    configure_file(
        ${DIM}d_convection_diffusion_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        ${DIM}d_convection_diffusion_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        COPYONLY)
    configure_file(
        ${DIM}d_advection_vector_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        ${DIM}d_advection_vector_poly_${POLYDEGREE}_gridsize_${GRIDSIZE}.mat
        COPYONLY)
    configure_file(copy_matrices.sh copy_matrices.sh COPYONLY)
endforeach()
endforeach()
endforeach()
