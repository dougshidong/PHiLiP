# Unit tests
add_subdirectory(numerical_flux)
add_subdirectory(regression)
add_subdirectory(euler_unit_test)

# Integration tests
add_subdirectory(advection_implicit)
add_subdirectory(diffusion_implicit)
add_subdirectory(convection_diffusion_implicit)
add_subdirectory(advection_vector_implicit)
add_subdirectory(burgers_inviscid_implicit)
add_subdirectory(burgers_split_stability)

add_subdirectory(euler_implicit)

add_subdirectory(euler_integration)
#add_subdirectory(euler_gaussian_bump)
#add_subdirectory(euler_vortex)
#add_subdirectory(euler_cylinder)
