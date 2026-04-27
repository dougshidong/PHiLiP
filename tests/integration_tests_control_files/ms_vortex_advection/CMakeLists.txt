set(TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})

if(${NUMBER_OF_SPECIES} EQUAL 2)
	configure_file(1d_multi_species_vortex_advection_ooa_2species.prm 1d_multi_species_vortex_advection_ooa_2species.prm COPYONLY)
	add_test(
	  NAME 1D_MULTISPECIES_VORTEX_ADVECTION_2_SPECIES_OOA
	  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection_ooa_2species.prm
	  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
	)
	set_tests_labels(1D_MULTISPECIES_VORTEX_ADVECTION_2_SPECIES_OOA MULTISPECIES_VORTEX_ADVECTION
																	1D
																	SERIAL
																	MULTISPECIES
																	2SPECIES
																	WEAK
																	COLLOCATED
																	OOA)
																	
	configure_file(2d_multi_species_vortex_advection_ooa_2species.prm 2d_multi_species_vortex_advection_ooa_2species.prm COPYONLY)
	add_test(
	  NAME 2D_MULTISPECIES_VORTEX_ADVECTION_2_SPECIES_OOA
	  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_vortex_advection_ooa_2species.prm
	  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
	)
	set_tests_labels(2D_MULTISPECIES_VORTEX_ADVECTION_2_SPECIES_OOA MULTISPECIES_VORTEX_ADVECTION
																	2D
																	PARALLEL
																	MULTISPECIES
																	2SPECIES
																	WEAK
																	COLLOCATED
																	OOA)
																	
	# configure_file(3d_multi_species_vortex_advection_ooa_2species.prm 3d_multi_species_vortex_advection_ooa_2species.prm COPYONLY)
	# add_test(
	#   NAME 3D_MULTISPECIES_VORTEX_ADVECTION_2_SPECIES_OOA
	#   COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_multi_species_vortex_advection_ooa_2species.prm
	#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
	# )
	# set_tests_labels(3D_MULTISPECIES_VORTEX_ADVECTION_2_SPECIES_OOA MULTISPECIES_VORTEX_ADVECTION
	# 																3D
	# 																PARALLEL
	# 																MULTISPECIES
	# 																2SPECIES
	# 																WEAK
	# 																COLLOCATED
	# 																OOA)
endif()

if(${NUMBER_OF_SPECIES} EQUAL 3)
	configure_file(1d_multi_species_vortex_advection_ooa_3species.prm 1d_multi_species_vortex_advection_ooa_3species.prm COPYONLY)
	add_test(
	  NAME 1D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES_OOA
	  COMMAND mpirun -n 1 ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_1D -i ${CMAKE_CURRENT_BINARY_DIR}/1d_multi_species_vortex_advection_ooa_3species.prm
	  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
	)
	set_tests_labels(1D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES_OOA MULTISPECIES_VORTEX_ADVECTION
																	1D
																	SERIAL
																	MULTISPECIES
																	3SPECIES
																	WEAK
																	COLLOCATED
																	OOA)
																	
	configure_file(2d_multi_species_vortex_advection_ooa_3species.prm 2d_multi_species_vortex_advection_ooa_3species.prm COPYONLY)
	add_test(
	  NAME 2D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES_OOA
	  COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_2D -i ${CMAKE_CURRENT_BINARY_DIR}/2d_multi_species_vortex_advection_ooa_3species.prm
	  WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
	)
	set_tests_labels(2D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES_OOA MULTISPECIES_VORTEX_ADVECTION
																	2D
																	PARALLEL
																	MULTISPECIES
																	3SPECIES
																	WEAK
																	COLLOCATED
																	OOA)

	# configure_file(3d_multi_species_vortex_advection_ooa_3species.prm 3d_multi_species_vortex_advection_ooa_3species.prm COPYONLY)
	# add_test(
	#   NAME 3D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES_OOA
	#   COMMAND mpirun -n ${MPIMAX} ${EXECUTABLE_OUTPUT_PATH}/PHiLiP_3D -i ${CMAKE_CURRENT_BINARY_DIR}/3d_multi_species_vortex_advection_ooa_3species.prm
	#   WORKING_DIRECTORY ${TEST_OUTPUT_DIR}
	# )
	# set_tests_labels(3D_MULTISPECIES_VORTEX_ADVECTION_3_SPECIES_OOA MULTISPECIES_VORTEX_ADVECTION
	# 																3D
	# 																PARALLEL
	# 																MULTISPECIES
	# 																3SPECIES
	# 																WEAK
	# 																COLLOCATED
	# 																OOA)
endif()