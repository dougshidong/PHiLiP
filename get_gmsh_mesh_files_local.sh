TARGET_DIR=tests/meshes/
# get files from GoogleDrive using gdown
gdown 1rzjT9DH4Cpe92SRTMig9aAKQNbDlQG5c # naca0012_hopw_ref0.msh
gdown 1c0HUhoTR-ZFE_uy-rtV3ObMBsSlyvCpI # naca0012_hopw_ref1.msh
gdown 1n5g3KOYJgsnzEtZIhPujbKS5sN0ztYWE # naca0012_hopw_ref2.msh
gdown 1Iv8qfwh_KCv9KDeOXlBhmCaPTNiPtjQG # naca0012_hopw_ref3.msh
gdown 15Tt4ZEcZye0Q-P2ynDy67ETqeuCgINjK # naca0012_hopw_ref4.msh
gdown 1qxWlxhqK3_OrPUe9gcBpUMdny_dfrrBy # naca0012_hopw_ref5.msh
# gdown  # 3d_gaussian_bump.msh
gdown 1e_vjwogI9CTIBmolbT0EatnuXcfqP_Pi # SD7003.msh
# move files to appropriate directory
mv *.msh ${TARGET_DIR}

# Mesh files for gmsh_reader unit tests
TARGET_DIR=tests/unit_tests/grid/gmsh_reader/
gdown 1HQAoa_dS8U91r0oPPuo1NN9ozXOnjymj # airfoil.msh
# gdown  # 3d_gaussian_bump.msh
# gdown # 3D_CUBE_2ndOrder.msh
gdown 1e_vjwogI9CTIBmolbT0EatnuXcfqP_Pi # SD7003.msh
# move files to appropriate directory
mv *.msh ${TARGET_DIR}