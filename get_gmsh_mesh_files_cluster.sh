# This must be ran at once whenever a clone of PHiLiP is made onto the cluster
# DESCRIPTION: Copies the large mesh files for NACA0012 that cannot be stored on GitHub
# NOTE: This is currently setup only for the Narval cluster
PATH_TO_FILES=~/projects/def-nadaraja/Libraries/NACA0012MeshFiles
# Copy meshes required for integration tests
TARGET_DIR=tests/meshes/
FILENAMES=(\
naca0012_hopw_ref5.msh \
naca0012_hopw_ref4.msh \
naca0012_hopw_ref3.msh \
naca0012_hopw_ref2.msh \
naca0012_hopw_ref1.msh \
naca0012_hopw_ref0.msh \
naca0012.msh \
3d_gaussian_bump.msh \
3d_cube_periodic.msh \
SD7003_1_cell_spanwise.msh \
SD7003_4_cell_spanwise.msh \
SD7003_12_cell_spanwise.msh \
)
for file in ${FILENAMES[@]}; do
    cp ${PATH_TO_FILES}/${file} ${TARGET_DIR}
done
# Copy meshes required by unit tests
TARGET_DIR=tests/unit_tests/grid/gmsh_reader/
FILENAMES=(\
airfoil.msh \
naca0012_hopw_ref2.msh \
3D_CUBE_2ndOrder.msh \
3d_gaussian_bump.msh \
3d_cube_periodic.msh \
SD7003_1_cell_spanwise.msh \
SD7003_4_cell_spanwise.msh \
SD7003_12_cell_spanwise.msh \
channel_structured.msh \
)
for file in ${FILENAMES[@]}; do
    cp ${PATH_TO_FILES}/${file} ${TARGET_DIR}
done