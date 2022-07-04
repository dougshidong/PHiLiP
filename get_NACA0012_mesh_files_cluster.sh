# This must be ran at once whenever a clone of PHiLiP is made onto the cluster
# DESCRIPTION: Copies the large mesh files for NACA0012 that cannot be stored on GitHub
# NOTE: This is currently setup only for the Narval cluster
PATH_TO_FILES=~/projects/def-nadaraja/Libraries/NACA0012MeshFiles
FILENAMES=(naca0012_hopw_ref5.msh naca0012_hopw_ref4.msh naca0012_hopw_ref3.msh naca0012_hopw_ref2.msh naca0012_hopw_ref1.msh naca0012_hopw_ref0.msh naca0012.msh)
TARGET_DIR=tests/meshes/

for file in ${FILENAMES[@]}; do
    cp ${PATH_TO_FILES}/${file} ${TARGET_DIR}
done

TARGET_DIR=tests/unit_tests/grid/gmsh_reader/
cp ${PATH_TO_FILES}/airfoil.msh ${TARGET_DIR}