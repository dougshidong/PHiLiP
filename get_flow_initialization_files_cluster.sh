# This must be ran at once whenever a clone of PHiLiP is made onto the cluster
# DESCRIPTION: Copies the large flow initialization files that cannot be stored on GitHub
# NOTE: This is currently setup only for the Narval cluster
PATH_TO_FILES=~/projects/def-nadaraja/Libraries/flow_initialization_files

TARGET_DIR=tests/integration_tests_control_files/decaying_homogeneous_isotropic_turbulence/setup_files/
cp ${PATH_TO_FILES}/1proc/* ${TARGET_DIR}/1proc/
cp ${PATH_TO_FILES}/4proc/* ${TARGET_DIR}/4proc/