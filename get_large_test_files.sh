# Download necessary test files from GDrive
# Mesh files for integration tests
TARGET_DIR=tests/
gdown --folder 1QOYxwyo4MMZasTiRnayHCnZ4E8oMR0gL -O ${TARGET_DIR}

# Mesh files for gmsh_reader unit tests
TARGET_DIR=tests/unit_tests/grid/
gdown --folder 1Z89Qq940bP9WwBr1Y5YVhZSAErOM0Rg6 -O ${TARGET_DIR}

# Initialization files for decaying homogeneous isotropic turbulence
TARGET_DIR=tests/integration_tests_control_files/decaying_homogeneous_isotropic_turbulence/setup_files/
gdown --folder 16yJ0mPB8EBLwexKJfNiEsPP40uzx6CHN -O ${TARGET_DIR}
gdown --folder 1HS7oXwXvFYRCF8CAMV3T3jdR6PBekL9n -O ${TARGET_DIR}