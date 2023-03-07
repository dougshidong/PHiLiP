TARGET_DIR_BASE=tests/integration_tests_control_files/decaying_homogeneous_isotropic_turbulence/setup_files/
TARGET_DIR=${TARGET_DIR_BASE}1proc/
gdown 1YSvCzz2P3tsIQ1E6G9ZGTvJFSn4cjZaQ # setup_philip-00000.dat
mv setup_philip-00000.dat ${TARGET_DIR}

TARGET_DIR=tests/integration_tests_control_files/decaying_homogeneous_isotropic_turbulence/setup_files/4proc/
TARGET_DIR=${TARGET_DIR_BASE}4proc/
gdown 1daHBpZMyClJ5DXRuZGw-iPmID20TLmjp # setup_philip-00000.dat
gdown 1CPsUS8QGXxAHlutgi2pBuOHU7AfuZvDz # setup_philip-00001.dat
gdown 1-q_znN9N4VsPfeUe7CIdFuG2Gt-GO3gN # setup_philip-00002.dat
gdown 1oJhZeopd66IbBHepbqlJZWEMNZKECage # setup_philip-00003.dat
mv setup_philip-00000.dat ${TARGET_DIR}
mv setup_philip-00001.dat ${TARGET_DIR}
mv setup_philip-00002.dat ${TARGET_DIR}
mv setup_philip-00003.dat ${TARGET_DIR}