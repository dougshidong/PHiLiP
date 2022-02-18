TARGET_DIR=tests/integration_tests_control_files/euler_integration/naca0012/
# get files from GoogleDrive using gdown
gdown --id 1rzjT9DH4Cpe92SRTMig9aAKQNbDlQG5c # naca0012_hopw_ref0.msh
gdown --id 1c0HUhoTR-ZFE_uy-rtV3ObMBsSlyvCpI # naca0012_hopw_ref1.msh
gdown --id 1n5g3KOYJgsnzEtZIhPujbKS5sN0ztYWE # naca0012_hopw_ref2.msh
gdown --id 1Iv8qfwh_KCv9KDeOXlBhmCaPTNiPtjQG # naca0012_hopw_ref3.msh
gdown --id 15Tt4ZEcZye0Q-P2ynDy67ETqeuCgINjK # naca0012_hopw_ref4.msh
gdown --id 1qxWlxhqK3_OrPUe9gcBpUMdny_dfrrBy # naca0012_hopw_ref5.msh
# move files to appropriate directory
mv naca0012_hopw_ref* ${TARGET_DIR}