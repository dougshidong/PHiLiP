#!/bin/bash

# Assign the filename
# filename="/home/julien/Codes/2022-06-15/PHiLiP/build_release/tests/integration_tests_control_files/taylor_green_vortex_integration/restart-00004.prm"
filename="restart-00004.prm"

if [[ ! -f "${filename}" ]]; then
	echo "ERROR: Cannot find file ${filename}. Must run MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_CHECK before running this test."
else 
	echo "Setting up .prm file for current test: "
	echo "- MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_FROM_PARAMETER_FILE_CHECK"
	echo "by modifying the outputted restart .prm file, ${filename}, generated from previously ran test:"
	echo " - MPI_VISCOUS_TAYLOR_GREEN_VORTEX_RESTART_CHECK"
	echo "..."
fi

# Take the search string
search="taylor_green_vortex_restart_check"

# Take the replace string
replace="taylor_green_vortex_energy_check"

if [[ $search != "" && $replace != "" ]]; then
sed -i "s/$search/$replace/" $filename
fi

echo "completed."
echo " "