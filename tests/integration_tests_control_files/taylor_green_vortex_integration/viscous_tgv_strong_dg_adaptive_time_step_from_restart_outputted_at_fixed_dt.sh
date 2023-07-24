#!/bin/bash

# Assign the filename
filename="tgv_strong_adaptive_time_step_output_restart_files_at_fixed_dt-restart-00002.txt"

if [[ ! -f "${filename}" ]]; then
    echo "ERROR: Cannot find file ${filename}. Must run MPI_VISCOUS_TGV_STRONG_DG_ADAPTIVE_TIME_STEP_OUTPUT_RESTART_FILES_AT_FIXED_DT before running this test.";
fi