#!/bin/bash

# Assign the filename
filename="tgv_strong_adaptive_time_step-restart-00004.txt"

if [[ ! -f "${filename}" ]]; then
    echo "ERROR: Cannot find file ${filename}. Must run MPI_VISCOUS_TGV_STRONG_DG_ADAPTIVE_TIME_STEP before running this test."
fi