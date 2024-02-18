RES=$(sbatch job_compile_PHiLiP.sh) &&
cd 2dnaca_less_cores/ref0 &&
sbatch --dependency=afterok:${RES##* } my_job_parameters_file.sh &&
cd ../ref1 &&
sbatch --dependency=afterok:${RES##* } my_job_parameters_file.sh &&
cd ../ref2 &&
sbatch --dependency=afterok:${RES##* } my_job_parameters_file.sh &&
cd ../ref3 &&
sbatch --dependency=afterok:${RES##* } my_job_parameters_file.sh

