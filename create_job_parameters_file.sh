filename=${1}
time=${2}
job_name=${3}
nodes=${4}
ntasks_per_node=${5}
user_email=${6}
username=${7}
parameters_file=${8}
dimension_of_problem=${9}
run_on_temp_dir=${10}
memory_per_node=${11}

let number_of_processors=${nodes}*${ntasks_per_node}

echo "Creating ${filename} ..."
if test -f "${filename}"; then
	rm ${filename}
fi
touch ${filename}

echo "#!/bin/bash">>${filename}
echo "#SBATCH --time=${time}">>${filename}
echo "#SBATCH --account=rrg-nadaraja-ac">>${filename}
echo "#SBATCH --job-name=${job_name}">>${filename}
echo "#SBATCH --output=log-%j.out">>${filename}
echo "#SBATCH --nodes=${nodes}">>${filename}
echo "#SBATCH --ntasks-per-node=${ntasks_per_node}                          ## <-- refer to https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling">>${filename}
echo "#SBATCH --mem=${memory_per_node}                                       ## <-- total shared memory; --mem=0 means to reserve all the available memory on each node assigned to the job">>${filename}
echo "#SBATCH --mail-user=${user_email} ## <-- for receiving job updates via email">>${filename}
echo "#SBATCH --mail-type=ALL                               ## <-- what kind of updates to receive by email">>${filename}
echo " ">>${filename}
echo "SLURM_USER=\"${username}\"                    ## <-- Enter compute canada username here">>${filename}
echo "PARAMETERS_FILE=\"${parameters_file}\" ## <-- Enter .prm filename here">>${filename}
echo "PHiLiP_DIMENSIONS=\"${dimension_of_problem}\"                    ## WARNING: must correspond to the DIM in the .prm file">>${filename}
echo "NUM_PROCS=\"${number_of_processors}\"                           ## WARNING: must correspond to nodes*(ntasks-per-node) above">>${filename}
echo "RUN_ON_TMPDIR=${run_on_temp_dir}                      ## Set as true for fast write speeds, however, output files will only be copied to your job submit directory once mpirun has completed.">>${filename}
echo " ">>${filename}
echo "PHiLiP_EXECUTABLE=\"/home/\${SLURM_USER}/scratch/PHiLiP_\${PHiLiP_DIMENSIONS}D\"">>${filename}
echo " ">>${filename}
echo "## Below are the modules needed to run the executable">>${filename}
echo "module --force purge # not needed?">>${filename}
echo "module load StdEnv/2020 # not needed?">>${filename}
echo "##module load intel/2020.1.217">>${filename}
echo "module load gcc/9.3.0 # not needed?">>${filename}
echo "module load openmpi/4.0.3 # required">>${filename}
echo " ">>${filename}
echo "if [ \${RUN_ON_TMPDIR} = true ]; then">>${filename}
echo "    cd \${SLURM_TMPDIR};">>${filename}
echo "fi">>${filename}
echo "">>${filename}
echo "mpirun -n \${NUM_PROCS} \"\${PHiLiP_EXECUTABLE}\" -i \"\${SLURM_SUBMIT_DIR}/\${PARAMETERS_FILE}\"">>${filename}
echo " ">>${filename}
echo "if [ \${RUN_ON_TMPDIR} = true ]; then">>${filename}
echo "    # Get output files, exclude subdirectories">>${filename}
echo "    rsync -axvH --no-g --no-p --exclude='*/' \${SLURM_TMPDIR}/* \${SLURM_SUBMIT_DIR};">>${filename}
echo "fi">>${filename}
echo "done."