#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=run_parameters_file
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40                          ## <-- refer to https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling
#SBATCH --mem=0                                       ## <-- total shared memory; --mem=0 means to reserve all the available memory on each node assigned to the job
#SBATCH --mail-user=firstname.lastname@mail.mcgill.ca ## <-- for receiving job updates via email
#SBATCH --mail-type=ALL                               ## <-- what kind of updates to receive by email

SLURM_USER="username"                    ## <-- Enter compute canada username here
PARAMETERS_FILE="my_parameters_file.prm" ## <-- Enter .prm filename here
PHiLiP_DIMENSIONS="2"                    ## WARNING: must correspond to the DIM in the .prm file
NUM_PROCS="40"                           ## WARNING: must correspond to nodes*(ntasks-per-node) above
RUN_ON_TMPDIR=true                       ## Set as true for fast write speeds (default) -- WARNING: Output files will only be copied to your output file directory once `mpirun` has completed. 

PHiLiP_EXECUTABLE="/home/${SLURM_USER}/scratch/PHiLiP_${PHiLiP_DIMENSIONS}D"

## Below are the modules needed to run the executable
module --force purge # not needed?
module load StdEnv/2020 # not needed?
##module load intel/2020.1.217
module load gcc/9.3.0 # not needed?
module load openmpi/4.0.3 # required

if [ ${RUN_ON_TMPDIR} = true ]; then
        cd ${SLURM_TMPDIR};      
fi

mpirun -n ${NUM_PROCS} "${PHiLiP_EXECUTABLE}" -i "${SLURM_SUBMIT_DIR}/${PARAMETERS_FILE}"

if [ ${RUN_ON_TMPDIR} = true ]; then
        # Get output files, exclude subdirectories
        rsync -axvH --no-g --no-p --exclude='*/' ${SLURM_TMPDIR}/* ${SLURM_SUBMIT_DIR};
fi