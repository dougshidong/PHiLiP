#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=rrg-nadaraja-ac_cpu
#SBATCH --job-name=run_parameters_file
#SBATCH --output=%x-%j.out
#SBATCH --distribution=block:block
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192                          ## <-- refer to https://docs.alliancecan.ca/wiki/Rorqual/en
#SBATCH --cpus-per-task=1
#SBATCH --mem=0                                       ## <-- total shared memory; --mem=0 means to reserve all the available memory on each node assigned to the job
#SBATCH --switches=1
#SBATCH --mail-user=firstname.lastname@mail.mcgill.ca ## <-- for receiving job updates via email
#SBATCH --mail-type=ALL                               ## <-- what kind of updates to receive by email

SLURM_USER="username"                    ## <-- Enter compute canada username here
PARAMETERS_FILE="parameters.prm" ## <-- Enter .prm filename here: restart/restart-00057.prm
PHiLiP_DIMENSIONS="3"                    ## WARNING: must correspond to the DIM in the .prm file
NUM_PROCS="$SLURM_NTASKS"                           ## WARNING: must correspond to nodes*(ntasks-per-node) above
RUN_ON_TMPDIR=false                      ## Set as true for fast write speeds, however, output files will only be copied to your job submit directory once mpirun has completed.

PHiLiP_EXECUTABLE="/home/${SLURM_USER}/links/scratch/PHiLiP_${PHiLiP_DIMENSIONS}D"

## Below are the modules needed to run the executable
module --force purge # not needed?
module load StdEnv/2023 # not needed?
##module load intel/2020.1.217
module load gcc/12.3 # not needed?
module load openmpi/4.1.5 # required

if [ ${RUN_ON_TMPDIR} = true ]; then
        cd ${SLURM_TMPDIR};
	mkdir restart;
fi

srun "${PHiLiP_EXECUTABLE}" -i "${SLURM_SUBMIT_DIR}/${PARAMETERS_FILE}"

if [ ${RUN_ON_TMPDIR} = true ]; then
	ls;
        # Get output files, exclude subdirectories
        # rsync -axvH --no-g --no-p --exclude='*/' ${SLURM_TMPDIR}/* ${SLURM_SUBMIT_DIR};
	rsync -axvH --no-g --no-p ${SLURM_TMPDIR}/* ${SLURM_SUBMIT_DIR};
fi
