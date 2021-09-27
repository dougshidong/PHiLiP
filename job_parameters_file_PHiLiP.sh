#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=run_parameters_file
#SBATCH --output=%x.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2                           ## <-- there are max 40 per node
#SBATCH --mem=32G                                     ## <-- total shared memory; default unit is megabytes
#SBATCH --mail-user=firstname.lastname@mail.mcgill.ca ## for receiving job updates via email
#SBATCH --mail-type=ALL                               ## what kind of updates to receive by email

SLURM_USER="username"                    ## <-- Enter beluga username here
PARAMETERS_FILE="my_parameters_file.prm" ## <-- Enter .prm filename here
PHiLiP_DIMENSIONS="2"                    ## WARNING: must correspond to the DIM in the .prm file
NUM_PROCS="2"                            ## WARNING: must correspond to --ntasks
RUN_ON_TMPDIR=true                       ## Set as true for fast write speeds (default) -- WARNING: Output files will only be copied to your output file directory once `mpirun` has completed. 

PHiLiP_EXECUTABLE="/home/${SLURM_USER}/scratch/PHiLiP_${PHiLiP_DIMENSIONS}D"
OUTPUT_FILES_DIRECTORY_NAME="output_files"

## Below are the modules needed to run the executable
module --force purge # not needed?
module load StdEnv/2020 # not needed?
##module load intel/2020.1.217
module load gcc/9.3.0 # not needed?
module load openmpi/4.0.3 # required

if ! [ -d "${SLURM_SUBMIT_DIR}/${OUTPUT_FILES_DIRECTORY_NAME}" ]; then
        mkdir ${SLURM_SUBMIT_DIR}/${OUTPUT_FILES_DIRECTORY_NAME};
fi

if [ ${RUN_ON_TMPDIR} = true ]; then
        cd ${SLURM_TMPDIR};      
fi

mpirun -n ${NUM_PROCS} "${PHiLiP_EXECUTABLE}" -i "${SLURM_SUBMIT_DIR}/${PARAMETERS_FILE}"

if [ ${RUN_ON_TMPDIR} = true ]; then
        # Get output files, exclude subdirectories
        rsync -axvH --no-g --no-p --exclude='*/' ${SLURM_TMPDIR}/* ${SLURM_SUBMIT_DIR}/${OUTPUT_FILES_DIRECTORY_NAME};
fi