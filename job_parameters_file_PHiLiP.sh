#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=run_parameters_file
#SBATCH --output=%x.out ##%x-%j.out                   ## <-- use this to append job ID
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2                           ## <-- there are max 40 per node
#SBATCH --mem=32G                                     ## <-- total shared memory; default unit is megabytes
#SBATCH --mail-user=firstname.lastname@mail.mcgill.ca ## for receiving job updates via email
#SBATCH --mail-type=ALL                               ## what kind of updates to receive by email

SLURM_USER="username"                    ## <-- Enter beluga username here
PARAMETERS_FILE="my_parameters_file.prm" ## <-- Enter .prm filename here
PHiLiP_DIMENSIONS="2"                    ## WARNING: must correspond to the DIM in the .prm file
NUM_PROCS="2"                            ## WARNING: must correspond to --ntasks
PHiLiP_EXECUTABLE_NAME="PHiLiP_${PHiLiP_DIMENSIONS}D"
OUTPUT_FILES_DIRECTORY_NAME="output_files"

## Below are the modules needed to run the executable
module --force purge # not needed?
module load StdEnv/2020 # not needed?
##module load intel/2020.1.217
module load gcc/9.3.0 # not needed?
module load openmpi/4.0.3 # required

cd ${SLURM_TMPDIR}

mpirun -n ${NUM_PROCS} "${SLURM_SUBMIT_DIR}/${PHiLiP_EXECUTABLE_NAME}" -i "${SLURM_SUBMIT_DIR}/${PARAMETERS_FILE}"

if ! [ -d "${SLURM_SUBMIT_DIR}/${OUTPUT_FILES_DIRECTORY_NAME}" ]; then
        mkdir ${SLURM_SUBMIT_DIR}/${OUTPUT_FILES_DIRECTORY_NAME};
fi

# Get output files, exclude subdirectories
rsync -axvH --no-g --no-p --exclude='*/' ${SLURM_TMPDIR}/* ${SLURM_SUBMIT_DIR}/${OUTPUT_FILES_DIRECTORY_NAME}
