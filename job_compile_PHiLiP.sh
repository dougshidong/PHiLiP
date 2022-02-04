#!/bin/bash
#SBATCH --time=1:00:00                                ## <-- increase if RUN_CTEST=true
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=compile_PHiLiP
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40                          ## <-- refer to https://docs.computecanada.ca/wiki/Advanced_MPI_scheduling
#SBATCH --mem=0                                       ## <-- total shared memory; --mem=0 means to reserve all the available memory on each node assigned to the job
#SBATCH --mail-user=firstname.lastname@mail.mcgill.ca ## <-- for receiving job updates via email
#SBATCH --mail-type=ALL                               ## <-- what kind of updates to receive by email

CLUSTER_NAME="narval" ## <-- Enter cluster name here in lowercase
SLURM_USER="username" ## <-- Enter compute canada username here
NUM_PROCS="40"        ## WARNING: must correspond to nodes*(ntasks-per-node) above
RUN_CTEST=false

## Below are the modules needed to compile PHiLiP
module --force purge
module load StdEnv/2020
##module load intel/2020.1.217
module load gcc/9.3.0
module load openmpi/4.0.3

module load petsc/3.14.1

module load trilinos/13.0.1
export TRILINOS_DIR=$EBROOTTRILINOS
module load opencascade/7.5.0
module load gmsh/4.7.0
module load metis/5.1.0
module load muparser/2.3.2
module load boost-mpi/1.72.0
module load p4est/2.2
export P4EST_DIR=$EBROOTP4EST
module load slepc/3.14.2
module load gsl/2.6
module load cmake/3.18.4

module load netcdf-c++-mpi/4.2

##module load netcdf-mpi
export METIS_DIR=$EBROOTMETIS
export GSL_DIR=$EBROOTGSL
export P4EST_DIR=$EBROOTP4EST
export METIS_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/intel2018.3/metis/5.1.0
if [ ${CLUSTER_NAME} == "beluga" ]; then
    ##export DEAL_II_DIR=/project/rrg-nadaraja-ac/Libraries/dealii/install
    export DEAL_II_DIR=/project/rrg-nadaraja-ac/Libraries/dealii_updated/dealii/install/install
elif [ ${CLUSTER_NAME} == "narval" ]; then
    export DEAL_II_DIR=/project/def-nadaraja/Libraries/dealii_updated_reinstalled/dealii/install/install
fi
export GMSH_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Compiler/intel2020/gmsh/4.7.0
export OMP_NUM_THREADS=1

cd ${SLURM_TMPDIR}
rsync  -axvH --no-g --no-p --exclude 'build*' --exclude .git --exclude '*.log' --exclude '*.out' ${SLURM_SUBMIT_DIR} .
mkdir build_release

cd build_release
cmake -DDEAL_II_DIR=$DEAL_II_DIR ../PHiLiP -DMPIMAX=${NUM_PROCS} -DCMAKE_BUILD_TYPE=Release -DGMSH_DIR=$GMSH_DIR/bin/gmsh -DGMSH_LIB=$GMSH_DIR -DCMAKE_SKIP_INSTALL_RPATH=ON 
make -j${NUM_PROCS}
if [ "${RUN_CTEST}" = true ]; then
    ctest
fi

for((i=1;i<=3;i++)); do
	cp ${SLURM_TMPDIR}/build_release/bin/PHiLiP_${i}D /home/${SLURM_USER}/scratch/PHiLiP_${i}D
done

rsync -axvH --no-g --no-p  ${SLURM_TMPDIR}/build_release ${SLURM_SUBMIT_DIR}