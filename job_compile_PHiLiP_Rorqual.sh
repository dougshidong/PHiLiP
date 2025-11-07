#!/bin/bash
#SBATCH --time=1:00:00                                ## <-- increase if RUN_CTEST=true
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=compile_PHiLiP
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16                          ## <-- refer to https://docs.alliancecan.ca/wiki/Rorqual/en
#SBATCH --mem=63G                                     ## <-- total shared memory per node; refer to https://docs.alliancecan.ca/wiki/Rorqual/en
#SBATCH --mail-user=firstname.lastname@mail.mcgill.ca ## <-- for receiving job updates via email
#SBATCH --mail-type=ALL                               ## <-- what kind of updates to receive by email


SLURM_USER="username" ## <-- Enter compute canada username here
NUM_PROCS="16"        ## WARNING: must correspond to nodes*(ntasks-per-node) above
RUN_CTEST=false

## Below are the modules needed to compile PHiLiP
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load openmpi/4.1.5
module load petsc/3.20.0

 

##Export working version of Trilinos from project directory
export TRILINOS_DIR=/project/def-nadaraja/Libraries/Trilinos/install
export Trilinos_ROOT=/project/def-nadaraja/Libraries/Trilinos/install
export Trilinos_DIR=$Trilinos_ROOT/lib/cmake/Trilinos

module load gmsh/4.13.1
module load metis/5.1.0
module load muparser/2.3.4
module load boost-mpi/1.82.0
module load p4est/2.8.6
export P4EST_DIR=$EBROOTP4EST
module load slepc/3.20.1
module load gsl/2.7
module load cmake/3.31.0
module load netcdf-c++4-mpi/4.3.1

 
export METIS_DIR=$EBROOTMETIS
export GSL_DIR=$EBROOTGSL
export P4EST_DIR=$EBROOTP4EST
export METIS_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/intel2018.3/metis/5.1.0
##Export deal.ii directory
export DEAL_II_DIR=/project/def-nadaraja/Libraries/dealii_updated_reinstalled/dealii/install/install
export GMSH_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Compiler/intel2020/gmsh/4.11.1
export OMP_NUM_THREADS=1

 

cd ${SLURM_TMPDIR}
rsync  -axvH --no-g --no-p --exclude 'build_release' --exclude 'build_debug' --exclude .git --exclude '*.log' --exclude '*.out' ${SLURM_SUBMIT_DIR} .
mkdir build_release

cd build_release
cmake -DDEAL_II_DIR=$DEAL_II_DIR ../PHiLiP -DMPIMAX=${NUM_PROCS} -DCMAKE_BUILD_TYPE=Release -DGMSH_DIR=$GMSH_DIR/bin/gmsh -DGMSH_LIB=$GMSH_DIR -DCMAKE_SKIP_INSTALL_RPATH=ON 
make -j${NUM_PROCS}


if [ "${RUN_CTEST}" = true ]; then
    ctest
fi
 

for((i=1;i<=3;i++)); do
	cp ${SLURM_TMPDIR}/build_release/bin/PHiLiP_${i}D /home/${SLURM_USER}/links/scratch/PHiLiP_${i}D
done
 
rsync -axvH --no-g --no-p  ${SLURM_TMPDIR}/build_release ${SLURM_SUBMIT_DIR}
