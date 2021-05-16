#!/bin/bash
#SBATCH --time=12:15:00
#SBATCH --account=rrg-nadaraja-ac
#SBATCH --job-name=compile_PHiLiP
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=2048M      # memory; default unit is megabytes
# #SBATCH --mail-user=doug.shi-dong@mail.mcgill.ca
# #SBATCH --mail-type=ALL

# Below are the modules to compile deal.II according to Bart
module purge
module load gcc/7.3.0 
module load trilinos/12.12.1 
export TRILINOS_DIR=$EBROOTTRILINOS
module load metis/5.1.0
module load muparser/2.2.6
module load boost-mpi/1.68.0
module load p4est/2.0
module load petsc/3.10.2
export P4EST_DIR=$EBROOTP4EST
module load slepc/3.10.2
module load gmsh/4.0.7
module load gsl/2.5
module load cmake/3.12.3
module load netcdf-mpi
export METIS_DIR=$EBROOTMETIS
export GSL_DIR=$EBROOTGSL
export P4EST_DIR=$EBROOTP4EST
export METIS_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/intel2018.3/metis/5.1.0
export DEAL_II_DIR=/project/rrg-nadaraja-ac/Libraries/dealii/install
export OMP_NUM_THREADS=1

cd ${SLURM_TMPDIR}
rsync  -axvH --no-g --no-p --exclude 'build*' --exclude .git --exclude '*.log' --exclude '*.out' ${SLURM_SUBMIT_DIR} .
mkdir build_release
mkdir build_debug

cd build_release
cmake ../PHiLiP -DMPIMAX=10 -DCMAKE_BUILD_TYPE=Release
make -j10
ctest --output-on-failure

cd ../build_debug
cmake ../PHiLiP -DMPIMAX=10 -DCMAKE_BUILD_TYPE=Debug
make -j10
ctest --output-on-failure

rsync -axvH --no-g --no-p  ${SLURM_TMPDIR}/build_release ${SLURM_SUBMIT_DIR}
rsync -axvH --no-g --no-p  ${SLURM_TMPDIR}/build_debug ${SLURM_SUBMIT_DIR}

