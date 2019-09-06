## Installation of PHiLiP on Beluga cluster

This section is aimed McGill's group who use Compute Canada's Beluga cluster.

The deal.II library is already installed in `/project/rrg-nadaraja-ac/Libraries/dealii/install`. Therefore, simply put the following line in your .bashrc and source it.
~~~~
export DEAL_II_DIR=/project/rrg-nadaraja-ac/Libraries/dealii/install
~~~~

Ideally, you would have forked your own version of PHiLiP if you plan on developing. See the following link for the [forking workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow).

You would then
~~~~
git clone https://github.com/dougshidong/PHiLiP
~~~~
or 
~~~~
git clone https://github.com/insertyourgithubname/PHiLiP
~~~~

~~~~
cd PHiLiP
mkdir build_release
mkdir build_debug
cd build_release
cmake ../ -DCMAKE_BUILD_TYPE=Release -DMPIMAX=8
~~~~

Note that `CMAKE_BUILD_TYPE` defaults to `Debug` if left unspecified and `MPIMAX` defaults to 4 if left unspecified.

It is recommended to use the `Release` build most of the time for performance and switch to `Debug` as needed.

The value of `MPIMAX` will determine the default number of processors `ctest` will use for the MPI jobs.

## deal.II

This is the main library being used by this code. Most of the packages are readily available through apt. p4est might need to be installed from source since the apt version is lower than what is required by deal.II.

There is an [example script](install_dealii.sh) for what has been used to install deal.II. You may need to provide the path such as `-DTRILINOS_DIR` if CMake does not find the package on its own.

The deal.II library has been setup with the following options:

~~~~
  deal.II configuration:
        CMAKE_BUILD_TYPE:       DebugRelease
        BUILD_SHARED_LIBS:      ON
        CMAKE_INSTALL_PREFIX:   /home/ddong/Codes/dealii/install
        CMAKE_SOURCE_DIR:       /home/ddong/Codes/dealii
                                (version 9.0.1, shortrev 097bf59e49)
        CMAKE_BINARY_DIR:       /home/ddong/Codes/dealii/build
        CMAKE_CXX_COMPILER:     GNU 7.3.0 on platform Linux x86_64
                                /usr/bin/mpicxx

  Configured Features (DEAL_II_ALLOW_BUNDLED = ON, DEAL_II_ALLOW_AUTODETECTION = ON):
      ( DEAL_II_WITH_64BIT_INDICES = OFF )
        DEAL_II_WITH_ADOLC set up with external dependencies
        DEAL_II_WITH_ARPACK set up with external dependencies
      ( DEAL_II_WITH_ASSIMP = OFF )
        DEAL_II_WITH_BOOST set up with external dependencies
      ( DEAL_II_WITH_CUDA = OFF )
        DEAL_II_WITH_CXX14 = ON
        DEAL_II_WITH_CXX17 = ON
        DEAL_II_WITH_GMSH set up with external dependencies
        DEAL_II_WITH_GSL set up with external dependencies
      ( DEAL_II_WITH_HDF5 = OFF )
        DEAL_II_WITH_LAPACK set up with external dependencies
        DEAL_II_WITH_METIS set up with external dependencies
        DEAL_II_WITH_MPI set up with external dependencies
        DEAL_II_WITH_MUPARSER set up with bundled packages
      ( DEAL_II_WITH_NANOFLANN = OFF )
      ( DEAL_II_WITH_NETCDF = OFF )
        DEAL_II_WITH_OPENCASCADE set up with external dependencies
        DEAL_II_WITH_P4EST set up with external dependencies
        DEAL_II_WITH_PETSC set up with external dependencies
        DEAL_II_WITH_SCALAPACK set up with external dependencies
        DEAL_II_WITH_SLEPC set up with external dependencies
        DEAL_II_WITH_SUNDIALS set up with external dependencies
      ( DEAL_II_WITH_THREADS = OFF )
        DEAL_II_WITH_TRILINOS set up with external dependencies
        DEAL_II_WITH_UMFPACK set up with external dependencies
        DEAL_II_WITH_ZLIB set up with external dependencies

  Component configuration:
      ( DEAL_II_COMPONENT_DOCUMENTATION = OFF )
        DEAL_II_COMPONENT_EXAMPLES
      ( DEAL_II_COMPONENT_PACKAGE = OFF )
      ( DEAL_II_COMPONENT_PYTHON_BINDINGS = OFF )

  Detailed information (compiler flags, feature configuration) can be found in detailed.log

  Run  $ make info  to print a help message with a list of top level targets
~~~~

To compile deal.II on the Beluga cluster, load the following modules.

~~~~
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
~~~~

and use the following cmake command
~~~~
cmake \
    ../ \
    -DCMAKE_INSTALL_PREFIX=/project/rrg-nadaraja-ac/Libraries/dealii/install \
    -DCMAKE_C_COMPILER=mpicc \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_CXX_FLAGS="-Wno-suggest-override -pthread" \
    -DCMAKE_C_FLAGS="-pthread" \
    -DCMAKE_Fortran_COMPILER= \
    -DDEAL_II_ALLOW_BUNDLED=ON \
    -DDEAL_II_WITH_MPI=ON \
    -DDEAL_II_WITH_HDF5=OFF \
    -DDEAL_II_WITH_TRILINOS=ON \
    -DDEAL_II_WITH_P4EST=ON \
    -DDEAL_II_COMPONENT_EXAMPLES=OFF \
    -DDEAL_II_COMPILER_HAS_FUSE_LD_GOLD=OFF \
~~~~
