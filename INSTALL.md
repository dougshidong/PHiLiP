
## Dependencies

There is a heavy dependencies on various different libraries. While it means we are not reinventing the wheel for many solved problems or optimized algorithms, it also leads to a more complicated setup.

On a clean install of Ubuntu 20.04, you may run the following script:
https://github.com/dougshidong/PHiLiP/blob/master/doc/install_ubuntu2004_VM.sh

If you are on another OS, the above script still serves as a nice guideline since it lists all the necessary options from the dependent libraries.

Note that some features and tests from a recent pull request only work with Gmsh 4.6. See
https://github.com/dougshidong/PHiLiP/pull/55

## NACA0012 Mesh Files

**If you are running on the cluster, please see the instructions below in the Compute Canada section.**

If you are running the code **on a local machine** (i.e. not on the cluster), you will need to download the [NACA0012 mesh files](https://drive.google.com/drive/folders/182JusbWV6NAA8ws1-TTg7M2GLc5jt6_r?usp=sharing) that are too large to store on GitHub by clicking the link, and place them in `tests/integration_tests_control_files/euler_integration/naca0012/`. To automate this process using `gdown`, do the following:
1. `sudo apt install python3-pip` (if `pip` is not already installed)
2. `pip install gdown` (if `gdown` is not already installed)
3. If you receive a warning such as:
   `WARNING: The script gdown is installed in '/home/parallels/.local/bin' which is not on PATH.`
   then simply add the path by doing the following (modify accordingly):
   `echo export PATH=/home/parallels/.local/bin:$PATH >> ~/.bashrc`
   and resource:
   `source ~/.bashrc`
4. Then run the following bash script inside the `PHiLiP` directory:
   `chmod +x get_NACA0012_mesh_files_local.sh`
   `./get_NACA0012_mesh_files_local.sh`

## deal.II

This is the main library being used by this code. We keep track of the working version from our [fork](https://github.com/dougshidong/dealii).
Most of the packages are readily available through `apt install`. p4est might need to be installed from source since the current p4est version available on `apt` is lower than what is required by deal.II (p4est 2.0). A small set of instructions is available [here](https://www.dealii.org/current/external-libs/p4est.html).

There is an [example script](doc/install_dealii.sh) for what has been used to install deal.II. You may need to provide the path such as `-DTRILINOS_DIR` if CMake does not find the package on its own.

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

## Installation of PHiLiP on Compute Canada clusters

This section is aimed at McGill's group who use the Compute Canada (CC) clusters.

If you have just cloned the code onto the cluster, **you must copy the large NACA0012 mesh files** that cannot be stored on GitHub, this can be done by explicitly running the following:
~~~~
chmod +x get_NACA0012_mesh_files_cluster.sh
./get_NACA0012_mesh_files_cluster.sh
~~~~
If you are a **new user on a CC cluster**, **you must configure git modules** by explicitly running the following on the cluster before proceeding:
~~~~
chmod +x configure_git_submodules_cluster.sh
./configure_git_submodules_cluster.sh
~~~~
in which the shell script runs the following commands:
~~~~
git submodule init
git submodule update
git config --global http.proxy ""
git pull --recurse-submodules
git submodule update --recursive
~~~~
For Beluga, the deal.II library is already installed in `/project/rrg-nadaraja-ac/Libraries/dealii/install`; install paths for other clusters are included in `job_compile_PHiLiP.sh`. The required modules were installed by Bart Oldeman from Compute Canada's team through modules. Therefore, simply put the following line in your .bashrc and source it.
~~~~
module --force purge
module load StdEnv/2020
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
export METIS_DIR=$EBROOTMETIS
export GSL_DIR=$EBROOTGSL
export P4EST_DIR=$EBROOTP4EST
export METIS_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/intel2018.3/metis/5.1.0
export DEAL_II_DIR=/project/rrg-nadaraja-ac/Libraries/dealii_updated/dealii/install/install
export GMSH_DIR=/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx512/Compiler/intel2020/gmsh/4.7.0
export OMP_NUM_THREADS=1
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

Afterwards, you can build the code:
~~~~
cd PHiLiP
mkdir build_release
mkdir build_debug
cd build_release
cmake ../ -DCMAKE_BUILD_TYPE=Release -DMPIMAX=8
make -j 6
~~~~
Note that `CMAKE_BUILD_TYPE` defaults to `Debug` if left unspecified and `MPIMAX` defaults to 4 if left unspecified.  
It is recommended to use the `Release` build most of the time for performance and switch to `Debug` as needed.
The value of `MPIMAX` will determine the default number of processors `ctest` will use for the MPI jobs. You will still be able to run those tests with more than the speficied `MPIMAX` processors. For example,
~~~~
ctest -R MPI_2D_EULER_INTEGRATION_CYLINDER -V
~~~~
will launch 
~~~~
/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx512/Compiler/gcc7.3/openmpi/3.1.2/bin/mpirun "-np" "8" "/home/ddong/projects/rrg-nadaraja-ac/ddong/PHiLiP/build_release/bin/PHiLiP_2D" "-i" "/home/ddong/projects/rrg-nadaraja-ac/ddong/PHiLiP/build_release/tests/euler_integration/2d_euler_cylinder.prm"
~~~~
However, you can manually launch this program through the command line and changing the "8" to whatever number you want.

Running ctest might take a while so, you may want to [request a computational node](https://docs.computecanada.ca/wiki/Running_jobs) before running
~~~~
ctest
~~~~
Note that you want to request at least as many processes as MPIMAX.

Unless you absolutely want to run a steady explicit case, I would suggest disabling the MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION test through
~~~~
ctest -E MPI_2D_ADVECTION_EXPLICIT_MANUFACTURED_SOLUTION
~~~~
since it takes a very long time.

If you have any questions, feel free to contact me.

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

## Running PHiLiP using Parameter Files on the Beluga cluster
After running `job_compile_PHiLiP.sh`, three PHiLiP executables are generated: `PHiLiP_1D`, `PHiLiP_2D`, `PHiLiP_3D`. These will be located in `home/username/scratch/`. For running PHiLiP using parameter files (`.prm`), create an appropriate directory for the runs e.g. `home/username/projects/rrg-nadaraja-ac/username/run_dir_name`, copy the 3 executables and the `job_parameters_file_PHiLiP.sh` to this directory, and modify the shell script accordingly. Submit jobs using `sbatch job_parameters_file_PHiLiP.sh`. 

## Trilinos
This is an important dependency of both deal.II and PHiLiP library being used by this code. We keep track of the working version from our [fork](https://github.com/dougshidong/Trilinos). Our fork also contains some bug fixes that have yet to be merged within their repository. An installation script with all the necessary options is given in
https://raw.githubusercontent.com/dougshidong/PHiLiP/master/doc/prep_trilinos.sh