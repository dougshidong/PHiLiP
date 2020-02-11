#include <iostream>

#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/solution_transfer.h>

#include "tests.h"
#include "grid_study.h"
#include "burgers_stability.h"
#include "euler_gaussian_bump.h"
#include "euler_cylinder.h"
#include "euler_vortex.h"
#include "euler_entropy_waves.h"
#include "advection_explicit_periodic.h"
#include "euler_split_inviscid_taylor_green_vortex.h"
#include "convection_diffusion_explicit_periodic.h"

namespace PHiLiP {
namespace Tests {

using AllParam = Parameters::AllParameters;

TestsBase::TestsBase(Parameters::AllParameters const *const parameters_input)
    : all_parameters(parameters_input)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{}

std::vector<int> TestsBase::get_number_1d_cells(const int n_grids) const
{
    std::vector<int> n_1d_cells(n_grids);
    Parameters::ManufacturedConvergenceStudyParam param = all_parameters->manufactured_convergence_study_param;
    n_1d_cells[0] = param.initial_grid_size;
    for (int igrid=1;igrid<n_grids;++igrid) {
        n_1d_cells[igrid] = static_cast<int>(n_1d_cells[igrid-1]*param.grid_progression) + param.grid_progression_add;
    }
    return n_1d_cells;

}

//template<int dim, int nstate>
// void TestsBase::globally_refine_and_interpolate(DGBase<dim, double> &dg) const
//{
//    dealii::LinearAlgebra::distributed::Vector<double> old_solution(dg->solution);
//    dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim>> solution_transfer(dg->dof_handler);
//    solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
//    grid.refine_global (1);
//    dg->allocate_system ();
//    solution_transfer.interpolate(old_solution, dg->solution);
//    solution_transfer.clear();
//}

template<int dim, int nstate>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate>
::select_test(const AllParam *const parameters_input) {
    using Test_enum = AllParam::TestType;
    Test_enum test_type = parameters_input->test_type;

    if(test_type == Test_enum::run_control) {
        return std::make_unique<GridStudy<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::burgers_energy_stability) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<BurgersEnergyStability<dim,nstate>>(parameters_input);
        //if constexpr (nstate == dim) return std::make_unique<BurgersEnergyStability<dim,nstate>>(parameters_input);
    } else if (test_type == Test_enum::advection_periodicity){
        if constexpr (nstate == 1) return std::make_unique<AdvectionPeriodic<dim,nstate>> (parameters_input);
    } else if (test_type == Test_enum::convection_diffusion_periodicity){
        if constexpr (nstate == 1) return std::make_unique<ConvectionDiffusionPeriodic<dim,nstate>> (parameters_input);
    } else if(test_type == Test_enum::euler_gaussian_bump) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBump<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_cylinder) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerCylinder<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_vortex) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerVortex<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_entropy_waves) {
        if constexpr (dim>=2 && nstate==PHILIP_DIM+2) return std::make_unique<EulerEntropyWaves<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_split_taylor_green) {
    	if constexpr (dim==3 && nstate == dim+2) return std::make_unique<EulerTaylorGreen<dim,nstate>>(parameters_input);
    } else {
        std::cout << "Invalid test." << std::endl;
    }

    return nullptr;
}

template<int dim, int nstate>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate>
::create_test(AllParam const *const parameters_input)
{
    // Recursive templating required because template parameters must be compile time constants
    // As a results, this recursive template initializes all possible dimensions with all possible nstate
    // without having 15 different if-else statements
    if(dim == parameters_input->dimension)
    {
        // This template parameters dim and nstate match the runtime parameters
        // then create the selected test with template parameters dim and nstate
        // Otherwise, keep decreasing nstate and dim until it matches
        if(nstate == parameters_input->nstate) 
            return TestsFactory<dim,nstate>::select_test(parameters_input);
        else if constexpr (nstate > 1)
            return TestsFactory<dim,nstate-1>::create_test(parameters_input);
        else
            return nullptr;
    }
    else if constexpr (dim > 1)
    {
        //return TestsFactory<dim-1,nstate>::create_test(parameters_input);
        return nullptr;
    }
    else
    {
        return nullptr;
    }
}

// Will recursively create all the possible test sizes
//template class TestsFactory <PHILIP_DIM,1>;
//template class TestsFactory <PHILIP_DIM,2>;
//template class TestsFactory <PHILIP_DIM,3>;
//template class TestsFactory <PHILIP_DIM,4>;
template class TestsFactory <PHILIP_DIM,5>;

} // Tests namespace
} // PHiLiP namespace
