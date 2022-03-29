#include <iostream>

#include <deal.II/grid/grid_out.h>

#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include "tests.h"
#include "grid_study.h"
#include "grid_refinement_study.h"
#include "burgers_stability.h"
#include "diffusion_exact_adjoint.h"
#include "euler_gaussian_bump.h"
#include "euler_gaussian_bump_enthalpy_check.h"
#include "euler_gaussian_bump_adjoint.h"
#include "euler_cylinder.h"
#include "euler_cylinder_adjoint.h"
#include "euler_vortex.h"
#include "euler_entropy_waves.h"
#include "advection_explicit_periodic.h"
#include "euler_split_inviscid_taylor_green_vortex.h"
#include "optimization_inverse_manufactured/optimization_inverse_manufactured.h"
#include "euler_bump_optimization.h"
#include "euler_naca0012_optimization.hpp"
#include "shock_1d.h"
#include "euler_naca0012.hpp"
#include "reduced_order_pod_adaptation.h"
#include "reduced_order.h"
#include "convection_diffusion_explicit_periodic.h"
#include "flow_solver.h"
#include "dual_weighted_residual_mesh_adaptation.h"

namespace PHiLiP {
namespace Tests {

using AllParam = Parameters::AllParameters;

TestsBase::TestsBase(Parameters::AllParameters const *const parameters_input)
    : all_parameters(parameters_input)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , n_mpi (dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
    , pcout(std::cout, mpi_rank==0)
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

template<int dim, int nstate, typename MeshType>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate,MeshType>
::select_mesh(const AllParam *const parameters_input) {
    using Mesh_enum = AllParam::MeshType;
    Mesh_enum mesh_type = parameters_input->mesh_type;

    if(mesh_type == Mesh_enum::default_triangulation) {
        #if PHILIP_DIM == 1
        return TestsFactory<dim,nstate,dealii::Triangulation<dim>>::select_test(parameters_input);
        #else
        return TestsFactory<dim,nstate,dealii::parallel::distributed::Triangulation<dim>>::select_test(parameters_input);
        #endif
    } else if(mesh_type == Mesh_enum::triangulation) {
        return TestsFactory<dim,nstate,dealii::Triangulation<dim>>::select_test(parameters_input);
    } else if(mesh_type == Mesh_enum::parallel_shared_triangulation) {
        return TestsFactory<dim,nstate,dealii::parallel::shared::Triangulation<dim>>::select_test(parameters_input);
    } else if(mesh_type == Mesh_enum::parallel_distributed_triangulation) {
        #if PHILIP_DIM == 1
        std::cout << "dealii::parallel::distributed::Triangulation is unavailible in 1D." << std::endl;
        #else
        return TestsFactory<dim,nstate,dealii::parallel::distributed::Triangulation<dim>>::select_test(parameters_input);
        #endif
    } else {
        std::cout << "Invalid mesh type." << std::endl;
    }

    return nullptr;
}

template<int dim, int nstate, typename MeshType>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate,MeshType>
::select_test(const AllParam *const parameters_input) {
    using Test_enum = AllParam::TestType;
    const Test_enum test_type = parameters_input->test_type;

    if(test_type == Test_enum::run_control) {
        return std::make_unique<GridStudy<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::grid_refinement_study) {
        return std::make_unique<GridRefinementStudy<dim,nstate,MeshType>>(parameters_input);
    } else if(test_type == Test_enum::burgers_energy_stability) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<BurgersEnergyStability<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::diffusion_exact_adjoint) {
        if constexpr (dim>=1 && nstate==1) return std::make_unique<DiffusionExactAdjoint<dim,nstate>>(parameters_input);
    } else if (test_type == Test_enum::advection_periodicity){
        if constexpr (dim == 2 && nstate == 1) return std::make_unique<AdvectionPeriodic<dim,nstate>> (parameters_input);
        //if constexpr (nstate == 1) return std::make_unique<AdvectionPeriodic<dim,nstate>> (parameters_input);
    } else if (test_type == Test_enum::convection_diffusion_periodicity){
        if constexpr (nstate == 1) return std::make_unique<ConvectionDiffusionPeriodic<dim,nstate>> (parameters_input);
    } else if(test_type == Test_enum::euler_gaussian_bump) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBump<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_gaussian_bump_enthalpy) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBumpEnthalpyCheck<dim,nstate>>(parameters_input);
    //} else if(test_type == Test_enum::euler_gaussian_bump_adjoint){
    //   if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBumpAdjoint<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_cylinder) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerCylinder<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_cylinder_adjoint) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerCylinderAdjoint<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_vortex) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerVortex<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_entropy_waves) {
        if constexpr (dim>=2 && nstate==PHILIP_DIM+2) return std::make_unique<EulerEntropyWaves<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_split_taylor_green) {
     if constexpr (dim==3 && nstate == dim+2) return std::make_unique<EulerTaylorGreen<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::optimization_inverse_manufactured) {
        return std::make_unique<OptimizationInverseManufactured<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_bump_optimization) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerBumpOptimization<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_naca_optimization) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerNACAOptimization<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::shock_1d) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<Shock1D<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::reduced_order) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<ReducedOrder<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::POD_adaptation) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<ReducedOrderPODAdaptation<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_naca0012) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerNACA0012<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::flow_solver) {
        if constexpr ((dim==3 && nstate==dim+2) || (dim==1 && nstate==1)) return FlowSolverFactory<dim,nstate>::create_FlowSolver(parameters_input);
    } else if(test_type == Test_enum::dual_weighted_residual_mesh_adaptation) {
        if constexpr (dim > 1)  return std::make_unique<DualWeightedResidualMeshAdaptation<dim, nstate>>(parameters_input);
    } else {
        std::cout << "Invalid test. You probably forgot to add it to the list of tests in tests.cpp" << std::endl;
        std::abort();
    }

    return nullptr;
}

template<int dim, int nstate, typename MeshType>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate,MeshType>
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
            return TestsFactory<dim,nstate>::select_mesh(parameters_input);
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
//template class TestsFactory <PHILIP_DIM,5>;

template class TestsFactory <PHILIP_DIM,5,dealii::Triangulation<PHILIP_DIM>>;
template class TestsFactory <PHILIP_DIM,5,dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class TestsFactory <PHILIP_DIM,5,dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // Tests namespace
} // PHiLiP namespace
