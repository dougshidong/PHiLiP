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
#include "TGV_scaling.h"
#include "optimization_inverse_manufactured/optimization_inverse_manufactured.h"
#include "euler_bump_optimization.h"
#include "euler_naca0012_optimization.hpp"
#include "shock_1d.h"
#include "euler_naca0012.hpp"
#include "reduced_order.h"
#include "convection_diffusion_explicit_periodic.h"
#include "dual_weighted_residual_mesh_adaptation.h"
#include "anisotropic_mesh_adaptation_cases.h"
#include "pod_adaptive_sampling_run.h"
#include "pod_adaptive_sampling_testing.h"
#include "taylor_green_vortex_energy_check.h"
#include "taylor_green_vortex_restart_check.h"
#include "time_refinement_study.h"
#include "time_refinement_study_reference.h"
#include "h_refinement_study_isentropic_vortex.h"
#include "rrk_numerical_entropy_conservation_check.h"
#include "euler_entropy_conserving_split_forms_check.h"
#include "homogeneous_isotropic_turbulence_initialization_check.h"
#include "khi_robustness.h"
#include "bound_preserving_limiter_tests.h"
#include "naca0012_unsteady_check_quick.h"
#include "build_NNLS_problem.h"
#include "hyper_reduction_comparison.h"
#include "hyper_adaptive_sampling_run.h"
#include "hyper_reduction_post_sampling.h"
#include "ROM_error_post_sampling.h"
#include "HROM_error_post_sampling.h"
#include "hyper_adaptive_sampling_new_error.h"
#include "halton_sampling_run.h"

namespace PHiLiP {
namespace Tests {

using AllParam = Parameters::AllParameters;

TestsBase::TestsBase(Parameters::AllParameters const *const parameters_input)
    : all_parameters(parameters_input)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
    , n_mpi(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
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

std::string TestsBase::get_pde_string(const Parameters::AllParameters *const param) const
{
    using PDE_enum       = Parameters::AllParameters::PartialDifferentialEquation;
    using Model_enum     = PHiLiP::Parameters::AllParameters::ModelType;
    using SGSModel_enum  = PHiLiP::Parameters::PhysicsModelParam::SubGridScaleModel;
    using RANSModel_enum = PHiLiP::Parameters::PhysicsModelParam::ReynoldsAveragedNavierStokesModel;
    
    const PDE_enum pde_type = param->pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::advection)            {pde_string = "advection";}
    if (pde_type == PDE_enum::advection_vector)     {pde_string = "advection_vector";}
    if (pde_type == PDE_enum::diffusion)            {pde_string = "diffusion";}
    if (pde_type == PDE_enum::convection_diffusion) {pde_string = "convection_diffusion";}
    if (pde_type == PDE_enum::burgers_inviscid)     {pde_string = "burgers_inviscid";}
    if (pde_type == PDE_enum::burgers_viscous)      {pde_string = "burgers_viscous";}
    if (pde_type == PDE_enum::burgers_rewienski)    {pde_string = "burgers_rewienski";}
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}
    if (pde_type == PDE_enum::physics_model) {
        pde_string = "physics_model";
        // add the model name + sub model name (if applicable)
        const Model_enum model = param->model_type;
        std::string model_string = "WARNING: invalid model";
        if(model == Model_enum::large_eddy_simulation) {
            // assign model string
            model_string = "large_eddy_simulation";
            // sub-grid scale (SGS)
            const SGSModel_enum sgs_model = param->physics_model_param.SGS_model_type;
            std::string sgs_model_string = "WARNING: invalid SGS model";
            // assign SGS model string
            if     (sgs_model==SGSModel_enum::smagorinsky) sgs_model_string = "smagorinsky";
            else if(sgs_model==SGSModel_enum::wall_adaptive_local_eddy_viscosity) sgs_model_string = "wall_adaptive_local_eddy_viscosity";
            else if(sgs_model==SGSModel_enum::vreman) sgs_model_string = "vreman";
            pde_string += std::string(" (Model: ") + model_string + std::string(", SGS Model: ") + sgs_model_string + std::string(")");
        }
        else if(model == Model_enum::reynolds_averaged_navier_stokes) {
            // assign model string
            model_string = "reynolds_averaged_navier_stokes";
            // reynolds-averaged navier-stokes (RANS)
            const RANSModel_enum rans_model = param->physics_model_param.RANS_model_type;
            std::string rans_model_string = "WARNING: invalid RANS model";
            // assign RANS model string
            if     (rans_model==RANSModel_enum::SA_negative) rans_model_string = "SA_negative";
            pde_string += std::string(" (Model: ") + model_string + std::string(", RANS Model: ") + rans_model_string + std::string(")");
        }
        if(pde_string == "physics_model") pde_string += std::string(" (Model: ") + model_string + std::string(")");
    }
    return pde_string;
}

std::string TestsBase::get_conv_num_flux_string(const Parameters::AllParameters *const param) const
{
    using CNF_enum = Parameters::AllParameters::ConvectiveNumericalFlux;
    const CNF_enum CNF_type = param->conv_num_flux_type;
    std::string conv_num_flux_string;
    if (CNF_type == CNF_enum::lax_friedrichs) {conv_num_flux_string = "lax_friedrichs";}
    if (CNF_type == CNF_enum::roe)            {conv_num_flux_string = "roe";}
    if (CNF_type == CNF_enum::l2roe)          {conv_num_flux_string = "l2roe";}
    if (CNF_type == CNF_enum::central_flux)   {conv_num_flux_string = "central_flux";}
    if (CNF_type == CNF_enum::two_point_flux) {conv_num_flux_string = "two_point_flux";}
    if (CNF_type == CNF_enum::two_point_flux_with_lax_friedrichs_dissipation) {
        conv_num_flux_string = "two_point_flux_with_lax_friedrichs_dissipation";
    } if (CNF_type == CNF_enum::two_point_flux_with_roe_dissipation) {
        conv_num_flux_string = "two_point_flux_with_roe_dissipation";
    } if (CNF_type == CNF_enum::two_point_flux_with_l2roe_dissipation) {
        conv_num_flux_string = "two_point_flux_with_l2roe_dissipation";
    }
    
    return conv_num_flux_string;
}

std::string TestsBase::get_diss_num_flux_string(const Parameters::AllParameters *const param) const
{
    using DNF_enum = Parameters::AllParameters::DissipativeNumericalFlux;
    const DNF_enum DNF_type = param->diss_num_flux_type;
    std::string diss_num_flux_string;
    if (DNF_type == DNF_enum::symm_internal_penalty) {diss_num_flux_string = "symm_internal_penalty";}
    if (DNF_type == DNF_enum::bassi_rebay_2)         {diss_num_flux_string = "bassi_rebay_2";}
    return diss_num_flux_string;
}

std::string TestsBase::get_manufactured_solution_string(const Parameters::AllParameters *const param) const
{
    using ManParam = Parameters::ManufacturedConvergenceStudyParam;
    ManParam manu_grid_conv_param = param->manufactured_convergence_study_param;
    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
    const ManufacturedSolutionEnum MS_type = manu_grid_conv_param.manufactured_solution_param.manufactured_solution_type;
    std::string manufactured_solution_string;
    if (MS_type == ManufacturedSolutionEnum::sine_solution)           {manufactured_solution_string = "sine_solution";}
    if (MS_type == ManufacturedSolutionEnum::cosine_solution)         {manufactured_solution_string = "cosine_solution";}
    if (MS_type == ManufacturedSolutionEnum::additive_solution)       {manufactured_solution_string = "additive_solution";}
    if (MS_type == ManufacturedSolutionEnum::exp_solution)            {manufactured_solution_string = "exp_solution";}
    if (MS_type == ManufacturedSolutionEnum::poly_solution)           {manufactured_solution_string = "poly_solution";}
    if (MS_type == ManufacturedSolutionEnum::even_poly_solution)      {manufactured_solution_string = "even_poly_solution";}
    if (MS_type == ManufacturedSolutionEnum::atan_solution)           {manufactured_solution_string = "atan_solution";}
    if (MS_type == ManufacturedSolutionEnum::boundary_layer_solution) {manufactured_solution_string = "boundary_layer_solution";}
    if (MS_type == ManufacturedSolutionEnum::s_shock_solution)        {manufactured_solution_string = "s_shock_solution";}
    if (MS_type == ManufacturedSolutionEnum::quadratic_solution)      {manufactured_solution_string = "quadratic_solution";}
    if (MS_type == ManufacturedSolutionEnum::example_solution)        {manufactured_solution_string = "example_solution";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_1)        {manufactured_solution_string = "navah_solution_1";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_2)        {manufactured_solution_string = "navah_solution_2";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_3)        {manufactured_solution_string = "navah_solution_3";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_4)        {manufactured_solution_string = "navah_solution_4";}
    if (MS_type == ManufacturedSolutionEnum::navah_solution_5)        {manufactured_solution_string = "navah_solution_5";}
    return manufactured_solution_string;
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
::select_mesh(const AllParam *const parameters_input,
              dealii::ParameterHandler &parameter_handler_input) {
    using Mesh_enum = AllParam::MeshType;
    Mesh_enum mesh_type = parameters_input->mesh_type;

    if(mesh_type == Mesh_enum::default_triangulation) {
        #if PHILIP_DIM == 1
        return TestsFactory<dim,nstate,dealii::Triangulation<dim>>::select_test(parameters_input,parameter_handler_input);
        #else
        return TestsFactory<dim,nstate,dealii::parallel::distributed::Triangulation<dim>>::select_test(parameters_input,parameter_handler_input);
        #endif
    } else if(mesh_type == Mesh_enum::triangulation) {
        return TestsFactory<dim,nstate,dealii::Triangulation<dim>>::select_test(parameters_input,parameter_handler_input);
    } else if(mesh_type == Mesh_enum::parallel_shared_triangulation) {
        return TestsFactory<dim,nstate,dealii::parallel::shared::Triangulation<dim>>::select_test(parameters_input,parameter_handler_input);
    } else if(mesh_type == Mesh_enum::parallel_distributed_triangulation) {
        #if PHILIP_DIM == 1
        std::cout << "dealii::parallel::distributed::Triangulation is unavailible in 1D." << std::endl;
        #else
        return TestsFactory<dim,nstate,dealii::parallel::distributed::Triangulation<dim>>::select_test(parameters_input,parameter_handler_input);
        #endif
    } else {
        std::cout << "Invalid mesh type." << std::endl;
    }

    return nullptr;
}

template<int dim, int nstate, typename MeshType>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate,MeshType>
::select_test(const AllParam *const parameters_input,
              dealii::ParameterHandler &parameter_handler_input) {
    using Test_enum = AllParam::TestType;
    const Test_enum test_type = parameters_input->test_type;

    // prevent warnings for when a create_FlowSolver is not being called (explicit and implicit cases)
    if((test_type != Test_enum::finite_difference_sensitivity) &&
       (test_type != Test_enum::taylor_green_vortex_energy_check) && 
       (test_type != Test_enum::taylor_green_vortex_restart_check)) {
        (void) parameter_handler_input;
    } else if (!((dim==3 && nstate==dim+2) || (dim==1 && nstate==1))) {
        (void) parameter_handler_input;
    }

    if(test_type == Test_enum::run_control) { // TO DO: rename to grid_study
        return std::make_unique<GridStudy<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::grid_refinement_study) {
        return std::make_unique<GridRefinementStudy<dim,nstate,MeshType>>(parameters_input);
    } else if(test_type == Test_enum::burgers_energy_stability) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<BurgersEnergyStability<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::diffusion_exact_adjoint) {
        if constexpr (dim>=1 && nstate==1) return std::make_unique<DiffusionExactAdjoint<dim,nstate>>(parameters_input);
    } else if (test_type == Test_enum::advection_periodicity){
        if constexpr (nstate == 1) return std::make_unique<AdvectionPeriodic<dim,nstate>> (parameters_input);
    } else if (test_type == Test_enum::convection_diffusion_periodicity){
        if constexpr (nstate == 1) return std::make_unique<ConvectionDiffusionPeriodic<dim,nstate>> (parameters_input);
    } else if(test_type == Test_enum::euler_gaussian_bump) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBump<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::euler_gaussian_bump_enthalpy) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerGaussianBumpEnthalpyCheck<dim,nstate>>(parameters_input, parameter_handler_input);
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
    } else if(test_type == Test_enum::taylor_green_scaling) {
        if constexpr (dim==3 && nstate == dim+2) return std::make_unique<EulerTaylorGreenScaling<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::optimization_inverse_manufactured) {
        return std::make_unique<OptimizationInverseManufactured<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_bump_optimization) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerBumpOptimization<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::euler_naca_optimization) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerNACAOptimization<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::shock_1d) {
        if constexpr (dim==1 && nstate==1) return std::make_unique<Shock1D<dim,nstate>>(parameters_input);
    } else if(test_type == Test_enum::reduced_order) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1)) return std::make_unique<ReducedOrder<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::POD_adaptive_sampling_run) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1)) return std::make_unique<AdaptiveSamplingRun<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::adaptive_sampling_testing) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1)) return std::make_unique<AdaptiveSamplingTesting<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::euler_naca0012) {
        if constexpr (dim==2 && nstate==dim+2) return std::make_unique<EulerNACA0012<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::dual_weighted_residual_mesh_adaptation) {
        if constexpr (dim==2 && nstate==1)  return std::make_unique<DualWeightedResidualMeshAdaptation<dim, nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::anisotropic_mesh_adaptation) {
        if constexpr( (dim==2 && nstate==1) || (dim==2 && nstate==dim+2)) return std::make_unique<AnisotropicMeshAdaptationCases<dim, nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::taylor_green_vortex_energy_check) {
        if constexpr (dim==3 && nstate==dim+2) return std::make_unique<TaylorGreenVortexEnergyCheck<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::taylor_green_vortex_restart_check) {
        if constexpr (dim==3 && nstate==dim+2) return std::make_unique<TaylorGreenVortexRestartCheck<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::homogeneous_isotropic_turbulence_initialization_check){
        if constexpr (dim==3 && nstate==dim+2) return std::make_unique<HomogeneousIsotropicTurbulenceInitializationCheck<dim,nstate>>(parameters_input,parameter_handler_input);
    } else if(test_type == Test_enum::time_refinement_study) {
        if constexpr (dim==1 && nstate==1)  return std::make_unique<TimeRefinementStudy<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::h_refinement_study_isentropic_vortex) {
        if constexpr (dim+2==nstate && dim!=1)  return std::make_unique<HRefinementStudyIsentropicVortex<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::time_refinement_study_reference) {
        if constexpr (dim==1 && nstate==1)  return std::make_unique<TimeRefinementStudyReference<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::rrk_numerical_entropy_conservation_check) {
        if constexpr ((dim==1 && nstate==1) || (dim==3 && nstate==dim+2))  return std::make_unique<RRKNumericalEntropyConservationCheck<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::euler_entropy_conserving_split_forms_check) {
        if constexpr (dim==3 && nstate==dim+2)  return std::make_unique<EulerSplitEntropyCheck<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::khi_robustness) {
        if constexpr (dim==2 && nstate==dim+2)  return std::make_unique<KHIRobustness<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::build_NNLS_problem) {
        if constexpr (dim==1 && nstate==1)  return std::make_unique<BuildNNLSProblem<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::hyper_reduction_comparison) {
        if constexpr (dim==1 && nstate==1)  return std::make_unique<HyperReductionComparison<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::hyper_adaptive_sampling_run) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1))  return std::make_unique<HyperAdaptiveSamplingRun<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::hyper_reduction_post_sampling) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1))  return std::make_unique<HyperReductionPostSampling<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::ROM_error_post_sampling) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1))  return std::make_unique<ROMErrorPostSampling<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::HROM_error_post_sampling) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1))  return std::make_unique<HROMErrorPostSampling<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::hyper_adaptive_sampling_new_error) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1))  return std::make_unique<HyperAdaptiveSamplingNewError<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::halton_sampling_run) {
        if constexpr ((dim==2 && nstate==dim+2) || (dim==1 && nstate==1))  return std::make_unique<HaltonSamplingRun<dim,nstate>>(parameters_input, parameter_handler_input);
    } else if (test_type == Test_enum::advection_limiter) {
        if constexpr (nstate == 1 && dim < 3) return std::make_unique<BoundPreservingLimiterTests<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if (test_type == Test_enum::burgers_limiter) {
        if constexpr (nstate == dim && dim < 3) return std::make_unique<BoundPreservingLimiterTests<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::low_density) {
        if constexpr (dim<3 && nstate==dim+2)  return std::make_unique<BoundPreservingLimiterTests<dim, nstate>>(parameters_input, parameter_handler_input);
    } else if(test_type == Test_enum::naca0012_unsteady_check_quick){
        if constexpr (dim==2 && nstate==dim+2)  return std::make_unique<NACA0012UnsteadyCheckQuick<dim, nstate>>(parameters_input, parameter_handler_input);
    } else {
        std::cout << "Invalid test. You probably forgot to add it to the list of tests in tests.cpp" << std::endl;
        std::abort();
    }

    return nullptr;
}

template<int dim, int nstate, typename MeshType>
std::unique_ptr< TestsBase > TestsFactory<dim,nstate,MeshType>
::create_test(AllParam const *const parameters_input,
              dealii::ParameterHandler &parameter_handler_input)
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
            return TestsFactory<dim,nstate>::select_mesh(parameters_input,parameter_handler_input);
        else if constexpr (nstate > 1)
            return TestsFactory<dim,nstate-1>::create_test(parameters_input,parameter_handler_input);
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
