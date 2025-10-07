#include "flow_solver_factory.h"

#include <stdlib.h>

// all flow solver cases:
#include "flow_solver_cases/periodic_turbulence.h"
#include "flow_solver_cases/periodic_1D_unsteady.h"
#include "flow_solver_cases/periodic_entropy_tests.h"
#include "flow_solver_cases/1D_burgers_rewienski_snapshot.h"
#include "flow_solver_cases/1d_burgers_viscous_snapshot.h"
#include "flow_solver_cases/naca0012.h"
#include "flow_solver_cases/gaussian_bump.h"
#include "flow_solver_cases/non_periodic_cube_flow.h"
#include "flow_solver_cases/positivity_preserving_tests.h"
#include "flow_solver_cases/flow_solver_zero.h"

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
//                  FLOW SOLVER FACTORY
//=========================================================
template <int dim, int nspecies, int nstate>
std::unique_ptr < FlowSolver<dim, nspecies, nstate> >
FlowSolverFactory<dim, nspecies, nstate>
::select_flow_case(const Parameters::AllParameters *const parameters_input,
                   const dealii::ParameterHandler &parameter_handler_input)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex){
        if constexpr (dim==3 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicTurbulence<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence){
        if constexpr (dim==3 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicTurbulence<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot){
        if constexpr (dim==1 && nspecies==1 && nstate==dim) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<BurgersViscousSnapshot<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if constexpr (dim==1 && nspecies==1 && nstate==dim){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<BurgersRewienskiSnapshot<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::naca0012){
        if constexpr (dim==2 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<NACA0012<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::periodic_1D_unsteady){
        if constexpr (dim==1 && nspecies==1 && nstate==dim){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<Periodic1DUnsteady<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::isentropic_vortex){
        if constexpr (nstate==dim+2 && nspecies==1 && dim!=1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicEntropyTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::gaussian_bump){
        if constexpr (dim>1 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<GaussianBump<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::kelvin_helmholtz_instability){
        if constexpr (dim==2 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicEntropyTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::non_periodic_cube_flow){
        if constexpr (dim==2 && nspecies==1 && nstate==1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<NonPeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::sod_shock_tube){
        if constexpr (dim==1 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::leblanc_shock_tube){
        if constexpr (dim==1 && nspecies==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::shu_osher_problem) {
        if constexpr (dim==1 && nspecies==1 && nstate==dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::double_mach_reflection) {
        if constexpr (dim==2 && nspecies==1 && nstate==dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::shock_diffraction) {
        if constexpr (dim==2 && nspecies==1 && nstate==dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::astrophysical_jet) {
        if constexpr (dim==2 && nspecies==1 && nstate==dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::strong_vortex_shock_wave) {
        if constexpr (dim==2 && nspecies==1 && nstate==dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PositivityPreservingTests<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::advection_limiter) {
        if constexpr (dim<3 && nspecies==1 && nstate==1) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_limiter) {
        if constexpr (dim<3 && nspecies==1 && nstate==dim) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::low_density) {
        if constexpr (dim<3 && nspecies==1 && nstate==dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_acoustic_wave){
        if constexpr (dim==2 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_vortex_advection){
        if constexpr (dim==1 && (nspecies==2||nspecies==3) && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_high_temperature_vortex_advection){
        if constexpr (dim==1 && (nspecies==2||nspecies==3) && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::euler_vortex_advection){
        if constexpr (dim==1 && nspecies==1 && nstate==dim+2){ 
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_calorically_perfect_euler_vortex_advection){
        if constexpr (dim==1 && (nspecies==2||nspecies==3) && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_isentropic_euler_vortex){
        if constexpr (dim==2 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_two_dimensional_vortex_advection){
        if constexpr (dim==2 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_fuel_drop_advection){
        if constexpr (dim==2 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_three_dimensional_vortex_advection){
        if constexpr (dim==3 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_taylor_green_vortex){
        if constexpr (dim==3 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::multi_species_mixture_taylor_green_vortex){
        if constexpr (dim==3 && nspecies==2 && nstate==dim+2+nspecies-1){
            std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<PeriodicCubeFlow<dim, nspecies, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nspecies, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::flow_solver_zero && (nstate==dim+2+nspecies-1)) {
        std::shared_ptr<FlowSolverCaseBase<dim, nspecies, nstate>> flow_solver_case = std::make_shared<FlowSolverCaseZero<dim, nspecies, nstate>>(parameters_input);
        std::cout << "No flow case has been created for " << dim << "D and " << nspecies << " species. Implement a test for the conditions." << std::endl;
        std::abort();
    } 
    else {
        std::cout << "Invalid flow case. You probably forgot to add it to the list of flow cases in flow_solver_factory.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

template <int dim, int nspecies, int nstate>
std::unique_ptr< FlowSolverBase > FlowSolverFactory<dim, nspecies, nstate>
::create_flow_solver(const Parameters::AllParameters *const parameters_input,
                     const dealii::ParameterHandler &parameter_handler_input)
{
    // Recursive templating required because template parameters must be compile time constants
    // As a results, this recursive template initializes all possible dimensions with all possible nstate
    // without having 15 different if-else statements
    if(dim == parameters_input->dimension)
    {
        // Get the flow case type
        using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
        const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;
        if(flow_type == FlowCaseEnum::flow_solver_zero){
            return FlowSolverFactory<dim, nspecies, dim + 2 + (nspecies-1)>::select_flow_case(parameters_input,parameter_handler_input);            
        }
        // This template parameters dim and nstate match the runtime parameters
        // then create the selected flow case with template parameters dim and nstate
        // Otherwise, keep decreasing nstate and dim until it matches
        if(nstate == parameters_input->nstate) {
            return FlowSolverFactory<dim, nspecies, nstate>::select_flow_case(parameters_input,parameter_handler_input);
        } else if constexpr (nstate > 1) {
            return FlowSolverFactory<dim, nspecies, nstate-1>::create_flow_solver(parameters_input,parameter_handler_input);
        } else
            return nullptr;
    }
    else if constexpr (dim > 1)
    {
        //return FlowSolverFactory<dim-1,nstate>::create_flow_solver(parameters_input);
        return nullptr;
    }
    else
    {
        return nullptr;
    }
}

#if PHILIP_SPECIES==1
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, 1>;
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, 2>;
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, 3>;
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, 4>;
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, 5>;
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, 6>;
#else
template class FlowSolverFactory <PHILIP_DIM, PHILIP_SPECIES, PHILIP_DIM+2+(PHILIP_SPECIES-1)>;
#endif
} // FlowSolver namespace
} // PHiLiP namespace

