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
#include "flow_solver_cases/limiter_convergence_tests.h"
#include "flow_solver_cases/wall_cube.h"
#include "flow_solver_cases/flat_plate_2D.h"
#include "flow_solver_cases/airfoil_2D.h"

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
//                  FLOW SOLVER FACTORY
//=========================================================
template <int dim, int nstate, int sub_nstate>
std::unique_ptr < FlowSolver<dim,nstate,sub_nstate> >
FlowSolverFactory<dim,nstate,sub_nstate>
::select_flow_case(const Parameters::AllParameters *const parameters_input,
                   const dealii::ParameterHandler &parameter_handler_input)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::taylor_green_vortex){
        if constexpr (dim==3 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<PeriodicTurbulence<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence){
        if constexpr (dim==3 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<PeriodicTurbulence<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_viscous_snapshot){
        if constexpr (dim==1 && nstate==dim) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<BurgersViscousSnapshot<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_rewienski_snapshot){
        if constexpr (dim==1 && nstate==dim){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<BurgersRewienskiSnapshot<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::naca0012){
        if constexpr (dim==2 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NACA0012<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::periodic_1D_unsteady){
        if constexpr (dim==1 && nstate==dim){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<Periodic1DUnsteady<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::isentropic_vortex){
        if constexpr (nstate==dim+2 && dim!=1){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<PeriodicEntropyTests<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::gaussian_bump){
        if constexpr (dim>1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<GaussianBump<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::kelvin_helmholtz_instability){
        if constexpr (dim==2 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<PeriodicEntropyTests<dim,nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim,nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::non_periodic_cube_flow){
        if constexpr (dim==2 && nstate==1){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NonPeriodicCubeFlow<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::wall_distance_evaluation){
        if constexpr (dim==2 && nstate==1){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<WallCube<dim, nstate>>(parameters_input);
            //std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NACA0012<dim,nstate>>(parameters_input[0]);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::flat_plate_2D){
        if constexpr (dim==2 && nstate==1){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<FlatPlate2D<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        } else if constexpr (dim==2 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<FlatPlate2D<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        } else if constexpr (dim==2 && nstate==dim+3){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<FlatPlate2D<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::airfoil_2D){
        if constexpr (dim==2 && nstate==1){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<Airfoil2D<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        } else if constexpr (dim==2 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<Airfoil2D<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        } else if constexpr (dim==2 && nstate==dim+3){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<Airfoil2D<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate,1>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::sod_shock_tube){
        if constexpr (dim==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NonPeriodicCubeFlow<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::leblanc_shock_tube){
        if constexpr (dim==1 && nstate==dim+2){
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NonPeriodicCubeFlow<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::shu_osher_problem) {
        if constexpr (dim == 1 && nstate == dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<NonPeriodicCubeFlow<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::advection_limiter) {
        if constexpr (dim < 3 && nstate == 1) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<LimiterConvergenceTests<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::burgers_limiter) {
        if constexpr (dim < 3 && nstate == dim) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<LimiterConvergenceTests<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else if (flow_type == FlowCaseEnum::low_density_2d) {
        if constexpr (dim == 2 && nstate == dim + 2) {
            std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<LimiterConvergenceTests<dim, nstate>>(parameters_input);
            return std::make_unique<FlowSolver<dim, nstate>>(parameters_input, flow_solver_case, parameter_handler_input);
        }
    } else {
        std::cout << "Invalid flow case in a single parameters inputs. You probably forgot to add it to the list of flow cases in flow_solver_factory.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

template <int dim, int nstate, int sub_nstate>
std::unique_ptr < FlowSolver<dim,nstate,sub_nstate> >
FlowSolverFactory<dim,nstate,sub_nstate>
::select_flow_case(const std::vector<Parameters::AllParameters*> &parameters_input,
                   const std::vector<dealii::ParameterHandler> &parameter_handler_input)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = parameters_input[0]->flow_solver_param.flow_case_type;
    if (flow_type == FlowCaseEnum::flat_plate_2D){
        if constexpr (dim==2 && nstate==dim+3){
            if (parameters_input.size()==2){
                if (sub_nstate==1){
                    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<FlatPlate2D<dim, nstate>>(parameters_input[0]);
                    std::shared_ptr<FlowSolverCaseBase<dim, sub_nstate>> sub_flow_solver_case = std::make_shared<FlatPlate2D<dim, sub_nstate>>(parameters_input[1]);
                    return std::make_unique<FlowSolver<dim, nstate, sub_nstate>>(parameters_input, flow_solver_case, sub_flow_solver_case, parameter_handler_input);
                }else{
                    std::cout << "Invalid nstate for wall distance sub model." << std::endl;
                    std::cout << "Only p-Poisson wall distance model is supported for RANS." << std::endl;
                    std::cout << "Aborting..." << std::endl;
                    std::abort();
                }
            } else {
                std::cout << "Wall distance sub model is required for RANS calculation." << std::endl;
                std::cout << "Aborting..." << std::endl;
                std::abort();
            }
        }
    } else if (flow_type == FlowCaseEnum::airfoil_2D){
        if constexpr (dim==2 && nstate==dim+3){
            if (parameters_input.size()==2){
                if (sub_nstate==1){
                    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case = std::make_shared<Airfoil2D<dim, nstate>>(parameters_input[0]);
                    std::shared_ptr<FlowSolverCaseBase<dim, sub_nstate>> sub_flow_solver_case = std::make_shared<Airfoil2D<dim, sub_nstate>>(parameters_input[1]);
                    return std::make_unique<FlowSolver<dim, nstate, sub_nstate>>(parameters_input, flow_solver_case, sub_flow_solver_case, parameter_handler_input);
                }else{
                    std::cout << "Invalid nstate for wall distance sub model." << std::endl;
                    std::cout << "Only p-Poisson wall distance model is supported for RANS." << std::endl;
                    std::cout << "Aborting..." << std::endl;
                    std::abort();
                }
            } else {
                std::cout << "Wall distance sub model is required for RANS calculation." << std::endl;
                std::cout << "Aborting..." << std::endl;
                std::abort();
            }
        }
    } else {
        std::cout << "Invalid flow case in a vector of parameters inputs. You probably forgot to add it to the list of flow cases in flow_solver.cpp" << std::endl;
        std::abort();
    }
    return nullptr;
}

template<int dim, int nstate, int sub_nstate>
std::unique_ptr< PHiLiP::FlowSolver::FlowSolverBase > FlowSolverFactory<dim,nstate,sub_nstate>
::create_flow_solver(const std::vector<Parameters::AllParameters*> &parameters_input,
                     const std::vector<dealii::ParameterHandler> &parameter_handler_input)
{
    // Recursive templating required because template parameters must be compile time constants
    // As a results, this recursive template initializes all possible dimensions with all possible nstate
    // without having 15 different if-else statements
    if(dim == parameters_input[0]->dimension)
    {
        // This template parameters dim and nstate match the runtime parameters
        // then create the selected flow case with template parameters dim and nstate
        // Otherwise, keep decreasing nstate and dim until it matches
        if(nstate == parameters_input[0]->nstate) 
            if (parameters_input.size()==1)
                return FlowSolverFactory<dim,nstate,sub_nstate>::select_flow_case(parameters_input[0],parameter_handler_input[0]);
            else
                return FlowSolverFactory<dim,nstate,sub_nstate>::select_flow_case(parameters_input,parameter_handler_input);
        else if constexpr (nstate > 1)
            return FlowSolverFactory<dim,nstate-1,sub_nstate>::create_flow_solver(parameters_input,parameter_handler_input);
        else
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

//template class FlowSolverFactory <PHILIP_DIM,5,0>;
template class FlowSolverFactory <PHILIP_DIM,1,1>;
template class FlowSolverFactory <PHILIP_DIM,4,1>;
template class FlowSolverFactory <PHILIP_DIM,5,1>;

} // FlowSolver namespace
} // PHiLiP namespace

