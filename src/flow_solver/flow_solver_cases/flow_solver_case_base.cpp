#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

template<int dim, int nstate>
FlowSolverCaseBase<dim, nstate>::FlowSolverCaseBase(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : initial_condition_function(InitialConditionFactory<dim, nstate, double>::create_InitialConditionFunction(parameters_input))
        , all_param(*parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , n_mpi(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
        {}

template<int dim, int nstate>
std::string FlowSolverCaseBase<dim, nstate>::get_pde_string() const
{
    using PDE_enum      = Parameters::AllParameters::PartialDifferentialEquation;
    using Model_enum    = Parameters::AllParameters::ModelType;
    using SGSModel_enum = Parameters::PhysicsModelParam::SubGridScaleModel;
    
    const PDE_enum pde_type = this->all_param.pde_type;
    std::string pde_string;
    if (pde_type == PDE_enum::advection)            {pde_string = "advection";}
    if (pde_type == PDE_enum::advection_vector)     {pde_string = "advection_vector";}
    if (pde_type == PDE_enum::diffusion)            {pde_string = "diffusion";}
    if (pde_type == PDE_enum::convection_diffusion) {pde_string = "convection_diffusion";}
    if (pde_type == PDE_enum::burgers_inviscid)     {pde_string = "burgers_inviscid";}
    if (pde_type == PDE_enum::burgers_viscous)      {pde_string = "burgers_viscous";}
    if (pde_type == PDE_enum::burgers_rewienski)    {pde_string = "burgers_rewienski";}
    if (pde_type == PDE_enum::mhd)                  {pde_string = "mhd";}
    if (pde_type == PDE_enum::euler)                {pde_string = "euler";}
    if (pde_type == PDE_enum::navier_stokes)        {pde_string = "navier_stokes";}
    if (pde_type == PDE_enum::physics_model) {
        pde_string = "physics_model";
        // add the model name + sub model name (if applicable)
        const Model_enum model = this->all_param.model_type;
        std::string model_string = "WARNING: invalid model";
        if(model == Model_enum::large_eddy_simulation) {
            // assign model string
            model_string = "large_eddy_simulation";
            // sub-grid scale (SGS)
            const SGSModel_enum sgs_model = this->all_param.physics_model_param.SGS_model_type;
            std::string sgs_model_string = "WARNING: invalid SGS model";
            // assign SGS model string
            if     (sgs_model==SGSModel_enum::smagorinsky) sgs_model_string = "smagorinsky";
            else if(sgs_model==SGSModel_enum::wall_adaptive_local_eddy_viscosity) sgs_model_string = "wall_adaptive_local_eddy_viscosity";
            else if(sgs_model==SGSModel_enum::vreman) sgs_model_string = "vreman";
            pde_string += std::string(" (Model: ") + model_string + std::string(", SGS Model: ") + sgs_model_string + std::string(")");
        }
        if(pde_string == "physics_model") pde_string += std::string(" (Model: ") + model_string + std::string(")");
    }
    
    return pde_string;
}

template<int dim, int nstate>
std::string FlowSolverCaseBase<dim, nstate>::get_flow_case_string() const
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_case_type = this->all_param.flow_solver_param.flow_case_type;
    
    std::string flow_case_string;
    if (flow_case_type == FlowCaseEnum::taylor_green_vortex)        {flow_case_string = "taylor_green_vortex";}
    if (flow_case_type == FlowCaseEnum::burgers_viscous_snapshot)   {flow_case_string = "burgers_viscous_snapshot";}
    if (flow_case_type == FlowCaseEnum::burgers_rewienski_snapshot) {flow_case_string = "burgers_rewienski_snapshot";}
    if (flow_case_type == FlowCaseEnum::naca0012)                   {flow_case_string = "naca0012";}
    if (flow_case_type == FlowCaseEnum::periodic_1D_unsteady)       {flow_case_string = "periodic_1D_unsteady";}
    if (flow_case_type == FlowCaseEnum::gaussian_bump)              {flow_case_string = "gaussian_bump";}
    
    return flow_case_string;
}

template <int dim, int nstate>
void FlowSolverCaseBase<dim,nstate>::display_flow_solver_setup(std::shared_ptr<DGBase<dim,double>> dg) const
{
    const std::string pde_string = this->get_pde_string();
    pcout << "- PDE Type: " << pde_string << " " << "(dim=" << dim << ", nstate=" << nstate << ")" << std::endl;
    
    pcout << "- Polynomial degree: " << this->all_param.flow_solver_param.poly_degree << std::endl;
    
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double number_of_degrees_of_freedom_per_dim = pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    pcout << "- Degrees of freedom (per state): " << number_of_degrees_of_freedom_per_state << " " << "(" << number_of_degrees_of_freedom_per_dim << " per state per dim)" << std::endl;
    pcout << "- Number of active cells: " << dg->triangulation->n_global_active_cells() << std::endl;

    const std::string flow_case_string = this->get_flow_case_string();
    pcout << "- Flow case: " << flow_case_string << " " << std::flush;
    if(this->all_param.flow_solver_param.steady_state == true) {
        pcout << "(Steady state)" << std::endl;
    } else {
        pcout << "(Unsteady)" << std::endl;
        pcout << "- - Final time: " << this->all_param.flow_solver_param.final_time << std::endl;
    }

    this->display_additional_flow_case_specific_parameters();
}

template <int dim, int nstate>
void FlowSolverCaseBase<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> /*dg*/) const
{
    // Do nothing
}

template <int dim, int nstate>
double FlowSolverCaseBase<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> /*dg*/) const
{
    pcout << "Using initial time step in ODE parameters." <<std::endl;
    return all_param.ode_solver_param.initial_time_step;
}

template <int dim, int nstate>
double FlowSolverCaseBase<dim,nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> /*dg*/) const
{
    pcout << "ERROR: Base definition for get_adaptive_time_step() has not yet been implemented. " <<std::flush;
    std::abort();
    return 0.0;
}

template <int dim, int nstate>
void FlowSolverCaseBase<dim, nstate>::steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> /*dg*/) const
{
    // do nothing by default
}

template <int dim, int nstate>
void FlowSolverCaseBase<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int /*current_iteration*/,
        const double /*current_time*/,
        const std::shared_ptr <DGBase<dim, double>> /*dg*/,
        const std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/)
{
    // do nothing by default
}

template <int dim, int nstate>
void FlowSolverCaseBase<dim, nstate>::add_value_to_data_table(
    const double value,
    const std::string value_string,
    const std::shared_ptr <dealii::TableHandler> data_table) const
{
    data_table->add_value(value_string, value);
    data_table->set_precision(value_string, 16);
    data_table->set_scientific(value_string, true);
}

#if PHILIP_DIM==1
template class FlowSolverCaseBase<PHILIP_DIM,PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
template class FlowSolverCaseBase<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace
