#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace Tests {

template<int dim, int nstate>
FlowSolverCaseBase<dim, nstate>::FlowSolverCaseBase(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : all_param(*parameters_input)
        , mpi_communicator(MPI_COMM_WORLD)
        , mpi_rank(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
        , n_mpi(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
        , pcout(std::cout, mpi_rank==0)
        {}


template <int dim, int nstate>
void FlowSolverCaseBase<dim,nstate>::display_flow_solver_setup() const
{
    pcout << "- Polynomial degree: " << this->all_param.grid_refinement_study_param.poly_degree << std::endl;
    pcout << "- Final time: " << this->all_param.flow_solver_param.final_time << std::endl;
}

template <int dim, int nstate>
double FlowSolverCaseBase<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> /*dg*/) const
{
    pcout << "Using initial time step in ODE parameters." <<std::endl;
    return all_param.ode_solver_param.initial_time_step;
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
        const std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/) const
{
    // do nothing by default
}


#if PHILIP_DIM==1
        template class FlowSolverCaseBase<PHILIP_DIM,PHILIP_DIM>;
#endif

#if PHILIP_DIM==3
        template class FlowSolverCaseBase<PHILIP_DIM,PHILIP_DIM+2>;
#endif

}
}