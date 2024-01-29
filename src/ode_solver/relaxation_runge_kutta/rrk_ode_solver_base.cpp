#include "rrk_ode_solver_base.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
RRKODESolverBase<dim,real,MeshType>::RRKODESolverBase(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RKNumEntropy<dim,real,MeshType>(rk_tableau_input)
{
    relaxation_parameter = 1.0;
}

template <int dim, typename real, typename MeshType>
double RRKODESolverBase<dim,real,MeshType>::modify_time_step(const double dt)
{
    // Update solution such that dg is holding u^n (not last stage of RK)
    this->dg->solution = this->solution_update;
    this->dg->assemble_residual();

    relaxation_parameter = compute_relaxation_parameter(dt);
    this->dg->relaxation_parameter = relaxation_parameter;

    if (relaxation_parameter < 0.5 ){
        this->pcout << "RRK failed to find a reasonable relaxation factor. Aborting..." << std::endl;
        relaxation_parameter=1.0;
        std::abort();
    }
    dt *= relaxation_parameter;
    return dt;
}

template class RRKODESolverBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RRKODESolverBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
