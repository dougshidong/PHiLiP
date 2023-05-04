#include "rrk_ode_solver_base.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, int n_rk_stages, typename MeshType>
RRKODESolverBase<dim,real,n_rk_stages,MeshType>::RRKODESolverBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input,
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RungeKuttaODESolver<dim,real,n_rk_stages,MeshType>(dg_input,rk_tableau_input)
{
    relaxation_parameter = 1.0;
    this->dg->FR_entropy_cumulative=0.0;
}

template <int dim, typename real, int n_rk_stages, typename MeshType>
void RRKODESolverBase<dim,real,n_rk_stages,MeshType>::modify_time_step(real &dt)
{
    // Update solution such that dg is holding u^n (not last stage of RK)
    this->dg->solution = this->solution_update;
    this->dg->assemble_residual();

    relaxation_parameter = compute_relaxation_parameter(dt);

    if (relaxation_parameter < 0.5 ){
        this->pcout << "RRK failed to find a reasonable relaxation factor. Aborting..." << std::endl;
        relaxation_parameter=1.0;
        std::abort();
    }
    dt *= relaxation_parameter;
}

template class RRKODESolverBase<PHILIP_DIM, double,1, dealii::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,2, dealii::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,3, dealii::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,4, dealii::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,1, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,2, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,3, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double,4, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RRKODESolverBase<PHILIP_DIM, double,1, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RRKODESolverBase<PHILIP_DIM, double,2, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RRKODESolverBase<PHILIP_DIM, double,3, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
    template class RRKODESolverBase<PHILIP_DIM, double,4, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
