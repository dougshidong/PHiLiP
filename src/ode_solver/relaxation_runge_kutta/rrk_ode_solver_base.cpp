#include "rrk_ode_solver_base.h"

namespace PHiLiP {
namespace ODE {

template <int dim, typename real, typename MeshType>
RRKODESolverBase<dim,real,MeshType>::RRKODESolverBase(
            std::shared_ptr<RKTableauBase<dim,real,MeshType>> rk_tableau_input)
        : RKNumEntropy<dim,real,MeshType>(rk_tableau_input)
{
    // Do nothing
}

template <int dim, typename real, typename MeshType>
real RRKODESolverBase<dim,real,MeshType>::modify_time_step(const real dt,
            std::shared_ptr<DGBase<dim,real,MeshType>> dg,
            std::vector<dealii::LinearAlgebra::distributed::Vector<double>> &rk_stage,
            dealii::LinearAlgebra::distributed::Vector<double> &solution_update)
{
    // Update solution such that dg is holding u^n (not last stage of RK)
    dg->solution = solution_update;
    dg->assemble_residual();

    relaxation_parameter = compute_relaxation_parameter(dt, dg, rk_stage, solution_update);
    dg->relaxation_parameter = relaxation_parameter;

    if (relaxation_parameter < 0.5 ){
        this->pcout << "RRK failed to find a reasonable relaxation factor. Aborting..." << std::endl;
        relaxation_parameter=1.0;
        std::abort();
    }
    const real dt_modified = dt * relaxation_parameter;
    return dt_modified;
}

template class RRKODESolverBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM> >;
template class RRKODESolverBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM> >;
#if PHILIP_DIM != 1
    template class RRKODESolverBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM> >;
#endif

} // ODESolver namespace
} // PHiLiP namespace
