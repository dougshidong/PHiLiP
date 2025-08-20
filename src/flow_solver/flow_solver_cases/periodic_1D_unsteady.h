#ifndef __PERIODIC_1D_UNSTEADY_H__
#define __PERIODIC_1D_UNSTEADY_H__

#include "periodic_cube_flow.h"
#include "ode_solver/runge_kutta_base.h"
//#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/runge_kutta_ode_solver.h"
#include "ode_solver/ode_solver_factory.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class Periodic1DUnsteady : public PeriodicCubeFlow<dim,nstate>
{
public:

    /// Constructor.
    explicit Periodic1DUnsteady(const Parameters::AllParameters *const parameters_input);

    /// Calculate energy as a matrix-vector product,  solution^T (M+K) solution
    double compute_energy(const std::shared_ptr <DGBase<dim, double>> dg) const;

    /// Calculate numerical entropy.
    /// Here, is a wrapper for compute_energy. Used by tests.
    double get_numerical_entropy(const std::shared_ptr <DGBase<dim, double>> dg) const;

    virtual void perk_partitioning(std::shared_ptr <DGBase<dim, double>> dg, std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver) const;

    //std::shared_ptr<ODE::ODESolverBase<dim, double>> ode_solver;

protected:
    mutable dealii::LinearAlgebra::distributed::Vector<int> locations_to_evaluate_rhs;
    mutable int evaluate_until_this_index;

    using FlowSolverCaseBase<dim,nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;
    
    /// Filename for unsteady data
    std::string unsteady_data_table_filename_with_extension;
    
};


} // FlowSolver namespace
} // PHiLiP namespace
#endif
