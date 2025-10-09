#ifndef __PERIODIC_1D_UNSTEADY_H__
#define __PERIODIC_1D_UNSTEADY_H__

#include "periodic_cube_flow.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nspecies, int nstate>
class Periodic1DUnsteady : public PeriodicCubeFlow<dim, nspecies, nstate>
{
public:

    /// Constructor.
    explicit Periodic1DUnsteady(const Parameters::AllParameters *const parameters_input);

    /// Calculate energy as a matrix-vector product,  solution^T (M+K) solution
    double compute_energy(const std::shared_ptr <DGBase<dim, nspecies, double>> dg) const;

    /// Calculate numerical entropy.
    /// Here, is a wrapper for compute_energy. Used by tests.
    double get_numerical_entropy(const std::shared_ptr <DGBase<dim, nspecies, double>> dg) const;
protected:

    using FlowSolverCaseBase<dim, nspecies, nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;
    
    /// Filename for unsteady data
    std::string unsteady_data_table_filename_with_extension;
    
};


} // FlowSolver namespace
} // PHiLiP namespace
#endif
