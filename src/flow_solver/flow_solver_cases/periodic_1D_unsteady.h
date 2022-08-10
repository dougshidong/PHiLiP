#ifndef __PERIODIC_1D_UNSTEADY_H__
#define __PERIODIC_1D_UNSTEADY_H__

#include "periodic_cube_flow.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class Periodic1DUnsteady : public PeriodicCubeFlow<dim,nstate>
{
public:

    /// Constructor.
    Periodic1DUnsteady(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~Periodic1DUnsteady() {};

    /// Calculate energy
    double compute_energy_collocated(const std::shared_ptr <DGBase<dim, double>> dg) const;
protected:

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
