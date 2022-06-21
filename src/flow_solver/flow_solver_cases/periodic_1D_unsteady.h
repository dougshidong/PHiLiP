#ifndef __PERIODIC_1D_UNSTEADY_H__
#define __PERIODIC_1D_UNSTEADY_H__

// for FlowSolver class:
#include "periodic_cube_flow.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>

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
protected:

    /// Compute the desired unsteady data and write it to a table
    /** Currently only prints to console. 
     * In the future, will be modified to also write to table if there is unsteady
     * data of interest
     */
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

};


} // FlowSolver namespace
} // PHiLiP namespace
#endif
