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

    /// Calculate energy
    double compute_energy(const std::shared_ptr <DGBase<dim, double>> dg) const;
protected:

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;
    
    const int number_of_cells_per_direction; ///< Number of cells per direction for the grid
    const double domain_left; ///< Domain left-boundary value for generating the grid
    const double domain_right; ///< Domain right-boundary value for generating the grid
    const double domain_size; ///< Domain size (length in 1D, area in 2D, and volume in 3D)
    
    /// Generate grid
    /** Burgers test needs to deal with periodic bcs with use_periodic_bc flag
     * rather than straight_periodic_cube, which is used for advection
     */
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Filename for unsteady data
    std::string unsteady_data_table_filename_with_extension;
};


} // FlowSolver namespace
} // PHiLiP namespace
#endif
