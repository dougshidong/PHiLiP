#ifndef __CHANNEL_FLOW_H__
#define __CHANNEL_FLOW_H__

#include "flow_solver_case_base.h"
#include "dg/dg.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class ChannelFlow : public FlowSolverCaseBase<dim,nstate>
{
public:
    /// Constructor.
    ChannelFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~ChannelFlow() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to set the higher order grid
    void set_higher_order_grid(std::shared_ptr <DGBase<dim, double>> dg) const override;

    /// Initialize model variables
    void initialize_model_variables(std::shared_ptr<DGBase<dim, double>> dg) const override;

    /// Update model variables
    void update_model_variables(std::shared_ptr<DGBase<dim, double>> dg) const override;

protected:
    const double channel_height; ///< Channel height
    const double half_channel_height; ///< Half channel height
    const double channel_friction_velocity_reynolds_number; ///< Channel Reynolds number based on wall friction velocity
    const int number_of_cells_x_direction; ///< Number of cells in x-direction
    const int number_of_cells_y_direction; ///< Number of cells in y-direction
    const int number_of_cells_z_direction; ///< Number of cells in z-direction

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

private:
    /// Get the integrated density over the domain
    double get_integrated_density_over_domain(DGBase<dim, double> &dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
