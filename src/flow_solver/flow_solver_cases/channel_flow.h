#ifndef __CHANNEL_FLOW_H__
#define __CHANNEL_FLOW_H__

#include "periodic_turbulence.h"
#include "dg/dg.h"

namespace PHiLiP {
namespace FlowSolver {

#if PHILIP_DIM==1
using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif

template <int dim, int nstate>
class ChannelFlow : public PeriodicTurbulence<dim,nstate>
{
public:
    /// Constructor.
    ChannelFlow(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~ChannelFlow() {};

    /** Computes the integrated quantities over the domain simultaneously and updates the array storing them
     *  Note: For efficiency, this also simultaneously updates the local maximum wave speed
     * */
    void compute_and_update_integrated_quantities(DGBase<dim, double> &dg) override;

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
    const double pi_val; ///< Value of pi
    const double domain_length_x; ///< Domain length in x-direction
    const double domain_length_y; ///< Domain length in y-direction
    const double domain_length_z; ///< Domain length in z-direction
    double minimum_approximate_grid_spacing; ///< Minimum approximate grid spacing

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const override;

    /// Function to compute the adaptive time step
    double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

private:
    /// Get the stretched mesh size
    double get_stretched_mesh_size(const int i) const;

    /// Get the integrated density over the domain
    double get_integrated_density_over_domain(DGBase<dim, double> &dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
