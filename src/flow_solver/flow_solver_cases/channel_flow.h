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

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to set the higher order grid
    void set_higher_order_grid(std::shared_ptr <DGBase<dim, double>> dg) const override;

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
    const double domain_volume; ///< Domain volume

    /** 
     * Bulk velocity Reynolds number computed from friction velocity based Reynolds numbers (Empirical relation)
     * Reference:
     *  - R. B. Dean, "Reynolds Number Dependence of Skin Friction and Other Bulk
     *    Flow Variables in Two-Dimensional Rectangular Duct Flow", 
     *    Journal of Fluids Engineering, 1978 
     * */
    const double channel_bulk_velocity_reynolds_number;

    /** 
     * Centerline velocity Reynolds number computed from friction velocity based Reynolds numbers (Empirical relation)
     * Reference:
     *  - R. B. Dean, "Reynolds Number Dependence of Skin Friction and Other Bulk
     *    Flow Variables in Two-Dimensional Rectangular Duct Flow", 
     *    Journal of Fluids Engineering, 1978 
     * */
    const double channel_centerline_velocity_reynolds_number;

    double minimum_approximate_grid_spacing; ///< Minimum approximate grid spacing

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Display grid parameters
    void display_grid_parameters() const override;

    /// Return a vector of mesh step sizes in the y-direction based on the desired stretching function
    std::vector<double> get_mesh_step_size_y_direction() const;

    /** Return a vector of mesh step sizes in the y-direction based on the High-Order Prediction Workshop (HOPW) case
     *  Reference: This stretching function comes from the structured GMSH .geo file obtained from https://how5.cenaero.be/content/ws2-les-plane-channel-ret550
     **/
    std::vector<double> get_mesh_step_size_y_direction_HOPW() const;

    /** Return a vector of mesh step sizes in the y-direction based on Gullbrand's stretching function
     *  Reference: Gullbrand, "Grid-independent large-eddy simulation in turbulent channel flow using three-dimensional explicit filtering", 2003.
     **/
    std::vector<double> get_mesh_step_size_y_direction_Gullbrand() const;

    /** Return a vector of mesh step sizes in the y-direction based on C. CARTON DE WIARTET. AL's stretching function
     *  Reference: C. CARTON DE WIARTET. AL, "Implicit LES of free and wall-bounded turbulent flows based onthe discontinuous Galerkin/symmetric interior penalty method", 2015.
     **/
    std::vector<double> get_mesh_step_size_y_direction_carton_de_wiart_et_al() const;

    dealii::Tensor<2,dim,double> zero_tensor; ///< Tensor of zeros

    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes_ChannelFlowConstantSourceTerm_WallModel<dim,dim+2,double> > navier_stokes_channel_flow_constant_source_term_wall_model_physics;

public:
    /// Function to compute the adaptive time step
    double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table,
            const bool do_write_unsteady_data_table_file) override;

    /// Get the number of degrees of freedom per state from a given poly degree
    unsigned int get_number_of_degrees_of_freedom_per_state_from_poly_degree(const unsigned int poly_degree_input) const override;

    /// Get the average wall shear stress
    double get_average_wall_shear_stress(DGBase<dim, double> &dg) const;

    /// Get the average wall shear stress from wall model
    double get_average_wall_shear_stress_from_wall_model(DGBase<dim, double> &dg) const;

    double get_bulk_density() const; ///< Getter for the bulk density
    double get_bulk_velocity() const; ///< Getter for the bulk velocity
    double get_bulk_mass_flow_rate() const; ///< Getter for the bulk mass flow rate

    /// Get the skin friction coefficient from the average wall shear stress
    double get_skin_friction_coefficient_from_average_wall_shear_stress(const double avg_wall_shear_stress) const;

    /// Set the bulk flow quantities
    void set_bulk_flow_quantities(DGBase<dim, double> &dg);
private:
    /// Get the stretched mesh size
    double get_stretched_mesh_size(const int i) const;

    double bulk_density; ///< Bulk density
    double bulk_mass_flow_rate; ///< Bulk mass flow rate
    double bulk_velocity; ///< Bulk velocity
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
