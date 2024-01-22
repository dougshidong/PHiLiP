#ifndef __PERIODIC_TURBULENCE_H__
#define __PERIODIC_TURBULENCE_H__

#include <deal.II/base/table.h>

#include "dg/dg_base.hpp"
#include "periodic_cube_flow.h"
#include "physics/navier_stokes.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class PeriodicTurbulence : public PeriodicCubeFlow<dim,nstate>
{
    /** Number of different computed quantities
     *  Corresponds to the number of items in IntegratedQuantitiesEnum
     * */
    static const int NUMBER_OF_INTEGRATED_QUANTITIES = 5;

public:
    /// Constructor.
    explicit PeriodicTurbulence(const Parameters::AllParameters *const parameters_input);

    /** Computes the integrated quantities over the domain simultaneously and updates the array storing them
     *  Note: For efficiency, this also simultaneously updates the local maximum wave speed
     * */
    void compute_and_update_integrated_quantities(DGBase<dim, double> &dg);

    /** Gets the nondimensional integrated kinetic energy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_kinetic_energy() const;

    /** Gets the nondimensional integrated enstrophy given a DG object from dg->solution
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_integrated_enstrophy() const;

    /** Gets non-dimensional theoretical vorticity tensor based dissipation rate 
     *  Note: For incompressible flows or when dilatation effects are negligible 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_vorticity_based_dissipation_rate() const;

    /** Evaluate non-dimensional theoretical pressure-dilatation dissipation rate
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_pressure_dilatation_based_dissipation_rate () const;

    /** Gets non-dimensional theoretical deviatoric strain-rate tensor based dissipation rate 
     *  -- Reference: Cox, Christopher, et al. "Accuracy, stability, and performance comparison 
     *                between the spectral difference and flux reconstruction schemes." 
     *                Computers & Fluids 221 (2021): 104922.
     * */
    double get_deviatoric_strain_rate_tensor_based_dissipation_rate() const;

    /** Gets non-dimensional theoretical strain-rate tensor based dissipation rate from integrated
     *  strain-rate tensor magnitude squared.
     *  -- Reference: Navah, Farshad, et al. "A High-Order Variational Multiscale Approach 
     *                to Turbulence for Compact Nodal Schemes." 
     * */
    double get_strain_rate_tensor_based_dissipation_rate() const;

    /// Output the velocity field to file
    void output_velocity_field(
            std::shared_ptr<DGBase<dim,double>> dg,
            const unsigned int output_file_index,
            const double current_time) const;

    /// Calculate numerical entropy by matrix-vector product
    double get_numerical_entropy(const std::shared_ptr <DGBase<dim, double>> dg) const;

protected:
    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    /// Number of times to output the velocity field
    const unsigned int number_of_times_to_output_velocity_field;

    /// Flag for outputting velocity field at fixed times
    const bool output_velocity_field_at_fixed_times;

    /// Flag for outputting vorticity magnitude field in addition to velocity field at fixed times
    const bool output_vorticity_magnitude_field_in_addition_to_velocity;

    /// Directory for writting flow field files
    const std::string output_flow_field_files_directory_name;

    const bool output_solution_at_exact_fixed_times;///< Flag for outputting the solution at exact fixed times by decreasing the time step on the fly

    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes<dim,dim+2,double> > navier_stokes_physics;

    bool is_taylor_green_vortex = false; ///< Identifies if taylor green vortex case; initialized as false.
    bool is_decaying_homogeneous_isotropic_turbulence = false; ///< Identified if DHIT case; initialized as false.
    bool is_viscous_flow = true; ///< Identifies if viscous flow; initialized as true.
    bool do_calculate_numerical_entropy = false; ///< Identifies if numerical entropy should be calculated; initialized as false.

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

    /// Function to compute the constant time step
    double get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

/*    /// Function to compute the adaptive time step
    double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;

    /// Updates the maximum local wave speed
    void update_maximum_local_wave_speed(DGBase<dim, double> &dg);*/

    /// Function to compute the adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step;

    /// Function to compute the initial adaptive time step
    using CubeFlow_UniformGrid<dim, nstate>::get_adaptive_time_step_initial;

    /// Updates the maximum local wave speed
    using CubeFlow_UniformGrid<dim, nstate>::update_maximum_local_wave_speed;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

    /// List of possible integrated quantities over the domain
    enum IntegratedQuantitiesEnum {
        kinetic_energy,
        enstrophy,
        pressure_dilatation,
        deviatoric_strain_rate_tensor_magnitude_sqr,
        strain_rate_tensor_magnitude_sqr
    };
    /// Array for storing the integrated quantities; done for computational efficiency
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrated_quantities;

    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Times at which to output the velocity field
    dealii::Table<1,double> output_velocity_field_times;

    /// Index of current desired time to output velocity field
    unsigned int index_of_current_desired_time_to_output_velocity_field;

    /// Flow field quantity filename prefix
    std::string flow_field_quantity_filename_prefix;

    /// Data table storing the exact output times for the velocity field files
    std::shared_ptr<dealii::TableHandler> exact_output_times_of_velocity_field_files_table;
};

} // FlowSolver namespace
} // PHiLiP namespace
#endif
