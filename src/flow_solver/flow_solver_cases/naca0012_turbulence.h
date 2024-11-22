#ifndef __NACA0012_LES__
#define __NACA0012_LES__

// for FlowSolver class:
#include "physics/initial_conditions/initial_condition_function.h"
#include "dg/dg.h"
#include "physics/physics.h"
#include "parameters/all_parameters.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nstate>
class NACA0012_LES : public FlowSolverCaseBase<dim,nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
    static const int NUMBER_OF_INTEGRATED_QUANTITIES = 5;
public:
    /// Constructor.
    NACA0012_LES(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~NACA0012_LES() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to set the higher order grid
    void set_higher_order_grid(std::shared_ptr <DGBase<dim, double>> dg) const override;

    /// Will compute and print lift and drag coefficients
    void steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const override;


    /// Output the velocity field to file
    void output_velocity_field(
            std::shared_ptr<DGBase<dim,double>> dg,
            const unsigned int output_file_index,
            const double current_time) const;

    /// Output the velocity field to file
    void output_kinetic_energy_at_points(
            std::shared_ptr<DGBase<dim,double>> dg,
            const double current_time,
            const dealii::Point<dim,double>,
            const dealii::Point<dim,double>,
            const dealii::Point<dim,double>,
            const std::shared_ptr <dealii::TableHandler> unsteady_data_table) const;

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

    /// Flag for outputting density field in addition to velocity field at fixed times
    const bool output_density_field_in_addition_to_velocity;
    /// Flag for outputting viscosity field in addition to velocity field at fixed times
    const bool output_viscosity_field_in_addition_to_velocity;

    /// Flag for outputting density field in addition to velocity field at fixed times
    const bool compute_time_averaged_solution;
    /// Flag for outputting viscosity field in addition to velocity field at fixed times
    const double time_to_start_averaging;
    
    /// Pointer to Navier-Stokes physics object for computing things on the fly
    std::shared_ptr< Physics::NavierStokes<dim,dim+2,double> > navier_stokes_physics;

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

public:
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table,
            const bool do_write_unsteady_data_table_file) override;

protected:
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

    /// Integrated kinetic energy over the domain at previous time step; used for ensuring a physically consistent simulation
    double integrated_kinetic_energy_at_previous_time_step;

    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Times at which to output the velocity field
    dealii::Table<1,double> output_velocity_field_times;

    /// Index of current desired time to output velocity field
    unsigned int index_of_current_desired_time_to_output_velocity_field;

    /// Index of current desired time to output velocity field
    unsigned int output_counter = -1;

    /// Index of current desired time to output to terminal
    unsigned int terminal_counter = -1;

    /// Flow field quantity filename prefix
    std::string flow_field_quantity_filename_prefix;

    /// Data table storing the exact output times for the velocity field files
    std::shared_ptr<dealii::TableHandler> exact_output_times_of_velocity_field_files_table;

private:
    /// Compute lift
    double compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const;

    /// Compute drag
    double compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
