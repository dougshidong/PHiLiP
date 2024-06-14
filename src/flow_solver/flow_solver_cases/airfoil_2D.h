#ifndef __AIRFOIL_2D__
#define __AIRFOIL_2D__

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
class Airfoil2D : public FlowSolverCaseBase<dim,nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
public:
    /// Constructor.
    Airfoil2D(const Parameters::AllParameters *const parameters_input);

    /// Destructor
    ~Airfoil2D() {};

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to postprocess flow solutions, i.e. extract boundary layer parameters, generate farfield acoustic noise...
    void steady_state_postprocessing(std::shared_ptr <DGBase<dim, double>> dg) const override;

protected:
    const double airfoil_length;

    const double height;

    const double length_b2;

    const double incline_factor;

    const double bias_factor;

    const unsigned int refinements;

    const unsigned int n_subdivision_x_0;

    const unsigned int n_subdivision_x_1;

    const unsigned int n_subdivision_x_2;

    const unsigned int n_subdivision_y;

    const unsigned int airfoil_sampling_factor;

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    void compute_unsteady_data_and_write_to_table(
            const unsigned int current_iteration,
            const double current_time,
            const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;


private:
    /// Compute lift
    double compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const;

    /// Compute drag
    double compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const;
};

} // FlowSolver namespace
} // PHiLiP namespace

#endif