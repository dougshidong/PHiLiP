#ifndef __LIMITER_CONVERGENCE_TESTS__
#define __LIMITER_CONVERGENCE_TESTS__

#include "flow_solver_case_base.h"

namespace PHiLiP {
namespace FlowSolver{

//===============================================================
/// Limiter Convergence Tests (Advection, Burgers, 2D Low Density)
//===============================================================
template <int dim, int nstate>
class LimiterConvergenceTests : public FlowSolverCaseBase<dim, nstate>
{
#if PHILIP_DIM==1
    using Triangulation = dealii::Triangulation<PHILIP_DIM>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<PHILIP_DIM>;
#endif
public:
    /// Constructor
    explicit LimiterConvergenceTests(const Parameters::AllParameters *const parameters_input);
    
    /// Destructor
    ~LimiterConvergenceTests() = default;

    /// Function to generate the grid
    std::shared_ptr<Triangulation> generate_grid() const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const override;

    /// Function to compute the initial adaptive time step
    double get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) override;    

    /// Updates the maximum local wave speed
    void update_maximum_local_wave_speed(DGBase<dim, double> &dg);

    /// Updates the maximum local wave speed
    void check_limiter_principle(DGBase<dim, double>& dg);

    /// Filename (with extension) for the unsteady data table
    const std::string unsteady_data_table_filename_with_extension;

    using FlowSolverCaseBase<dim,nstate>::compute_unsteady_data_and_write_to_table;

    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;

protected:
    /// Display additional more specific flow case parameters
    void display_additional_flow_case_specific_parameters() const override;

     protected:
    /// Maximum local wave speed (i.e. convective eigenvalue)
    double maximum_local_wave_speed;

    /// Pointer to Physics object for computing things on the fly
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,double> > pde_physics;

};

} // FlowSolver namespace
} // PHiLiP namespace

#endif
