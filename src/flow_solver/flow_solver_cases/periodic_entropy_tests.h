#ifndef __PERIODIC_ENTROPY_TESTS_H__
#define __PERIODIC_ENTROPY_TESTS_H__

#include "periodic_cube_flow.h"
#include "physics/euler.h"

namespace PHiLiP {
namespace FlowSolver {

template <int dim, int nspecies, int nstate>
class PeriodicEntropyTests : public PeriodicCubeFlow<dim, nspecies, nstate>
{
public:

    /// Constructor.
    explicit PeriodicEntropyTests(const Parameters::AllParameters *const parameters_input);

    /// Calculate numerical entropy
    /// Calls compute_integrated_quantities
    double compute_entropy(const std::shared_ptr <DGBase<dim, nspecies, double>> dg) const;

    /// Function to compute the constant time step
    /** Calculates based on CFL for Euler, and from parameters otherwise */
    double get_constant_time_step(std::shared_ptr<DGBase<dim,nspecies,double>> dg) const override;
protected:

    /// Enum of integrated quantities to calculate
    enum IntegratedQuantityEnum { kinetic_energy, max_wave_speed, numerical_entropy};

    /// Compute and update integrated quantities
    /** Same function as in periodic_turbulence. Has some computational inefficiency
     * due to inclusion of solution gradient, but leaving to make it easier to add
     * other integrated quantities if needed in the future.
     * This only returns one quantity, specified by the second argument
     * Will need to be modified in the future if multiple quantites are needed
     * See structure in periodic_turbulence
     */
    double compute_integrated_quantities(DGBase<dim, nspecies, double> &dg,
            IntegratedQuantityEnum quantity, 
            const int overintegrate=10 // Overintegrate for KE, don't for num. entropy
            ) const;

    using FlowSolverCaseBase<dim, nspecies, nstate>::compute_unsteady_data_and_write_to_table;
    /// Compute the desired unsteady data and write it to a table
    void compute_unsteady_data_and_write_to_table(
            const std::shared_ptr<ODE::ODESolverBase<dim, nspecies, double>> ode_solver, 
            const std::shared_ptr <DGBase<dim, nspecies, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table) override;
    
    /// Filename for unsteady data
    std::string unsteady_data_table_filename_with_extension;
    
    /// Storing entropy at first step
    double initial_entropy;

    /// Store previous entropy
    double previous_numerical_entropy;
    
    /// Last time (for calculating relaxation factor)
    double previous_time=0;

    // euler physics pointer for computing physical quantities.
    std::shared_ptr < Physics::Euler<dim, nstate, double > > euler_physics;

};


} // FlowSolver namespace
} // PHiLiP namespace
#endif
