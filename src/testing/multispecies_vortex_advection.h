#ifndef __MULTISPECIES_VORTEX_ADVECTION__
#define __MULTISPECIES_VORTEX_ADVECTION__

#include "tests.h"
#include "dg/dg_base.hpp"
#include "parameters/all_parameters.h"
#include "flow_solver/flow_solver.h"
#include "physics/real_gas.h"

namespace PHiLiP {
namespace Tests {
/// Class used to run tests that verify implementation of multispecies
/************************************************************
* Cases include: 1D Multispecies Vortex Advection - Low Temp and High Temp
*************************************************************/
template <int dim, int nspecies, int nstate>
class MultispeciesVortexAdvection : public TestsBase
{
public:
    /// Constructor.
    explicit MultispeciesVortexAdvection(const Parameters::AllParameters* const parameters_input,
        const dealii::ParameterHandler& parameter_handler_input);

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler& parameter_handler;

    /// Runs an order of accuracy convergence test
    /// Passes if the expected order is reached.
    /// Expected order can be set in the prm file
    int run_test() const override;
    
    /// Flag to determine which exact solution is used
    bool high_temp;
private:
    /// Calculate and return the L2 Error
    std::array<std::array<double,3>,nstate+1> calculate_l_n_error(
        std::shared_ptr<DGBase<dim, nspecies, double>> flow_solver_dg, 
        const int poly_degree, 
        const double final_time,
        std::shared_ptr<FlowSolver::FlowSolver<dim, nspecies, nstate>> flow_solver) const;
        
    /// Function to compute the initial adaptive time step
    double get_time_step(std::shared_ptr<DGBase<dim, nspecies, double>> dg) const; 

    /// Real Gas physics pointer for computing physical quantities.
    std::shared_ptr < Physics::RealGas<dim, nspecies, nstate, double > > real_gas_physics;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
