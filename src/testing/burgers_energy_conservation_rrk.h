#ifndef __BURGERS_ENERGY_CONSERVATION_RRK__
#define __BURGERS_ENERGY_CONSERVATION_RRK__

#include "tests.h"
#include <deal.II/base/convergence_table.h>
#include "dg/dg.h"

namespace PHiLiP {
namespace Tests {

/// Time refinement study which compares to a reference solution
template <int dim, int nstate>
class BurgersEnergyConservationRRK: public TestsBase
{
public:
    /// Constructor
    BurgersEnergyConservationRRK(
            const Parameters::AllParameters *const parameters_input,
            const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~BurgersEnergyConservationRRK() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Run test
    int run_test () const override;
protected:

    /// Reinitialize parameters. Necessary because all_parameters is constant.
    Parameters::AllParameters reinit_params(bool use_rrk, double time_step_size) const;

    int compare_energy_to_initial(
            const std::shared_ptr <DGBase<dim, double>> dg,
            const double initial_energy,
            bool expect_conservation
            ) const;
    /// Compute energy from dg. Only valid for collocated nodes.
    double compute_energy_collocated(const std::shared_ptr <DGBase<dim, double>> dg) const;

    /// runs flow solver. Returns 0 (pass) or 1 (fail) based on energy conservation of calculation.
    int run_flow_solver(
            const Parameters::AllParameters params,
            const double energy_initial,
            bool expect_conservation
            ) const;

};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
