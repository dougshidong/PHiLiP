#ifndef __CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_H__
#define __CONVECTION_DIFFUSION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
/// Convection Diffusion periodic unsteady test (currently only diffusion)
class ConvectionDiffusionPeriodic: public TestsBase
{
public:
        /// delete
        ConvectionDiffusionPeriodic() = delete;
        /// Constructor
	ConvectionDiffusionPeriodic(const Parameters::AllParameters *const parameters_input);
        /// Run the testcase
        int run_test () const override;
private:
    /// MPI communicator
        const MPI_Comm mpi_communicator;
    /// print for first rank
        dealii::ConditionalOStream pcout;
    /// Function computes the energy
	double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    /// Function computes the conservation
        double compute_conservation(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg, const double poly_degree) const;
protected:
    ///Setup initial condition
        void initialize(DGBase<dim,double> &dg, const PHiLiP::Parameters::AllParameters &all_parameters_new) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
