#ifndef __ADVECTION_EXPLICIT_PERIODIC_H__
#define __ADVECTION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

#include <deal.II/grid/manifold_lib.h>

namespace PHiLiP {
namespace Tests {

template <int dim>
/// Curvilinear manifold.
class CurvManifold: public dealii::ChartManifold<dim,dim,dim> {
    /// Pull Back function
    virtual dealii::Point<dim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    /// Push forward function
    virtual dealii::Point<dim> push_forward(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    /// Derivative of mapping
    virtual dealii::DerivativeForm<1,dim,dim> push_forward_gradient(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    /// Clone
    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.
};


template <int dim, int nstate>
/// Advection periodic unsteady test
class AdvectionPeriodic: public TestsBase
{
public:
        /// delete
	AdvectionPeriodic() = delete;
        /// Constructor
	AdvectionPeriodic(const Parameters::AllParameters *const parameters_input);
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
    /// Warping for nonlinear manifold (see CurvManifold above)
    static dealii::Point<dim> warp (const dealii::Point<dim> &p);
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
