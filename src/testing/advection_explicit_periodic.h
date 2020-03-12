#ifndef __ADVECTION_EXPLICIT_PERIODIC_H__
#define __ADVECTION_EXPLICIT_PERIODIC_H__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

#include <deal.II/grid/manifold_lib.h>

namespace PHiLiP {
namespace Tests {

/// Curvilinear manifold.
//class CurvManifold: public dealii::ChartManifold<2,2,2> {
template <int dim>
class CurvManifold: public dealii::ChartManifold<dim,dim,dim> {
#if 0
public:
    virtual dealii::Point<2> pull_back(const dealii::Point<2> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<2> push_forward(const dealii::Point<2> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,2,2> push_forward_gradient(const dealii::Point<2> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<2,2> > clone() const override; ///< See dealii::Manifold.
#endif
    virtual dealii::Point<dim> pull_back(const dealii::Point<dim> &space_point) const override; ///< See dealii::Manifold.
    virtual dealii::Point<dim> push_forward(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    virtual dealii::DerivativeForm<1,dim,dim> push_forward_gradient(const dealii::Point<dim> &chart_point) const override; ///< See dealii::Manifold.
    
    virtual std::unique_ptr<dealii::Manifold<dim,dim> > clone() const override; ///< See dealii::Manifold.
};


template <int dim, int nstate>
class AdvectionPeriodic: public TestsBase
{
public:
	AdvectionPeriodic() = delete;
	AdvectionPeriodic(const Parameters::AllParameters *const parameters_input);
    int run_test () const override;
private:
    //const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    const MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;
    double compute_energy(std::shared_ptr < PHiLiP::DGBase<dim, double> > &dg) const;
    static dealii::Point<dim> warp (const dealii::Point<dim> &p);
};

} // End of Tests namespace
} // End of PHiLiP namespace
#endif
