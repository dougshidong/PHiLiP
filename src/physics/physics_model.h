#ifndef __PHYSICS_MODEL__
#define __PHYSICS_MODEL__

/// Files for the baseline physics
#include "navier_stokes.h"

namespace PHiLiP {
namespace Physics {

/// Physics Model equations. Derived from PhysicsBase, holds a baseline physics and model terms and equations. 
template <int dim, int nstate, typename real>
class PhysicsModel : public PhysicsBase <dim, nstate, real>
{
public:
    /// Constructor
    PhysicsModel(
        Parameters::AllParameters::PartialDifferentialEquation       baseline_physics_type,
        const int                                                    nstate_baseline_physics,
        std::shared_ptr< PHiLiP::PhysicsModelBase<dim,nstate,real> > physics_model_input,
        const dealii::Tensor<2,3,double>                             input_diffusion_tensor,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> >    manufactured_solution_function);

    /// Number of states in for the corresponding baseline physics
    int nstate_baseline_physics;

    /// Number of model equations (i.e. those additional to the baseline physics)
    int n_model_equations;

    /// Physics model object
    std::shared_ptr< PHiLiP::PhysicsModel::PhysicsModelBase<dim,nstate,real> > physics_model;

    /// Baseline physics object
    std::shared_ptr< PhysicsBase<dim,nstate,real> > physics_baseline;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    // TO DO: add all the other member functions that have to be defined as per PhysicsBase
    // use if constexpr(nstate==nstate_baseline_equations) for all the .cpp definitions of these 
};

} // Physics namespace
} // PHiLiP namespace

#endif
