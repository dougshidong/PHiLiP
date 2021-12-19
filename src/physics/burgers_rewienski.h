#ifndef __BURGERS_REWIENSKI__
#define __BURGERS_REWIENSKI__

#include "burgers.h"

namespace PHiLiP {
namespace Physics {
/// Burgers Rewienski equation. Derived from Burgers, which is derived from PhysicsBase. Based on eq.(18) in Carlberg 2011
template <int dim, int nstate, typename real>
class BurgersRewienski : public Burgers <dim, nstate, real>
{
public:
    /// Constructor
    BurgersRewienski(
            const double                                              rewienski_a,
            const double                                              rewienski_b,
            const bool                                                rewienski_manufactured_solution,
            const bool                                                convection,
            const bool                                                diffusion,
            const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
            std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr);

    /// Destructor
    ~BurgersRewienski () {};

    /// Parameter a for eq.(18) in Carlberg 2011
    const double rewienski_a;

    /// Parameter b for eq.(18) in Carlberg 2011
    const double rewienski_b;

    /** Run manufactured solution for this case if set to "true"
      *  * Additional parameter since parameter "use_manufactured_solution_source" is already set to "true" for the PDE's source term
      */
    const bool rewienski_manufactured_solution;

    /// PDE Source term. If rewienski_manufactured_solution==true then the manufactured solution source term is also included.
    std::array<real,nstate> source_term (
            const dealii::Point<dim,real> &pos,
            const std::array<real,nstate> &solution,
            const real current_time) const override;

    /// If diffusion is present, assign Dirichlet boundary condition
    void boundary_face_values (
            const int /*boundary_type*/,
            const dealii::Point<dim, real> &/*pos*/,
            const dealii::Tensor<1,dim,real> &/*normal*/,
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const override;

};
} // Physics namespace
} // PHiLiP namespace

#endif
