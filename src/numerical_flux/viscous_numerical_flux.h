#ifndef __VISCOUS_NUMERICAL_FLUX__
#define __VISCOUS_NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
namespace PHiLiP
{
    using namespace dealii;
    using AllParam = Parameters::AllParameters;

    /// Numerical flux associated with dissipation
    template<int dim, int nstate, typename real>
    class NumericalFluxDissipative
    {
    public:
    /// Base class destructor.
    /// Abstract classes required virtual destructors.
    /// Even if it is a pure function, its definition is still required.
    virtual ~NumericalFluxDissipative() = 0;

    virtual std::array<real, nstate> evaluate_solution_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const Tensor<1,dim,real> &normal_int) const = 0;

    virtual std::array<real, nstate> evaluate_auxiliary_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_int,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_ext,
        const Tensor<1,dim,real> &normal_int,
        const real &penalty) const = 0;

    Tensor<1,dim, Tensor<1,nstate,real>> diffusion_matrix_int;
    Tensor<1,dim, Tensor<1,nstate,real>> diffusion_matrix_int_transpose;
    Tensor<1,dim, Tensor<1,nstate,real>> diffusion_matrix_ext;
    Tensor<1,dim, Tensor<1,nstate,real>> diffusion_matrix_ext_transpose;

    };

    template<int dim, int nstate, typename real>
    class SymmetricInternalPenalty: public NumericalFluxDissipative<dim, nstate, real>
    {
    public:
    /// Constructor
    SymmetricInternalPenalty(Physics<dim, nstate, real> *physics_input)
    :
    pde_physics(physics_input)
    {};
    /// Destructor
    ~SymmetricInternalPenalty() {};

    /// Evaluate solution and gradient flux
    /*  $\hat{u} = {u_h}$, 
     *  $ \hat{A} = {{ A \nabla u_h }} - \mu {{ A }} [[ u_h ]] $
     */
    std::array<real, nstate> evaluate_solution_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const Tensor<1,dim,real> &normal_int) const;

    std::array<real, nstate> evaluate_auxiliary_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_int,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_ext,
        const Tensor<1,dim,real> &normal_int,
        const real &penalty) const;
        
    protected:
    const Physics<dim, nstate, real> *pde_physics;

    };

}

#endif
