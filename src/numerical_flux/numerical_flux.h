#ifndef __NUMERICAL_FLUX__
#define __NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
namespace PHiLiP
{
    using namespace dealii;
    using AllParam = Parameters::AllParameters;

    /// Numerical flux associated with convection
    template<int dim, int nstate, typename real>
    class NumericalFluxConvective
    {
    public:
    virtual ~NumericalFluxConvective() = 0;

    virtual std::array<real, nstate> evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const Tensor<1,dim,real> &normal1) const = 0;

    };


    template<int dim, int nstate, typename real>
    class LaxFriedrichs: public NumericalFluxConvective<dim, nstate, real>
    {
    public:

    /// Constructor
    LaxFriedrichs(Physics<dim, nstate, real> *physics_input)
    :
    pde_physics(physics_input)
    {};
    /// Destructor
    ~LaxFriedrichs() {};

    std::array<real, nstate> evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const Tensor<1,dim,real> &normal1) const;

    protected:
    const Physics<dim, nstate, real> *pde_physics;

    };

    /// Numerical flux associated with dissipation
    template<int dim, int nstate, typename real>
    class NumericalFluxDissipative
    {
    public:
    /// Base class destructor.
    /// Abstract classes required virtual destructors.
    /// Even if it is a pure function, its definition is still required.
    virtual ~NumericalFluxDissipative() = 0;

    virtual void evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_int,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_ext,
        const Tensor<1,dim,real> &normal1,
        std::array<real, nstate> &soln_flux,
        std::array<Tensor<1,dim,real>, nstate> &grad_flux) const = 0;

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
    void evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_int,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_ext,
        const Tensor<1,dim,real> &normal1,
        std::array<real, nstate> &soln_flux,
        std::array<Tensor<1,dim,real>, nstate> &grad_flux) const;
    protected:
    const Physics<dim, nstate, real> *pde_physics;

    };


    template <int dim, int nstate, typename real>
    class NumericalFluxFactory
    {
    public:
        static NumericalFluxConvective<dim,nstate,real>*
            create_convective_numerical_flux
                (AllParam::ConvectiveNumericalFlux conv_num_flux_type,
                Physics<dim, nstate, real> *physics_input);
        static NumericalFluxDissipative<dim,nstate,real>*
            create_dissipative_numerical_flux
                (AllParam::DissipativeNumericalFlux diss_num_flux_type,
                Physics<dim, nstate, real> *physics_input);
    };


}

#endif
