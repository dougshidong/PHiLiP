#ifndef __NUMERICAL_FLUX__
#define __NUMERICAL_FLUX__

#include <deal.II/base/tensor.h>
#include "physics/physics.h"
#include "numerical_flux/viscous_numerical_flux.h"
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
