#ifndef __ADVECTION_SPACETIME__
#define ___ADVECTION_SPACETIME__

#include <deal.II/base/tensor.h>

#include "parameters/all_parameters.h"
#include "convection_diffusion.h"

namespace PHiLiP {
namespace Physics {
/// Convection-diffusion with linear advective and diffusive term.  Derived from PhysicsBase.
/** State variable: \f$ u \f$
 *  
 *  Convective flux \f$ \mathbf{F}_{conv} =  u \f$
 *
 *  Dissipative flux \f$ \mathbf{F}_{diss} = -\boldsymbol\nabla u \f$
 *
 *  Source term \f$ s(\mathbf{x}) \f$
 *
 *  Equation:
 *  \f[ \boldsymbol{\nabla} \cdot
 *         (  \mathbf{F}_{conv}( u ) 
 *          + \mathbf{F}_{diss}( u, \boldsymbol{\nabla}(u) )
 *      = s(\mathbf{x})
 *  \f]
 *
 *
 *  The spacetime version only supports advection.
 */
template <int dim, int nstate, typename real>
class AdvectionSpacetime : public ConvectionDiffusion<dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following:
    */
    using PhysicsBase<dim,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nstate,real>::source_term;
public:

    /// Constructor
    AdvectionSpacetime (
        const Parameters::AllParameters *const                    parameters_input,
        const bool                                                convection = true, 
        const bool                                                diffusion = true,
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        const dealii::Tensor<1,3,double>                          input_advection_vector = Parameters::ManufacturedSolutionParam::get_default_advection_vector(),
        const double                                              input_diffusion_coefficient = Parameters::ManufacturedSolutionParam::get_default_diffusion_coefficient(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const Parameters::AllParameters::TestType                 parameters_test = Parameters::AllParameters::TestType::run_control,
        const bool                                                has_nonzero_physical_source = false) : 
            ConvectionDiffusion<dim,nstate,real>(parameters_input, convection, diffusion,input_diffusion_tensor,input_advection_vector,input_diffusion_coefficient,manufactured_solution_function,parameters_test,has_nonzero_physical_source)
    {};


    /// Convective flux 
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (const std::array<real,nstate> &solution) const;

    /// Convective eigenvalues, calculated based on spatial convection only
    /// These are used in calculation of numerical flux; since we use a 
    /// different numerical flux in time, max eigenvalues should only 
    /// come from space.
    std::array<real,nstate> convective_eigenvalues (const std::array<real,nstate> &solution,
            const dealii::Tensor<1,dim,real> &normal) const;

    /// Maximum convective eigenvalue in only the spatial direction
    real max_convective_eigenvalue (const std::array<real,nstate> &solution) const;

protected:
    /// Linear advection speed:  c
    /// Only spatial part.
    dealii::Tensor<1,dim-1,real> advection_speed () const;

};


} // Physics namespace
} // PHiLiP namespace

#endif
