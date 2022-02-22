#ifndef __BURGERS__
#define __BURGERS__

#include <deal.II/base/tensor.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

#include "physics.h"

namespace PHiLiP {
namespace Physics {
/// Burger's equation with nonlinear advective term and linear diffusive term.  Derived from PhysicsBase.
/** State variable: \f$ u \f$
 *  
 *  Convective flux \f$ \mathbf{F}_{conv} =  0.5u^2 \f$
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
 */
template <int dim, int nstate, typename real>
class Burgers : public PhysicsBase <dim, nstate, real>
{
protected:
    /// Diffusion scaling coefficient in front of the diffusion tensor.
    double diffusion_scaling_coeff = 0.1*atan(1)*4.0/exp(1);
public:
    /// Turns on convective part of the Burgers problem.
    /** Without the nonlinear convection, it's simply diffusion */
    const bool hasConvection;
    /// Turns on diffusive part of the Burgers problem.
    const bool hasDiffusion;
    ///Allows Burgers to distinguish between different unsteady test types.
    const Parameters::AllParameters::TestType test_type; ///< Pointer to all parameters

    /// Constructor
    Burgers(
        const bool                                                convection = true, 
        const bool                                                diffusion = true, 
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
       // const Parameters::AllParameters::TestType parameters_test = Parameters::AllParameters::TestType::burgers_energy_stability) : 
        const Parameters::AllParameters::TestType parameters_test = Parameters::AllParameters::TestType::run_control) : 
        //const Parameters::AllParameters::TestType parameters_test) : 
            PhysicsBase<dim,nstate,real>(input_diffusion_tensor, manufactured_solution_function), 
            hasConvection(convection), 
            hasDiffusion(diffusion),
            test_type(parameters_test)
    {
        static_assert(nstate==dim, "Physics::Burgers() should be created with nstate==dim");
    };

    /// Destructor
    ~Burgers () {};
    /// Convective flux: \f$ \mathbf{F}_{conv} =  u \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (const std::array<real,nstate> &solution) const;

    /// Convective split flux
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
                const std::array<real,nstate> &soln_const,
                const std::array<real,nstate> & soln_loop) const;

    /// Convective surface split flux
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_surface_numerical_split_flux (
                const std::array< dealii::Tensor<1,dim,real>, nstate > &surface_flux,
                const std::array< dealii::Tensor<1,dim,real>, nstate > &flux_interp_to_surface) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    //  /// Diffusion matrix is identity
    //  std::array<dealii::Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
    //      const std::array<real,nstate> &solution,
    //      const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_grad) const;

    /// Dissipative flux: u
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const real current_time) const;

    /// If diffusion is present, assign Dirichlet boundary condition
    /** Using Neumann boundary conditions might need to modify the functional
     *  in order to obtain the optimal 2p convergence of the functional error
     */
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

protected:
    /// Diffusion coefficient
    real diffusion_coefficient () const;
};


} // Physics namespace
} // PHiLiP namespace

#endif
