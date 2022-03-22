#ifndef __CONVECTION_DIFFUSION__
#define __CONVECTION_DIFFUSION__

#include <deal.II/base/tensor.h>

#include "parameters/all_parameters.h"
#include "physics.h"
#include "parameters/parameters_manufactured_solution.h"

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
 */
template <int dim, int nstate, typename real>
class ConvectionDiffusion : public PhysicsBase <dim, nstate, real>
{
protected:
    /// Linear advection velocity in x, y, and z directions.
    double linear_advection_velocity[3] = { 1.1, -atan(1)*4.0 / exp(1), exp(1)/(atan(1)*4.0) };
    /// Diffusion scaling coefficient in front of the diffusion tensor.
    double diffusion_scaling_coeff = 0.1*atan(1)*4.0/exp(1);
public:
    const bool hasConvection; ///< Turns ON/OFF convection term.

    const bool hasDiffusion; ///< Turns ON/OFF diffusion term.

    /// Constructor
    ConvectionDiffusion (
        const bool                                                convection = true, 
        const bool                                                diffusion = true, 
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        const dealii::Tensor<1,3,double>                          input_advection_vector = Parameters::ManufacturedSolutionParam::get_default_advection_vector(),
        const double                                              input_diffusion_coefficient = Parameters::ManufacturedSolutionParam::get_default_diffusion_coefficient(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr) : 
            PhysicsBase<dim,nstate,real>(input_diffusion_tensor, manufactured_solution_function), 
            linear_advection_velocity{input_advection_vector[0], input_advection_vector[1], input_advection_vector[2]},
            diffusion_scaling_coeff(input_diffusion_coefficient),
            hasConvection(convection), 
            hasDiffusion(diffusion)
    {
        static_assert(nstate<=5, "Physics::ConvectionDiffusion() should be created with nstate<=5");
    };

    /// Destructor
    ~ConvectionDiffusion () {};
    /// Convective flux: \f$ \mathbf{F}_{conv} =  u \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (const std::array<real,nstate> &solution) const;

    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &soln1,
        const std::array<real,nstate> &soln2) const;

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
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// (function overload) Dissipative flux: u
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const dealii::types::global_dof_index cell_index) const;

    /// (function overload) Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution) const;

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
    /// Linear advection speed:  c
    dealii::Tensor<1,dim,real> advection_speed () const;
    /// Diffusion coefficient
    real diffusion_coefficient () const;
};


} // Physics namespace
} // PHiLiP namespace

#endif
