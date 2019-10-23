#ifndef __CONVECTION_DIFFUSION__
#define __CONVECTION_DIFFUSION__

#include <deal.II/base/tensor.h>

#include "parameters/all_parameters.h"
#include "physics.h"

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
public:
    const bool hasConvection; ///< Turns ON/OFF convection term.

    const bool hasDiffusion; ///< Turns ON/OFF diffusion term.

    /// Constructor
    ConvectionDiffusion (const bool convection = true, const bool diffusion = true)
        : hasConvection(convection), hasDiffusion(diffusion)
    {
        static_assert(nstate<=2, "Physics::ConvectionDiffusion() should be created with nstate<=2");
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
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,double> &pos,
        const std::array<real,nstate> &solution) const;

    /// If diffusion is present, assign Dirichlet boundary condition
    /** Using Neumann boundary conditions might need to modify the functional
     *  in order to obtain the optimal 2p convergence of the functional error
     */
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, double> &/*pos*/,
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
