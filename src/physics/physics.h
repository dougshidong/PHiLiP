#ifndef __PHYSICS__
#define __PHYSICS__

#include <deal.II/base/tensor.h>

#include "parameters/all_parameters.h"
#include "physics/manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Base class from which Advection, Diffusion, ConvectionDiffusion, and Euler is derived.
/**
 *  Main interface for all the convective and diffusive terms.
 *
 *  LinearAdvection, Diffusion, ConvectionDiffusion, Euler are derived from this class.
 *
 *  Partial differential equation is given by the divergence of the convective and
 *  diffusive flux equal to the source term
 *
 *  \f[ \boldsymbol{\nabla} \cdot
 *         (  \mathbf{F}_{conv}( u ) 
 *          + \mathbf{F}_{diss}( u, \boldsymbol{\nabla}(u) )
 *      = s(\mathbf{x})
 *  \f]
 */
template <int dim, int nstate, typename real>
class PhysicsBase
{
public:
    /// Default constructor that will set the constants.
    PhysicsBase();

    /// Virtual destructor required for abstract classes.
    virtual ~PhysicsBase() = 0;

    /// Manufactured solution function
    const ManufacturedSolutionFunction<dim,real> manufactured_solution_function;

    /// Convective fluxes that will be differentiated once in space.
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &solution) const = 0;

    /// Spectral radius of convective term Jacobian.
    /// Used for scalar dissipation
    virtual std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const = 0;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    virtual real max_convective_eigenvalue (const std::array<real,nstate> &soln) const = 0;

    // /// Evaluate the diffusion matrix \f$ A \f$ such that \f$F_v = A \nabla u\f$.
    // virtual std::array<dealii::Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
    //     const std::array<real,nstate> &solution,
    //     const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_grad) const = 0;

    /// Dissipative fluxes that will be differentiated once in space.
    /// Evaluates the dissipative flux through the linearization F = A(u)*grad(u).
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux_A_gradu (
        const real scaling,
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<dealii::Tensor<1,dim,real>,nstate> &diss_flux) const;

    /// Dissipative fluxes that will be differentiated once in space.
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;

    /// Source term that does not require differentiation.
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,double> &pos,
        const std::array<real,nstate> &solution) const = 0;

    /// Evaluates boundary values and gradients on the other side of the face.
    virtual void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, double> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
protected:
    /// Not yet implemented
    virtual void set_manufactured_dirichlet_boundary_condition (
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
    /// Not yet implemented
    virtual void set_manufactured_neumann_boundary_condition (
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;


    /// Some constants used to define manufactured solution
    double velo_x, velo_y, velo_z;
    double diff_coeff;

    /// Anisotropic diffusion matrix
    /** As long as the diagonal components are positive and diagonally dominant
     *  we should have a stable diffusive system
     */
    dealii::Tensor<2,dim,double> diffusion_tensor;
};



} // Physics namespace
} // PHiLiP namespace

#endif
