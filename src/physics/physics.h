#ifndef __PHYSICS__
#define __PHYSICS__

#include <deal.II/base/tensor.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/fe/fe_update_flags.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"
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
    PhysicsBase(
        const dealii::Tensor<2,3,double>                          input_diffusion_tensor = Parameters::ManufacturedSolutionParam::get_default_diffusion_tensor(),
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input = nullptr);

    /// Virtual destructor required for abstract classes.
    virtual ~PhysicsBase() = 0;

    /// Manufactured solution function
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function;

    /// Convective fluxes that will be differentiated once in space.
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &solution) const = 0;

    /// Convective Numerical Split Flux for split form
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
            const std::array<real,nstate> &soln_const, const std::array<real,nstate> &soln_loop) const = 0;

/// Convective Numerical Split Flux for split form
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> convective_surface_numerical_split_flux (
                const std::array< dealii::Tensor<1,dim,real>, nstate > &surface_flux,
                const std::array< dealii::Tensor<1,dim,real>, nstate > &flux_interp_to_surface) const = 0;

    /// Spectral radius of convective term Jacobian.
    /** Used for scalar dissipation
     */
    virtual std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const = 0;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    virtual real max_convective_eigenvalue (const std::array<real,nstate> &soln) const = 0;

    // /// Evaluate the diffusion matrix \f$ A \f$ such that \f$F_v = A \nabla u\f$.
    // virtual std::array<dealii::Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
    //     const std::array<real,nstate> &solution,
    //     const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_grad) const = 0;

    /// Dissipative fluxes that will be differentiated ONCE in space.
    /** Evaluates the dissipative flux through the linearization F = A(u)*grad(u).
     */
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux_A_gradu (
        const real scaling,
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<dealii::Tensor<1,dim,real>,nstate> &diss_flux) const;

    /// Dissipative fluxes that will be differentiated ONCE in space.
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const = 0;

    /// Artificial dissipative fluxes that will be differentiated ONCE in space.
    /** Stems from the Persson2006 paper on subcell shock capturing */
/*    virtual std::array<dealii::Tensor<1,dim,real>,nstate> artificial_dissipative_flux (
        const real viscosity_coefficient,
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;
*/ 
    /// Source term that does not require differentiation.
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,//) const;
        const real /*current_time*/) const = 0;

    /// Artificial source term that does not require differentiation stemming from artificial dissipation.
    virtual std::array<real,nstate> artificial_source_term (
        const real viscosity_coefficient,
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution) const;

    /// Evaluates boundary values and gradients on the other side of the face.
    virtual void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

    /// Returns current vector solution to be used by PhysicsPostprocessor to output current solution.
    /** The implementation in this Physics base class simply returns the stored solution.
     */
    virtual dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &/*duh*/,
        const std::vector<dealii::Tensor<2,dim> > &/*dduh*/,
        const dealii::Tensor<1,dim>               &/*normals*/,
        const dealii::Point<dim>                  &/*evaluation_points*/) const;

    /// Returns current scalar solution to be used by PhysicsPostprocessor to output current solution.
    /** The implementation in this Physics base class simply returns the stored solution.
     */
    virtual dealii::Vector<double> post_compute_derived_quantities_scalar (
        const double              &uh,
        const dealii::Tensor<1,dim> &/*duh*/,
        const dealii::Tensor<2,dim> &/*dduh*/,
        const dealii::Tensor<1,dim> &/*normals*/,
        const dealii::Point<dim>    &/*evaluation_points*/) const;
    /// Returns names of the solution to be used by PhysicsPostprocessor to output current solution.
    /** The implementation in this Physics base class simply returns "state0, state1, etc.".
     */
    virtual std::vector<std::string> post_get_names () const;
    /// Returns DataComponentInterpretation of the solution to be used by PhysicsPostprocessor to output current solution.
    /** Treats every solution state as an independent scalar.
     */
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;
    /// Returns required update flags of the solution to be used by PhysicsPostprocessor to output current solution.
    /** Only update the solution at the output points.
     */
    virtual dealii::UpdateFlags post_get_needed_update_flags () const;
protected:
    /// Anisotropic diffusion matrix
    /** As long as the diagonal components are positive and diagonally dominant
     *  we should have a stable diffusive system
     */
    dealii::Tensor<2,dim,double> diffusion_tensor;

};
} // Physics namespace
} // PHiLiP namespace

#endif
