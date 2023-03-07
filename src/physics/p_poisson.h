#ifndef __P_POISSON__
#define __P_POISSON__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
class p_Poisson : public PhysicsBase <dim, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following: */
    using PhysicsBase<dim,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nstate,real>::source_term;
    using PhysicsBase<dim,nstate,real>::physical_source_term;
public:
    /// Constructor
    p_Poisson (
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function = nullptr,
        const bool                                                has_nonzero_diffusion_input = true,
        const bool                                                has_nonzero_physical_source = true);

    /// Destructor
    ~p_Poisson() {};

    /// Convective flux: 
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative flux: 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Physical source term: 
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Source term for manufactured solution functions
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;

    /** Convective flux contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    std::array<real,nstate> convective_source_term_computed_from_manufactured_solution (
        const dealii::Point<dim,real> &pos) const;

    /** Dissipative flux contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    std::array<real,nstate> dissipative_source_term_computed_from_manufactured_solution (
        const dealii::Point<dim,real> &pos) const;

    /** Convective flux Jacobian 
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /** Dissipative flux Jacobian
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal) const;

    /** Dissipative flux Jacobian wrt gradient component 
     *  Note: Only used for computing the manufactured solution source term;
     *        computed using automatic differentiation
     */
    dealii::Tensor<2,nstate,real> dissipative_flux_directional_jacobian_wrt_gradient_component (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::Tensor<1,dim,real> &normal,
        const int d_gradient) const;

    /** Physical source contribution to the source term
     *  Note: Only used for computing the manufactured solution source term;
     */
    std::array<real,nstate> physical_source_term_computed_from_manufactured_solution(
        const dealii::Point<dim,real> &pos,
        const dealii::types::global_dof_index cell_index) const;

    /// Convective Numerical Split Flux for split form
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Maximum viscous eigenvalue
    real max_viscous_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Computes the entropy variables.
    std::array<real,nstate> compute_entropy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Computes the conservative variables from the entropy variables. 
    std::array<real,nstate> compute_conservative_variables_from_entropy_variables (
                const std::array<real,nstate> &entropy_var) const;

    /// Boundary condition handler
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
    
    /// For post processing purposes
    dealii::Vector<double> post_compute_derived_quantities_scalar (
        const double                &uh,
        const dealii::Tensor<1,dim> &duh,
        const dealii::Tensor<2,dim> &dduh,
        const dealii::Tensor<1,dim> &normals,
        const dealii::Point<dim>    &evaluation_points) const;

    /// For post processing purposes, sets the base names (with no prefix or suffix) of the computed quantities
    std::vector<std::string> post_get_names () const;

    /// For post processing purposes, sets the interpretation of each computed quantity as either scalar or vector
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;
    
    /// For post processing purposes 
    dealii::UpdateFlags post_get_needed_update_flags () const;

protected:
    /// Templated convective flux
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> convective_flux_templated (
        const std::array<real2,nstate> &conservative_soln) const;

    /// Templated dissipative flux
    template <typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> dissipative_flux_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &solution_gradient) const;

    /// Get manufactured solution value
    std::array<real,nstate> get_manufactured_solution_value(
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution gradient
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient(
        const dealii::Point<dim,real> &pos) const;

    /// Wall boundary condition
    void boundary_wall (
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Symmetric boundary condition
    void boundary_slip_wall (
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Farfield boundary conditions 
    void boundary_farfield (
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif