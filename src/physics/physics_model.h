#ifndef __PHYSICS_MODEL__
#define __PHYSICS_MODEL__

/// Files for the baseline physics
#include "physics.h"
#include "navier_stokes.h"
#include "model.h"

namespace PHiLiP {
namespace Physics {

/// Physics Model equations. Derived from PhysicsBase, holds a baseline physics and model terms and equations. 
template <int dim, int nstate, typename real, int nstate_baseline_physics>
class PhysicsModel : public PhysicsBase <dim, nstate, real>
{
public:
    /// Constructor
    PhysicsModel(
        const Parameters::AllParameters                              *const parameters_input,
        Parameters::AllParameters::PartialDifferentialEquation       baseline_physics_type,
        std::shared_ptr< ModelBase<dim,nstate,real> >                model_input,
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> >    manufactured_solution_function,
        const bool                                                   has_nonzero_diffusion,
        const bool                                                   has_nonzero_physical_source);

    /// Destructor
    ~PhysicsModel() {};

    /// Number of model equations (i.e. those additional to the baseline physics)
    const int n_model_equations;

    /// Baseline physics object with nstate==nstate_baseline_physics
    std::shared_ptr< PhysicsBase<dim,nstate_baseline_physics,real> > physics_baseline;

    /// Model object
    std::shared_ptr< ModelBase<dim,nstate,real> > model;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const;

    /// Dissipative (i.e. viscous) flux: \f$ \mathbf{F}_{diss} \f$ 
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Physical source term
    std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index,
        const real post_processed_scalar) const;

    /// Source term that does not require differentiation.
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;

    //===========================================================================================
    // All other functions required by PhysicsBase:
    //===========================================================================================
    /// Convective Numerical Split Flux for split form
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Computes the entropy variables.
    std::array<real,nstate> compute_entropy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Computes the conservative variables from the entropy variables.
    std::array<real,nstate> compute_conservative_variables_from_entropy_variables (
                const std::array<real,nstate> &entropy_var) const;

    /** Spectral radius of convective term Jacobian.
     *  Used for scalar dissipation
     */
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const;

    /// Maximum convective eigenvalue used in Lax-Friedrichs
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Maximum viscous eigenvalue.
    real max_viscous_eigenvalue (const std::array<real,nstate> &soln) const;

    /// Evaluates boundary values and gradients on the other side of the face.
    void boundary_face_values (
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
    dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &duh,
        const std::vector<dealii::Tensor<2,dim> > &dduh,
        const dealii::Tensor<1,dim>               &normals,
        const dealii::Point<dim>                  &evaluation_points) const;
    
    /// Returns names of the solution to be used by PhysicsPostprocessor to output current solution.
    /** The implementation in this Physics base class simply returns "state0, state1, etc.".
     */
    std::vector<std::string> post_get_names () const;
    
    /// Returns DataComponentInterpretation of the solution to be used by PhysicsPostprocessor to output current solution.
    /** Treats every solution state as an independent scalar.
     */
    std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;
    
    /// Returns required update flags of the solution to be used by PhysicsPostprocessor to output current solution.
    /** Only update the solution at the output points.
     */
    dealii::UpdateFlags post_get_needed_update_flags () const; 

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.
    const int n_mpi; ///< Number of MPI processes.
    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;
};

} // Physics namespace
} // PHiLiP namespace

#endif
