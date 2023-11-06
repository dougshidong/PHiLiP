#ifndef __MODEL__
#define __MODEL__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/types.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/la_parallel_vector.templates.h>
#include <deal.II/numerics/data_component_interpretation.h>

#include "parameters/parameters_manufactured_solution.h"
#include "physics/manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Physics model additional terms and equations to the baseline physics. 
template <int dim, int nstate, typename real>
class ModelBase
{
public:
	/// Constructor
	ModelBase(
        std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function_input = nullptr);

    /// Virtual destructor required for abstract classes.
    virtual ~ModelBase() = 0;

    /// Manufactured solution function
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function;

protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

public:
    /// Convective flux terms additional to the baseline physics (including convective flux terms in additional PDEs of model)
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
    convective_flux (
        const std::array<real,nstate> &conservative_soln) const = 0;

    /// Dissipative flux terms additional to the baseline physics (including dissipative flux terms in additional PDEs of model)
	virtual std::array<dealii::Tensor<1,dim,real>,nstate> 
	dissipative_flux (
    	const std::array<real,nstate> &conservative_soln,
    	const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Dissipative flux terms additional to the baseline physics (including dissipative flux terms in additional PDEs of model) dot normal vector
    virtual std::array<real,nstate> dissipative_flux_dot_normal (
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const std::array<real,nstate> &filtered_solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &filtered_solution_gradient,
        const bool on_boundary,
        const dealii::types::global_dof_index cell_index,
        const dealii::Tensor<1,dim,real> &normal,
        const int boundary_type) const = 0;

    /// Convective eigenvalues of the additional models' PDEs
    /** Note: Only support for zero convective term additional to the baseline physics 
      *       The entries associated with baseline physics are assigned to be zero */
    virtual std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const = 0;

    /// Maximum convective eigenvalue of the additional models' PDEs
    /** Note: Only support for zero convective term additional to the baseline physics */
    virtual real max_convective_eigenvalue (const std::array<real,nstate> &soln) const = 0;

    /// Maximum convective normal eigenvalue (used in Lax-Friedrichs) of the additional models' PDEs
    /** Note: Only support for zero convective term additional to the baseline physics */
    virtual real max_convective_normal_eigenvalue (
        const std::array<real,nstate> &soln,
        const dealii::Tensor<1,dim,real> &normal) const = 0;

    /// Physical source terms additional to the baseline physics (including physical source terms in additional PDEs of model)
    virtual std::array<real,nstate> physical_source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// Manufactured source terms additional to the baseline physics (including manufactured source terms in additional PDEs of model)
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &solution,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const = 0;

    /// Boundary condition handler
    void boundary_face_values (
        const int boundary_type,
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Returns current vector solution to be used by PhysicsPostprocessor to output current solution.
    /** The implementation in this Model base class simply returns the stored solution. */
    virtual dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &/*duh*/,
        const std::vector<dealii::Tensor<2,dim> > &/*dduh*/,
        const dealii::Tensor<1,dim>               &/*normals*/,
        const dealii::Point<dim>                  &/*evaluation_points*/) const;

    /// Returns DataComponentInterpretation of the solution to be used by PhysicsPostprocessor to output current solution.
    /** Treats every solution state as an independent scalar. */
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const;

    /// Returns names of the solution to be used by PhysicsPostprocessor to output current solution.
    /** The implementation in this Model base class simply returns "state(dim+2+0), state(dim+2+1), etc.". */
    virtual std::vector<std::string> post_get_names () const;

    // Quantities needed to be updated by DG for the model -- accomplished by DGBase update_model_variables()
    dealii::LinearAlgebra::distributed::Vector<int> cellwise_poly_degree; ///< Cellwise polynomial degree
    dealii::LinearAlgebra::distributed::Vector<double> cellwise_volume; ////< Cellwise element volume
    double bulk_density; ///< bulk density
    double time_step; ///< Current time step
    dealii::LinearAlgebra::distributed::Vector<double> cellwise_mean_strain_rate_tensor_magnitude; ////< Cellwise mean strain rate tensor magnitude; used for shear-improved eddy viscosity model

protected:
    /// Evaluate the manufactured solution boundary conditions.
    virtual void boundary_manufactured_solution (
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Wall boundary condition
    virtual void boundary_wall (
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Outflow Boundary Condition 
    virtual void boundary_outflow (
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Inflow boundary conditions 
    virtual void boundary_inflow (
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Farfield boundary conditions based on freestream values
    virtual void boundary_farfield (
        std::array<real,nstate> &soln_bc) const;

    /// Riemann-based farfield boundary conditions based on freestream values.
    virtual void boundary_riemann (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc) const;

    /// Slip wall boundary condition
    virtual void boundary_slip_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif
