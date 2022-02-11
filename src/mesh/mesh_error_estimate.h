#ifndef __MESHERRORESTIMATE_H__
#define __MESHERRORESTIMATE_H__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/grid_refinement.h>

#include <vector>
#include <iostream>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include "functional/functional.h"
#include "physics/physics.h"

namespace PHiLiP {

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif

/// Abstract class to estimate error for mesh adaptation
class MeshErrorEstimateBase
{

public:

    /// Computes the vector containing errors in each cell.
    virtual dealii::Vector<real> compute_cellwise_errors () = 0;

    /// Constructor
    MeshErrorEstimateBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    /// Virtual Destructor
    virtual ~MeshErrorEstimateBase() = 0;

    /// Pointer to DGBase
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

};

#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
/// Class to compute residual based error
class ResidualErrorEstimate : public MeshErrorEstimateBase <dim, real, MeshType>
{

public:
    /// Computes maximum residual error in each cell.
    dealii::Vector<real> compute_cellwise_errors () override;

    /// Constructor
    ResidualErrorEstimate(std::shared_ptr<DGBase<dim,real,MeshType>> dg_input);

    /// Destructor
    ~ResidualErrorEstimate() {};

};


/// DualWeightedResidualError class
/** 
  * This class computes the discrete adjoint of the system based on a functional of interest and
  * a computed DG solution. Uses the Sacado functions Functional::evaluate_dIdw() and DGBase::assemble_residual()
  * to generate and solve the discrete adjoint system
  * 
  * \f[
  *     \left( \frac{\partial \mathbf{R}}{\partial \mathbf{u}} \right)^T \psi 
  *     + \left(\frac{\partial \mathcal{J}}{\partial \mathbf{u}}\right)^T = \mathbf{0}
  * \f]
  * 
  * Includes functions for solving both the coarse and fine \f$p\f$-enriched adjoint problems. Subscripts \f$H\f$ 
  * and \f$h\f$ are used to denote coarse and fine grid variables respectively.
  * Reference: Venditti AND Darmafol, "Adjoint Error Estimation and Grid Adaptation for Functional Outputs: Application to Quasi-One-Dimensional Flow". Journal of Computational Physics 164, 1 (2000), 204–227.
  */ 
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DualWeightedResidualError : public MeshErrorEstimateBase <dim, real, MeshType>
{
public:

    /// For storing the current state in the adjoint
    enum SolutionRefinementStateEnum{
        coarse, ///< Initial state
        fine,   ///< Refined state
    };

    /// Constructor
    /** Initializes the Adjoint as being in the AdjointEnum::coarse state.
     *  Also stores the current solution and distribution of polynomial orders
     *  for the mesh for converting back to coarse state after refinement.
     */
    DualWeightedResidualError(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    ///destructor
    ~DualWeightedResidualError() {};

    /// Reinitialize Adjoint with the same pointers
    /** Sets adjoint_state to AdjointEnum::coarse and stores the current
     *  solution and polynomial order distribution
     */
    void reinit();
    // to reinitialize with other pointers, just create a new class

    /// Converts the DG solution to specified state
    /** Calls the functions coarse_to_fine() or fine_to_coarse()
     *  if the DualWeightedResidualError::adjoint_state is different than the input \p state
     */
    void convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum refinement_state);

    /// Projects the problem to a p-enriched space
    /** Raises the FE_index on each cell and transfers the coarse 
     *  solution to a fine solution (stored in DGBase::solution)
     */
    void coarse_to_fine();

    /// Return the problem to the original solution and polynomial distribution
    /** Copies the values that were stored in solution_coarse and 
     *  DualWeightedResidualError::coarse_fe_index at intilization
     */
    void fine_to_coarse();

    /// Computes the fine grid adjoint
    /** Converts the state to a refined grid (if needed) and solves for DualWeightedResidualError::adjoint_fine from 
     *  \f[
     *      \left(\left. \frac{\partial \mathbf{R}_h}{\partial \mathbf{u}} \right|_{\mathbf{u}_h^H}\right)^T \psi_h 
     *      + \left(\left. \frac{\partial \mathcal{J}_h}{\partial \mathbf{u}} \right|_{\mathbf{u}_h^H}\right)^T=\mathbf{0}
     *  \f]
     *  where \f$\mathbf{u}_h^H\f$ is the projected solution on the fine grid.
     *  Eq(7) from Venditti and Darmafol (2000), cited above.
     */ 
    dealii::LinearAlgebra::distributed::Vector<real> fine_grid_adjoint();

    /// Computes the coarse grid adjoint
    /** Reverts the state to the coarse grid (if needed) and solves for DualWeightedResidualError::adjoint_coarse from
     * \f[
     *      \left(\left. \frac{\partial \mathbf{R}_H}{\partial \mathbf{u}} \right|_{\mathbf{u}_H}\right)^T \psi_H 
     *      + \left(\left. \frac{\partial \mathcal{J}_H}{\partial \mathbf{u}} \right|_{\mathbf{u}_H}\right)^T=\mathbf{0}
     * \f]
     * Eq(7) from Venditti and Darmafol (2000), cited above, using coarse-grid solution.
     */
    dealii::LinearAlgebra::distributed::Vector<real> coarse_grid_adjoint();

    /// Computes the Dual Weighted Residual (DWR)
    /** Computes DualWeightedResidualError::dual_weighted_resiudal_fine (\f$\eta\f$) on the fine grid. This value should be
     *  zero on the coarse grid due to Galerkin Orthogonality. It is calculated from
     *  \f[
     *      \eta = \mathbf{R}_h(\mathbf{u}_h^H)^T \psi_h
     *  \f]
     *  Uses DualWeightedResidualError::adjoint_fine and should only be called after fine_grid_adjoint().
     *  Eq(6) from Venditti and Darmafol (2000), cited above.
     */
    dealii::Vector<real> dual_weighted_residual();

    /// Computes dual weighted residual error in each cell, by integrating over all quadrature points. Overwrites the virtual function in MeshErrorEstimateBase.
    dealii::Vector<real> compute_cellwise_errors () override;

    /// Computes the sum of dual weighted residual error over all the cells in the domain.
    real total_dual_weighted_residual_error();

    /// Outputs the current solution and adjoint values
    /** Similar to DGBase::output_results_vtk() but will also include the adjoint and derivative_functional_wrt_solution
     *  related to the current adjoint state. Will also output DualWeightedResidualError::dual_weighted_residual_fine
     *  if currenly on the fine grid.
     */
    void output_results_vtk(const unsigned int cycle);
    
    /// Solves the adjoint equation.
    dealii::LinearAlgebra::distributed::Vector<real> compute_adjoint(dealii::LinearAlgebra::distributed::Vector<real> &derivative_functional_wrt_solution, 
                                                                     dealii::LinearAlgebra::distributed::Vector<real> &adjoint_variable);

    /// Functional class pointer
    std::shared_ptr< Functional<dim, nstate, real, MeshType> > functional;
    
    /// original solution
    dealii::LinearAlgebra::distributed::Vector<real> solution_coarse;
    /// functional derivative (on the fine grid)
    dealii::LinearAlgebra::distributed::Vector<real> derivative_functional_wrt_solution_fine;
    /// functional derivative (on the coarse grid)
    dealii::LinearAlgebra::distributed::Vector<real> derivative_functional_wrt_solution_coarse;
    /// fine grid adjoint (\f$\psi_h\f$)
    dealii::LinearAlgebra::distributed::Vector<real> adjoint_fine;
    /// coarse grid adjoint (\f$\psi_H\f$)
    dealii::LinearAlgebra::distributed::Vector<real> adjoint_coarse;
    /// dual weighted residual
    /** always on the fine grid due to galerkin orthogonality
     */ 
    dealii::Vector<real> dual_weighted_residual_fine;
    
    /// Original FE_index distribution
    dealii::Vector<real> coarse_fe_index;

    /// Current refinement state of the solution
    SolutionRefinementStateEnum solution_refinement_state;

protected:
    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // DualWeightedResidualError class

} // namespace PHiLiP

#endif
