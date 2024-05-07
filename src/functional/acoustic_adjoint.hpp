#ifndef __ACOUSTIC_ADJOINT_H__
#define __ACOUSTIC_ADJOINT_H__

#include <vector>
#include <iostream>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include "parameters/all_parameters.h"

#include "functional.h"
#include "dg/dg_base.hpp"
#include "physics/physics.h"

namespace PHiLiP {

/// AcousticAdjoint class
/** 
  * This class computes the discrete adjoint of the system based on a acoustic functional of interest and
  * a computed DG solution. Uses the Sacado functions Functional::evaluate_functional() and DGBase::assemble_residual()
  * to generate and solve the discrete adjoint system
  * 
  * \f[
  *     \left( \frac{\partial \mathbf{R}}{\partial \mathbf{u}} \right)^T \psi 
  *     + \left(\frac{\partial \mathcal{J}}{\partial \mathbf{u}}\right)^T = \mathbf{0}
  * \f]
  * 
  * Includes functions for evaluating objective surface gradient dIdXv, where Xv indicates all volume and surface nodes.
  */ 
 #if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class AcousticAdjoint
{
public:
    /// Constructor
    /** Initializes the Adjoint by dg and functional pointer.
     */
    AcousticAdjoint(
        std::shared_ptr< DGBase<dim,real,MeshType> > dg_input,
        std::shared_ptr< Functional<dim, nstate, real, MeshType> > functional_input);

    /// Destructor
    ~AcousticAdjoint();

    /// Function to solve adjoint linear system.
    void compute_adjoint();

    /// Function to evaluate objective gradient wrt volume nodes.
    void compute_dIdXv();

    /// Outputs the adjoint solutions.
    /** Similar to DGBase::output_results_vtk() but generates separate file only includes the adjoint solutions and dIdw.
     */
    void output_results_vtk(const unsigned int cycle);

    /// DG class pointer
    std::shared_ptr< DGBase<dim,real,MeshType> > dg;
    /// Functional class pointer
    std::shared_ptr< Functional<dim, nstate, real, MeshType> > functional;
    
    /// Grid
    std::shared_ptr<MeshType> triangulation;
    /// functional derivative wrt solution
    dealii::LinearAlgebra::distributed::Vector<real> dIdw;
    /// functional derivative wrt volume nodes
    dealii::LinearAlgebra::distributed::Vector<real> dIdXv;
    /// adjoint (\f$\psi_h\f$)
    dealii::LinearAlgebra::distributed::Vector<real> adjoint;

protected:
    /// MPI communicator
    MPI_Comm mpi_communicator;
    /// Parallel std::cout that only outputs on mpi_rank==0
    dealii::ConditionalOStream pcout;

}; // AcousticAdjoint class


} // PHiLiP namespace

#endif // __ACOUSTIC_ADJOINT_H__