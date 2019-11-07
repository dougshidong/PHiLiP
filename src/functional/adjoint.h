#ifndef __ADJOINT_H__
#define __ADJOINT_H__

/* includes */
#include <vector>
#include <iostream>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include "parameters/all_parameters.h"

#include "functional.h"
#include "dg/dg.h"
#include "dg/high_order_grid.h"
#include "physics/physics.h"

namespace PHiLiP {

// for storing the current state in the adjoint
enum AdjointEnum {
    coarse,
    fine,
};

// Adjoint class
template <int dim, int nstate, typename real>
class Adjoint
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
public:

    //constructor
    Adjoint(
        DGBase<dim,real> &_dg,
        Functional<dim, nstate, real> &_functional,
        const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &_physics);

    //destructor
    ~Adjoint();

    // for conversions between states
    void convert_to_state(AdjointEnum state);

    // project to a p-enriched solution
    void coarse_to_fine();
    // return to teh original solution and DOF distribution
    void fine_to_coarse();

    // compute the fine grid adjoint
    dealii::LinearAlgebra::distributed::Vector<real> fine_grid_adjoint();

    // compute the coarse grid adjoint
    dealii::LinearAlgebra::distributed::Vector<real> coarse_grid_adjoint();

    // compute the dual weighted residual
    dealii::Vector<real> dual_weighted_residual();

    // for outputs (copy mostly of the one in DGbase) - Leaving this out for now
    void output_results_vtk(const unsigned int cycle);

    // DG class 
    DGBase<dim,real> &dg;
    // Functional class
    Functional<dim, nstate, real> &functional;
    // physics for calling the functional class 
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics;
    
    // fine grid triangulation
    Triangulation *const triangulation;
    // Solution Transfer to fine grid
    dealii::parallel::distributed::SolutionTransfer< 
        dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim> 
        > solution_transfer;
    // original solution
    dealii::LinearAlgebra::distributed::Vector<real> solution_coarse;
    // functional derivative (fine grid)
    dealii::LinearAlgebra::distributed::Vector<real> dIdw_fine;
    // functional derivative (coarse grid)
    dealii::LinearAlgebra::distributed::Vector<real> dIdw_coarse;
    // fine grid adjoint
    dealii::LinearAlgebra::distributed::Vector<real> adjoint_fine;
    // coarse grid adjoint
    dealii::LinearAlgebra::distributed::Vector<real> adjoint_coarse;
    // dual weighted residual (always fine due to galerkin orthogonality)
    dealii::Vector<real> dual_weighted_residual_fine;
    
    // stores the original FE_index distribution
    dealii::Vector<real> coarse_fe_index;

    // adjoint state for conversion tracking
    AdjointEnum adjoint_state;

protected:
    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // Adjoint class


} // PHiLiP namespace

#endif // __ADJOINT_H__