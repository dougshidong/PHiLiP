#ifndef __INITIAL_CONDITION_H__
#define __INITIAL_CONDITION_H__

#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "initial_condition_function.h"

namespace PHiLiP {

///Initial Condition class.

template <int dim, int nstate, typename real>
class InitialCondition
{
public:
    ///Constructor
    InitialCondition(
        std::shared_ptr< PHiLiP::DGBase<dim, real> > dg_input,
        const Parameters::AllParameters *const parameters_input);
    ///Destructor
    ~InitialCondition();

    ///Input parameters.
    const Parameters::AllParameters *const all_parameters;

    /// Smart pointer to DGBase
    std::shared_ptr<PHiLiP::DGBase<dim,real>> dg;

    ///Interpolates the initial condition function onto the dg solution.
    void interpolate_initial_condition(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg); 
    ///Projects the initial condition function physical value onto the dg solution modal coefficients.
    /*This is critical for curvilinear coordinates since the physical coordinates are
    * nonlinear functions of the reference coordinates. This leads to the interpolation
    * of the dealii::function not equivalent to the projection of the function on the flux
    * nodes to the modal coefficients. The latter is the correct form.
    * This differs from dealii::project, since here, the projection is purely in the reference space. That is, we solve
    * \f$ \hat{\mathbf{f}}^T = \mathbf{M}^{-1} \int_{\mathbf{\Omega}_r} \mathbf{\chi}^T \mathbf{f} d\mathbf{\Omega}_r \f$ where \f$\mathbf{\chi}\f$ are the reference basis functions, and \f$\mathbf{M}\f$ is the reference mass matrix.
    * Note that the physical mapping only appears in the function to be projected \f$\mathbf{f}\f$ and the determinant of the metric Jacobian is not in the projection.
    * For more information, please refer to Sections 3.1 and 3.2 in Cicchino, Alexander, et al. "Provably stable flux reconstruction high-order methods on curvilinear elements." Journal of Computational Physics (2022): 111259.
    */
    void project_initial_condition(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg); 
        
    protected:
    ///Initial condition function
    std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function;
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

}//end PHiLiP namespace
#endif
