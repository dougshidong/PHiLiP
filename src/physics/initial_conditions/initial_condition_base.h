#ifndef __INITIAL_CONDITION_BASE_H__
#define __INITIAL_CONDITION_BASE_H__

#include <deal.II/lac/vector.h>
#include <deal.II/base/function.h>
#include "parameters/all_parameters.h"
#include "parameters/parameters.h"
#include "dg/dg.h"
#include "initial_condition_function.h"

namespace PHiLiP {

///Initial Condition Base class.

template <int dim, typename real>
class InitialConditionBase
{
public:
    ///Constructor
    InitialConditionBase(
        std::shared_ptr< PHiLiP::DGBase<dim, real> > dg_input,
        const Parameters::AllParameters *const parameters_input,
        const int nstate_input);
    ///Destructor
    ~InitialConditionBase();

    ///Input parameters.
    const Parameters::AllParameters *const all_parameters;
    ///Number of states.
    const int nstate;

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
    */
    void project_initial_condition(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg); 
        
    protected:
    ///Initial condition function
    std::shared_ptr< InitialConditionFunction<dim,double> > initial_condition_function;
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

}//end PHiLiP namespace
#endif
