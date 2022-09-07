#ifndef __SET_INITIAL_CONDITION_H__
#define __SET_INITIAL_CONDITION_H__

#include "parameters/all_parameters.h"
#include "dg/dg.h"
#include "initial_condition_function.h"
#include <string>

namespace PHiLiP {

/// Class for setting/applying the initial condition
template <int dim, int nstate, typename real>
class SetInitialCondition
{
public:
    /// Applies the given initial condition function to the given dg object
    static void set_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function_input,
        std::shared_ptr< PHiLiP::DGBase<dim,real> > dg_input,
        const Parameters::AllParameters *const parameters_input);

    /// Reads values from file and projects
    static void read_values_from_file_and_project(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg,
        const std::string filename_with_extension);
private:
    ///Interpolates the initial condition function onto the dg solution.
    static void interpolate_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
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
    static void project_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg); 
};

}//end PHiLiP namespace
#endif
