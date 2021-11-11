#ifndef __FUNCTIONAL_OBJECTIVE_H__
#define __FUNCTIONAL_OBJECTIVE_H__

#include "ROL_Objective_SimOpt.hpp"
#include "functional/functional.h"

namespace PHiLiP {

using ROL_Vector = ROL::Vector<double>;

/// Interface between the ROL::Objective_SimOpt PHiLiP::Functional.
/** Uses FFD to parametrize the geometry.
*  An update on the simulation variables updates the DGBase object within the Functional
*  and an update on the control variables updates the FreeFormDeformation object, which in
*  turn, updates the DGBase.HighOrderGrid.volume_nodes.
*/
template <int dim, int nstate>
class FunctionalObjective : public ROL::Objective_SimOpt<double> {
private:
    /// Functional to be evaluated
    Functional<dim,nstate,double> &functional;

public:

    /// Constructor.
    FunctionalObjective(Functional<dim,nstate,double> &_functional);

    using ROL::Objective_SimOpt<double>::value;
    using ROL::Objective_SimOpt<double>::update;

    /// Update the simulation and control variables.
    void update(
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            bool flag = true,
            int iter = -1) override;

    /// Returns the value of the Functional object.
    double value(
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

    /// Returns the gradient w.\ r.\ t.\ the simulation variables of the Functional object.
    void gradient_1(
            ROL::Vector<double> &g,
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

    /// Returns the gradient w.\ r.\ t.\ the control variables of the Functional object.
    void gradient_2(
            ROL::Vector<double> &g,
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

    /// Applies the functional Hessian w.\ r.\ t.\ the simulation variables onto a vector.
    /** More specifically,
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial w \partial w} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_11(
            ROL::Vector<double> &output_vector,
            const ROL::Vector<double> &input_vector,
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

    /// Applies the functional Hessian w.\ r.\ t.\ the simulation and control variables onto a vector.
    /** More specifically,
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial w \partial x} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_12(
            ROL::Vector<double> &output_vector,
            const ROL::Vector<double> &input_vector,
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

    /// Applies the functional Hessian w.\ r.\ t.\ the control and simulation variables onto a vector.
    /** More specifically,
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial x \partial w} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_21(
            ROL::Vector<double> &output_vector,
            const ROL::Vector<double> &input_vector,
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

    /// Applies the functional Hessian w.\ r.\ t.\ the control variables onto a vector.
    /** More specifically,
     *  \f[
     *      \mathbf{v}_{output} = \left( \frac{\partial^2 \mathcal{I}}{\partial x \partial x} \right)^T \mathbf{v}_{input}
     *  \f]
     */
    void hessVec_22(
            ROL::Vector<double> &output_vector,
            const ROL::Vector<double> &input_vector,
            const ROL::Vector<double> &des_var_sim,
            const ROL::Vector<double> &des_var_ctl,
            double &/*tol*/ ) override;

}; // FunctionalObjective
} // PHiLiP namespace

#endif