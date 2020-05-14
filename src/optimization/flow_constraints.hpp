#include <deal.II/optimization/rol/vector_adaptor.h>

#include "ROL_Constraint_SimOpt.hpp"

#include "linear_solver/linear_solver.h"

#include "parameters/all_parameters.h"

#include "mesh/free_form_deformation.h"
#include "mesh/meshmover_linear_elasticity.hpp"

#include "dg/dg.h"

namespace PHiLiP {

using dealii_Vector = dealii::LinearAlgebra::distributed::Vector<double>;
using ROL_Vector = ROL::Vector<double>;
using AdaptVector = dealii::Rol::VectorAdaptor<dealii_Vector>;

template<int dim>
class FlowConstraints : public ROL::Constraint_SimOpt<double> {
private:
    /// Smart pointer to DGBase
    std::shared_ptr<DGBase<dim,double>> dg;

    /// FFD used to parametrize the grid and used as design variables.
    FreeFormDeformation<dim> ffd;

    /// Linear solver parameters.
    /** Currently set such that the linear systems are fully solved
     */
    Parameters::LinearSolverParam linear_solver_param;

    /// Set of indices and axes that point to the FFD values that are design variables.
    const std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    /// Design variables values.
    dealii::LinearAlgebra::distributed::Vector<double> ffd_des_var;

protected:
    /// ID used when outputting the flow solution.
    int i_out = 1000;
public:
    using ROL::Constraint_SimOpt<double>::value;

    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_1;
    using ROL::Constraint_SimOpt<double>::applyAdjointJacobian_2;

    using ROL::Constraint_SimOpt<double>::applyAdjointHessian_11;
    using ROL::Constraint_SimOpt<double>::applyAdjointHessian_12;
    using ROL::Constraint_SimOpt<double>::applyAdjointHessian_21;
    using ROL::Constraint_SimOpt<double>::applyAdjointHessian_22;

    /// Constructor
    FlowConstraints(
        std::shared_ptr<DGBase<dim,double>> &_dg, 
        const FreeFormDeformation<dim> &_ffd,
        std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim);

    /// Update the simulation variables.
    void update_1( const ROL_Vector& des_var_sim, bool flag = true, int iter = -1 );

    /// Update the control variables.
    /** Update FFD from design variables and then deforms the mesh.
     */
    void update_2( const ROL_Vector& des_var_ctl, bool flag = true, int iter = -1 );

    /// Solve the Simulation Constraint and returns the constraints values.
    /** In this case, we use the ODESolver to solve the steady state solution
     *  of the flow given the geometry.
     */
    void solve( ROL_Vector& constraint_values, ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;

    /// Returns the current constraint residual given a set of control and simulation variables.
    void value( ROL_Vector& constraint_values, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double &/*tol*/ ) override;
    
    /// Applies the Jacobian of the Constraints w.\ r.\ t.\ the simulation variables onto a vector.
    void applyJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;

    /// Applies the inverse Jacobian of the Constraints w.\ r.\ t.\ the simulation variables onto a vector.
    void applyInverseJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;

    /// Applies the adjoint Jacobian of the Constraints w.\ r.\ t.\ the simulation variables onto a vector.
    void applyInverseAdjointJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;

    /// Applies the Jacobian of the Constraints w.\ r.\ t.\ the control variables onto a vector.
    void applyJacobian_2( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;

    /// Applies the Jacobian of the Constraints w.\ r.\ t.\ the simulation variables onto a vector.
    void applyAdjointJacobian_1( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;


    /// Applies the Jacobian of the Constraints w.\ r.\ t.\ the control variables onto a vector.
    void applyAdjointJacobian_2( ROL_Vector& output_vector, const ROL_Vector& input_vector, const ROL_Vector& des_var_sim, const ROL_Vector& des_var_ctl, double& /*tol*/ ) override;

    /// Applies the adjoint of the Hessian of the constraints w.\r.\t.\ the simulation variables onto a vector.
    /** More specifically, apply
     *  \f[
     *      \mathbf{v}_{out} = \left( \sum_i \psi_i \frac{\partial^2 R_i}{\partial u \partial u} \right)^T \mathbf{v}_{in}
     *  \f]
     *  onto the @p input_vector to obtain the @p output_vector
     */
    void applyAdjointHessian_11 ( ROL_Vector &output_vector,
                                  const ROL_Vector &dual,
                                  const ROL_Vector &input_vector,
                                  const ROL_Vector &des_var_sim,
                                  const ROL_Vector &des_var_ctl,
                                  double &tol) override;

    /// Applies the adjoint of the Hessian of the constraints w.\r.\t.\ the simulation variables onto a vector.
    void applyAdjointHessian_12 ( ROL_Vector &output_vector,
                                  const ROL_Vector &dual,
                                  const ROL_Vector &input_vector,
                                  const ROL_Vector &des_var_sim,
                                  const ROL_Vector &des_var_ctl,
                                  double &tol) override;

    void applyAdjointHessian_21 ( ROL_Vector &output_vector,
                                  const ROL_Vector &dual,
                                  const ROL_Vector &input_vector,
                                  const ROL_Vector &des_var_sim,
                                  const ROL_Vector &des_var_ctl,
                                  double &tol) override;

    void applyAdjointHessian_22 ( ROL_Vector &output_vector,
                                  const ROL_Vector &dual,
                                  const ROL_Vector &input_vector,
                                  const ROL_Vector &des_var_sim,
                                  const ROL_Vector &des_var_ctl,
                                  double &tol) override;

};

} // PHiLiP namespace
