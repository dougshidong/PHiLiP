#ifndef __JACOBIAN_VECTOR_PRODUCT__
#define __JACOBIAN_VECTOR_PRODUCT__

#include "dg/dg.h"

namespace PHiLiP {
namespace ODE{

//template things
//UPDATE WITH #IF STUFF
template <int dim, typename real, typename MeshType>
class JacobianVectorProduct{
public:

    JacobianVectorProduct(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input);

    ~JacobianVectorProduct() {};

    void reinit_for_next_timestep(double dt_input,
                double epsilon_input,
                dealii::LinearAlgebra::distributed::Vector<double> &previous_step_solution_input);

    void reinit_for_next_Newton_iter(dealii::LinearAlgebra::distributed::Vector<double> &current_solution_estimate_input);

    // Application of matrix to vector src. Write result into dst.
    void vmult (dealii::LinearAlgebra::distributed::Vector<double> &dst,
                const dealii::LinearAlgebra::distributed::Vector<double> &src) const;
    
    /// Unsteady residual = dw/dt - R
    dealii::LinearAlgebra::distributed::Vector<double> compute_unsteady_residual(dealii::LinearAlgebra::distributed::Vector<double> &solution,
            bool do_negate = false) const;
protected:

    /// pointer to dg
    std::shared_ptr<DGBase<dim,real,MeshType>> dg;

    /// timestep size for implicit Euler step
    double dt;
    
    /// small number for finite difference
    double epsilon;
    
    /// solution at previous timestep
    // POSSIBLY COULD USE A POINTER HERE
    dealii::LinearAlgebra::distributed::Vector<double> previous_step_solution;
    
    
    /// current estimate for the solution
    // NOTE: the same vector is stored in JFNK_solver
    // Should not be stored twice
    // Could store in this class, with risk of being counterintuitive
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate;
    
    /// residual of current estimate for the solution
    dealii::LinearAlgebra::distributed::Vector<double> current_solution_estimate_residual;
};

}
}
#endif
