#ifndef __MERIT_FUNCTION_L1_H__
#define __MERIT_FUNCTION_L1_H__

#include "ROL_Objective.hpp"
#include "ROL_Constraint.hpp"
#include "ROL_Vector.hpp"


namespace PHiLiP {

class MeritFunctionL1 : public ROL::Objective<double> 
{
private:
    const ROL::Ptr<ROL::Objective<double> > obj;
    const ROL::Ptr<ROL::Constraint<double> > con;
    double penalty_parameter;
    ROL::Ptr<ROL::Vector<double> > constraint_vec;

    double evaluate_l1_norm(const ROL::Vector<double> &input_vector);
    double evaluate_linfty_norm(const ROL::Vector<double> &input_vector);

public:
    /// Constructor
    MeritFunctionL1(const ROL::Ptr<ROL::Objective<double> > &obj_,
                    const ROL::Ptr<ROL::Constraint<double> > &con_,
                    const ROL::Vector<double> &constraint_vec_);

    void set_penalty_parameter(const ROL::Vector<double> &x);

    double compute_directional_derivatve(
        const ROL::Vector<double> &x,
        const ROL::Vector<double> &search_direction);

    /// Update the simulation and control variables.
    void update(
        const ROL::Vector<double> &x,
        bool flag = true,
        int iter = -1) override;
  
    /// Returns the value of the Functional object.
    double value(
        const ROL::Vector<double> &x,
        double &/*tol*/ ) override;
  
    /// Returns the gradient w.\ r.\ t.\ the simulation variables of the Functional object.
    void gradient(
        ROL::Vector<double> &g,
        const ROL::Vector<double> &x,
        double &/*tol*/ ) override;
  
    void hessVec(
        ROL::Vector<double> &hv,
        const ROL::Vector<double> &v,
        const ROL::Vector<double> &x,
        double &/*tol*/ ) override;

};
} // PHiLiP namespace
#endif
