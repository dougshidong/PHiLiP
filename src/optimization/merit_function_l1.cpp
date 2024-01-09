#include "merit_function_l1.hpp"
#include "rol_to_dealii_vector.hpp"

namespace PHiLiP {

MeritFunctionL1::MeritFunctionL1(
    const ROL::Ptr<ROL::Objective<double> > &obj_,
    const ROL::Ptr<ROL::Constraint<double> > &con_,
    const ROL::Vector<double> &constraint_vec_)
    : obj(obj_)
    , con(con_)
    , penalty_parameter(0.0)
{ 
    constraint_vec = constraint_vec_.clone();
}

void MeritFunctionL1::set_penalty_parameter(const double penalty_parameter_input)
{
    penalty_parameter = penalty_parameter_input;
}


void MeritFunctionL1::update(
    const ROL::Vector<double> &x,
    bool flag,
    int iter)
{
    obj->update(x,flag,iter);
    con->update(x,flag,iter);
}

double MeritFunctionL1::value(
    const ROL::Vector<double> &x,
    double &tol )
{
    const double objective_val = obj->value(x,tol);
    con->value(*constraint_vec,x,tol); 
    const double constraint_l1_norm = evaluate_l1_norm(*constraint_vec);

    return (objective_val + penalty_parameter*constraint_l1_norm);
}

double MeritFunctionL1::compute_directional_derivatve(
    const ROL::Vector<double> &x,
    const ROL::Vector<double> &search_direction)
{
    double tol = std::sqrt(ROL::ROL_EPSILON<double>());
    ROL::Ptr<ROL::Vector<double>> objective_gradient = search_direction.clone();
    obj->gradient(*objective_gradient, x, tol);
    const double directional_derivative = objective_gradient->dot(search_direction);

    con->value(*constraint_vec,x,tol); 
    const double constraint_l1_norm = evaluate_l1_norm(*constraint_vec);

    return (directional_derivative - penalty_parameter*constraint_l1_norm);
}

double MeritFunctionL1::evaluate_l1_norm(
    const ROL::Vector<double> &input_vector)
{
    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);
    return dealii_input.l1_norm();
}

void MeritFunctionL1::gradient(
    ROL::Vector<double> &/*g*/,
    const ROL::Vector<double> &/*x*/,
    double &/*tol*/ ) 
{
    std::cout<<"In MeritFunctionL1::gradient. Shouldn't have reached here. Aborting..."<<std::endl;
    std::abort();
}
  
void MeritFunctionL1::hessVec(
    ROL::Vector<double> &/*hv*/,
    const ROL::Vector<double> &/*v*/,
    const ROL::Vector<double> &/*x*/,
    double &/*tol*/ )
{
    std::cout<<"In MeritFunctionL1::hesVec. Shouldn't have reached here. Aborting..."<<std::endl;
    std::abort();
}


} // PHiLiP namespace
