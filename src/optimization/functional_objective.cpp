#include "functional_objective.h"
#include "rol_to_dealii_vector.hpp"
#include <deal.II/optimization/rol/vector_adaptor.h>

namespace PHiLiP {

template <int dim, int nstate>
FunctionalObjective<dim,nstate>::FunctionalObjective(Functional<dim,nstate,double> &_functional)
        : functional(_functional)
{
}


template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::update(
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        bool /*flag*/, int /*iter*/)
{
    functional.set_state(ROL_vector_to_dealii_vector_reference(des_var_sim));
    functional.dg->high_order_grid->volume_nodes = ROL_vector_to_dealii_vector_reference(des_var_ctl);
}


template <int dim, int nstate>
double FunctionalObjective<dim,nstate>::value(
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &tol )
{
    // Tolerance tends to not be used except in the case of a reduced objective function.
    // In that scenario, tol is the constraint norm.
    // If the flow has not converged (>1e-5 or is nan), simply return a high functional.
    // This is likely happening in the linesearch while optimizing in the reduced-space.
    if (tol > 1e-5 || std::isnan(tol)) return 1e200;
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    return functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
}

template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::gradient_1(
        ROL::Vector<double> &gradient_sim,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = true;
    const bool compute_dIdX = false;
    const bool compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
    auto &dIdW = ROL_vector_to_dealii_vector_reference(gradient_sim);
    dIdW = functional.dIdw;
}

template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::gradient_2(
        ROL::Vector<double> &gradient_ctl,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false, compute_dIdX = true, compute_d2I = false;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dIdXv = functional.dIdX;

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(gradient_ctl);
    dealii_output = dIdXv;
}

template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::hessVec_11(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);
    auto &hv = ROL_vector_to_dealii_vector_reference(output_vector);

    functional.d2IdWdW.vmult(hv, dealii_input);
}

template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::hessVec_12(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    {
        const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
        functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        functional.d2IdWdX.vmult(dealii_output, dealii_input);
    }
}

template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::hessVec_21(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);

    const bool compute_dIdW = false;
    const bool compute_dIdX = false;
    const bool compute_d2I = true;
    functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );

    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    auto d2IdXdW_input = functional.dg->high_order_grid->volume_nodes;
    functional.d2IdWdX.Tvmult(d2IdXdW_input, dealii_input);

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    dealii_output = d2IdXdW_input;
}

template <int dim, int nstate>
void FunctionalObjective<dim,nstate>::hessVec_22(
        ROL::Vector<double> &output_vector,
        const ROL::Vector<double> &input_vector,
        const ROL::Vector<double> &des_var_sim,
        const ROL::Vector<double> &des_var_ctl,
        double &/*tol*/ )
{
    update(des_var_sim, des_var_ctl);


    const auto &dealii_input = ROL_vector_to_dealii_vector_reference(input_vector);

    auto d2IdXdX_input = functional.dg->high_order_grid->volume_nodes;
    {
        const bool compute_dIdW = false, compute_dIdX = false, compute_d2I = true;
        functional.evaluate_functional( compute_dIdW, compute_dIdX, compute_d2I );
        functional.d2IdXdX.vmult(d2IdXdX_input, dealii_input);
    }

    auto &dealii_output = ROL_vector_to_dealii_vector_reference(output_vector);
    dealii_output = d2IdXdX_input;

}

template class FunctionalObjective <PHILIP_DIM,1>;
template class FunctionalObjective <PHILIP_DIM,2>;
template class FunctionalObjective <PHILIP_DIM,3>;
template class FunctionalObjective <PHILIP_DIM,4>;
template class FunctionalObjective <PHILIP_DIM,5>;

} // PHiLiP namespace
