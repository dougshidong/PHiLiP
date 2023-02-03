#include "flow_solver_channel_flow.h"
#include <deal.II/dofs/dof_tools.h>
// #include <deal.II/grid/grid_tools.h>
// #include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
// #include <deal.II/base/tensor.h>
#include "math.h"
#include <deal.II/base/quadrature_lib.h>

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// FLOW SOLVER CHANNEL FLOW CLASS
//=========================================================
template <int dim, int nstate>
FlowSolverChannelFlow<dim, nstate>::FlowSolverChannelFlow(
    const PHiLiP::Parameters::AllParameters *const parameters_input, 
    std::shared_ptr<FlowSolverCaseBase<dim, nstate>> flow_solver_case_input,
    const dealii::ParameterHandler &parameter_handler_input)
    : FlowSolver(parameters_input, flow_solver_case_input, parameter_handler_input)
{ }

template <int dim, int nstate>
void FlowSolverChannelFlow<dim,nstate>::update_model_variables()
{
    // initialize
    double integrated_density_over_domain = get_integrated_density_over_domain(this->dg);

    // update the model variables
    this->dg->pde_model_double->integrated_density_over_domain  = integrated_density_over_domain;
    this->dg->pde_model_fad->integrated_density_over_domain     = integrated_density_over_domain;
    this->dg->pde_model_rad->integrated_density_over_domain     = integrated_density_over_domain;
    this->dg->pde_model_fad_fad->integrated_density_over_domain = integrated_density_over_domain;
    this->dg->pde_model_rad_fad->integrated_density_over_domain = integrated_density_over_domain;
}

template<int dim, int nstate>
double FlowSolverChannelFlow<dim, nstate>::get_integrated_density_over_domain(DGBase<dim, double> &dg) const
{
    double integral_value = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values /*| dealii::update_gradients*/ | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;
    // std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            // for (int s=0; s<nstate; ++s) {
            //     for (int d=0; d<dim; ++d) {
            //         soln_grad_at_q[s][d] = 0.0;
            //     }
            // }
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                // soln_grad_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_grad_component(idof,iquad,istate);
            }
            // const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            double integrand_value = soln_at_q[0]; // density
            integral_value += integrand_value * fe_values_extra.JxW(iquad);
        }
    }
    
    return dealii::Utilities::MPI::sum(integral_value, this->mpi_communicator);
}

#if PHILIP_DIM==1
template class FlowSolverChannelFlow <PHILIP_DIM,PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
template class FlowSolverChannelFlow <PHILIP_DIM,1>;
template class FlowSolverChannelFlow <PHILIP_DIM,2>;
template class FlowSolverChannelFlow <PHILIP_DIM,3>;
template class FlowSolverChannelFlow <PHILIP_DIM,4>;
template class FlowSolverChannelFlow <PHILIP_DIM,5>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace