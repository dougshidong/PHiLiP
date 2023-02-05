#include "channel_flow.h"
#include <deal.II/dofs/dof_tools.h>
// #include <deal.II/grid/grid_tools.h>
// #include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
// #include <deal.II/base/tensor.h>
#include "math.h"
#include <deal.II/base/quadrature_lib.h>
#include "mesh/gmsh_reader.hpp"
#include <deal.II/grid/grid_generator.h>

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// CHANNEL FLOW CLASS
//=========================================================
template <int dim, int nstate>
ChannelFlow<dim, nstate>::ChannelFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , channel_height(this->all_param.flow_solver_param.turbulent_channel_height)
        , channel_bulk_reynolds_number(this->all_param.flow_solver_param.turbulent_channel_bulk_reynolds_number)
{ }

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
    std::string flow_type_string;
    // if(this->is_taylor_green_vortex || this->is_decaying_homogeneous_isotropic_turbulence) {
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
        this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
        this->pcout << "- - Reynolds number based on bulk flow: " << this->all_param.flow_solver_param.turbulent_channel_bulk_reynolds_number << std::endl;
        this->pcout << "- - Channel height: " << this->all_param.flow_solver_param.turbulent_channel_height << std::endl;
    // }
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> ChannelFlow<dim,nstate>::generate_grid() const
{
    // Dummy triangulation
    // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    const int grid_order = 0;
    std::shared_ptr<HighOrderGrid<dim,double>> mesh = read_gmsh<dim, dim> (mesh_filename, grid_order, use_mesh_smoothing);
    return mesh->triangulation;
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    const int grid_order = this->all_param.flow_solver_param.grid_degree;
    std::shared_ptr<HighOrderGrid<dim,double>> mesh = read_gmsh<dim, dim> (mesh_filename, grid_order, use_mesh_smoothing);
    dg->set_high_order_grid(mesh);
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::update_model_variables(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double integrated_density_over_domain = get_integrated_density_over_domain(*dg);

    dg->set_model_variables(
        integrated_density_over_domain,
        this->channel_height,
        this->channel_bulk_reynolds_number,
        this->get_time_step());
}

template<int dim, int nstate>
double ChannelFlow<dim, nstate>::get_integrated_density_over_domain(DGBase<dim, double> &dg) const
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
    const double mpi_sum_integral_value = dealii::Utilities::MPI::sum(integral_value, this->mpi_communicator);
    return mpi_sum_integral_value;
}

#if PHILIP_DIM==3
template class ChannelFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace