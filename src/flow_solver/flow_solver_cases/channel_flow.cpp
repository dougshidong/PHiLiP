#include "channel_flow.h"
#include <deal.II/dofs/dof_tools.h>
// #include <deal.II/grid/grid_tools.h>
// #include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
// #include <deal.II/base/tensor.h>
#include "math.h"
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
// #include "mesh/gmsh_reader.hpp" // uncomment this to use the gmsh reader

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// CHANNEL FLOW CLASS
//=========================================================
template <int dim, int nstate>
ChannelFlow<dim, nstate>::ChannelFlow(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicTurbulence<dim, nstate>(parameters_input)
        , channel_height(this->all_param.flow_solver_param.turbulent_channel_height)
        , half_channel_height(0.5*channel_height)
        , channel_friction_velocity_reynolds_number(this->all_param.flow_solver_param.turbulent_channel_friction_velocity_reynolds_number)
        , number_of_cells_x_direction(this->all_param.flow_solver_param.turbulent_channel_number_of_cells_x_direction)
        , number_of_cells_y_direction(this->all_param.flow_solver_param.turbulent_channel_number_of_cells_y_direction)
        , number_of_cells_z_direction(this->all_param.flow_solver_param.turbulent_channel_number_of_cells_z_direction)
        , pi_val(3.141592653589793238)
        , domain_length_x(2.0*pi_val*half_channel_height)
        , domain_length_y(2.0*half_channel_height)
        , domain_length_z(pi_val*half_channel_height)
        , channel_bulk_velocity_reynolds_number(pow(0.073, -4.0/7.0)*pow(2.0, 5.0/7.0)*pow(channel_friction_velocity_reynolds_number, 8.0/7.0))
{ }

template <int dim, int nstate>
void ChannelFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int /*current_iteration*/,
        const double /*current_time*/,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> /*unsteady_data_table*/)
{
    // Update maximum local wave speed for adaptive time_step
    this->update_maximum_local_wave_speed(*dg);
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    this->pcout << "- - Reynolds number based on wall friction velocity: " << this->all_param.flow_solver_param.turbulent_channel_friction_velocity_reynolds_number << std::endl;
    this->pcout << "- - Reynolds number based on bulk velocity: " << this->all_param.flow_solver_param.turbulent_channel_bulk_velocity_reynolds_number << std::endl;
    this->pcout << "- - Channel height: " << this->all_param.flow_solver_param.turbulent_channel_height << std::endl;
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::display_grid_parameters() const
{
    const std::string grid_type_string = "subdivided_hyper_rectangle_for_channel_flow";
    // Display the information about the grid
    this->pcout << "- Grid type: " << grid_type_string << std::endl;
    this->pcout << "- - Grid degree: " << this->all_param.flow_solver_param.grid_degree << std::endl;
    this->pcout << "- - Domain dimensionality: " << dim << std::endl;
    this->pcout << "- - Domain length x: " << this->domain_length_x << std::endl;
    this->pcout << "- - Domain length y: " << this->domain_length_y << std::endl;
    this->pcout << "- - Domain length z: " << this->domain_length_z << std::endl;
    this->pcout << "- - Number of cells in x-direction: " << this->number_of_cells_x_direction << std::endl;
    this->pcout << "- - Number of cells in y-direction: " << this->number_of_cells_y_direction << std::endl;
    this->pcout << "- - Number of cells in z-direction: " << this->number_of_cells_z_direction << std::endl;
}

template <int dim, int nstate>
double ChannelFlow<dim,nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> /*dg*/) const
{
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const double cfl_number = this->all_param.flow_solver_param.courant_friedrichs_lewy_number;
    const double time_step = cfl_number * this->minimum_approximate_grid_spacing / this->maximum_local_wave_speed;
    return time_step;
}

template <int dim, int nstate>
double ChannelFlow<dim,nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg)
{
    // initialize the maximum local wave speed
    this->update_maximum_local_wave_speed(*dg);
    // set the minimum approximate grid spacing
    const double minimum_element_size = get_stretched_mesh_size(0)*domain_length_y;
    this->minimum_approximate_grid_spacing = minimum_element_size/double(this->all_param.flow_solver_param.poly_degree+1);
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const double time_step = get_adaptive_time_step(dg);
    return time_step;
}

template <int dim, int nstate>
double ChannelFlow<dim,nstate>::get_stretched_mesh_size(const int i) const
{
    // - Note: This stretching function comes from the structured GMSH .geo file obtained from https://how5.cenaero.be/content/ws2-les-plane-channel-ret550
    const double N_streching_param = 1.0;
    const double r_streching_param = pow(1.2,N_streching_param/2.0);
    const double num_cells_y = (double)number_of_cells_y_direction;
    const double h0_streching_param = 0.5*(1.0-r_streching_param)/(1.0-pow(r_streching_param,(num_cells_y/2.0)));
    return h0_streching_param*pow(r_streching_param,(double)i);
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> ChannelFlow<dim,nstate>::generate_grid() const
{
    // // uncomment this to use the gmsh reader
    // // Dummy triangulation
    // // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
    // const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    // const bool use_mesh_smoothing = false;
    // const int grid_order = 0;
    // std::shared_ptr<HighOrderGrid<dim,double>> mesh = read_gmsh<dim, dim> (mesh_filename, grid_order, use_mesh_smoothing);
    // return mesh->triangulation;

    // define domain to be centered about x and z axis
    // and start domain from y=0 for the convenience of computing wall distance
    const dealii::Point<dim> p1(-0.5*domain_length_x, 0.0, -0.5*domain_length_z);
    const dealii::Point<dim> p2(0.5*domain_length_x, domain_length_y, 0.5*domain_length_z);

    // get step size for each cell
    // - uniform spacing in x and z
    const double uniform_spacing_x = domain_length_x/double(number_of_cells_x_direction);
    const double uniform_spacing_z = domain_length_z/double(number_of_cells_z_direction);
    // - get stretched spacing for y-direction to capture boundary layer
    const int number_of_edges_y_direction = number_of_cells_y_direction+1;
    std::vector<double> element_edges_y_direction(number_of_edges_y_direction);
    // - Note: This stretching function comes from the structured GMSH .geo file obtained from https://how5.cenaero.be/content/ws2-les-plane-channel-ret550
    const double num_cells_y = (double)number_of_cells_y_direction;
    const int max_loop_index = (int)((num_cells_y-2.0)/2.0);
    double h_streching_param = 0.0;
    element_edges_y_direction[0] = h_streching_param;
    for (int i=0; i<max_loop_index; i++) {
        h_streching_param += get_stretched_mesh_size(i);
        element_edges_y_direction[i+1] = h_streching_param;
        element_edges_y_direction[number_of_cells_y_direction-i-1] = 1.0-h_streching_param;
    }
    element_edges_y_direction[(int)(num_cells_y/2.0)] = 0.5;
    element_edges_y_direction[number_of_cells_y_direction] = 1.0;
    // - now multiply these defined for $y\in[0,1]$ by the length of the domain in the y-direction
    for (int j=0; j<number_of_edges_y_direction; j++) {
        element_edges_y_direction[j] *= domain_length_y;
    }
    // - compute the step size in y-direction as the difference between element edges in y-direction
    std::vector<double> step_size_y_direction(number_of_cells_y_direction);
    for (int j=0; j<number_of_cells_y_direction; j++) {
        step_size_y_direction[j] = element_edges_y_direction[j+1] - element_edges_y_direction[j];
    }

    std::vector<std::vector<double> > step_sizes(dim);
    // x-direction
    for (int i=0; i<number_of_cells_x_direction; i++) {
        step_sizes[0].push_back(uniform_spacing_x);
    }
    // y-direction
    for (int j=0; j<number_of_cells_y_direction; j++) {
        step_sizes[1].push_back(step_size_y_direction[j]);
    }
    // z-direction
    for (int k=0; k<number_of_cells_z_direction; k++) {
        step_sizes[2].push_back(uniform_spacing_z);
    }

    // generate grid usign dealii
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation> (this->mpi_communicator);
    const bool colorize = true;
    dealii::GridGenerator::subdivided_hyper_rectangle(*grid, step_sizes, p1, p2, colorize);

    // assign periodic boundary conditions in x and z
    std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator> > matched_pairs;
    dealii::GridTools::collect_periodic_faces(*grid,0,1,0,matched_pairs); // x-direction
    dealii::GridTools::collect_periodic_faces(*grid,4,5,2,matched_pairs); // z-direction
    grid->add_periodicity(matched_pairs);

    // assign wall boundary conditions
    for (typename Triangulation::active_cell_iterator cell = grid->begin_active(); cell != grid->end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        for (unsigned int face=0; face<dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (cell->face(face)->at_boundary()) {
                unsigned int current_id = cell->face(face)->boundary_id();
                if (current_id == 2 || current_id == 3) cell->face(face)->set_boundary_id (1001); // Bottom and top wall
                // could simply introduce different boundary id if using a wall model
            }
        }
    }

    return grid;
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> /*dg*/) const
{
    // // uncomment this to use the gmsh reader
    // const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    // const bool use_mesh_smoothing = false;
    // const int grid_order = this->all_param.flow_solver_param.grid_degree;
    // std::shared_ptr<HighOrderGrid<dim,double>> mesh = read_gmsh<dim, dim> (mesh_filename, grid_order, use_mesh_smoothing);
    // dg->set_high_order_grid(mesh);

    // do nothing if using dealii mesh generator
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::initialize_model_variables(std::shared_ptr<DGBase<dim, double>> dg) const
{
    dg->set_constant_model_variables(
        this->channel_height,
        this->half_channel_height,
        this->channel_friction_velocity_reynolds_number,
        this->channel_bulk_velocity_reynolds_number);
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::update_model_variables(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double integrated_density_over_domain = get_integrated_density_over_domain(*dg);

    dg->set_unsteady_model_variables(
        integrated_density_over_domain,
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