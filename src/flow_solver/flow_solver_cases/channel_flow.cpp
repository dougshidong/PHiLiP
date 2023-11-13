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
        , channel_height(this->all_param.flow_solver_param.turbulent_channel_domain_length_y_direction)
        , half_channel_height(channel_height/2.0)
        , channel_friction_velocity_reynolds_number(this->all_param.flow_solver_param.turbulent_channel_friction_velocity_reynolds_number)
        , number_of_cells_x_direction(this->all_param.flow_solver_param.turbulent_channel_number_of_cells_x_direction)
        , number_of_cells_y_direction(this->all_param.flow_solver_param.turbulent_channel_number_of_cells_y_direction)
        , number_of_cells_z_direction(this->all_param.flow_solver_param.turbulent_channel_number_of_cells_z_direction)
        , pi_val(3.141592653589793238)
        , domain_length_x(this->all_param.flow_solver_param.turbulent_channel_domain_length_x_direction)
        , domain_length_y(channel_height)
        , domain_length_z(this->all_param.flow_solver_param.turbulent_channel_domain_length_z_direction)
        , domain_volume(domain_length_x*domain_length_y*domain_length_z)
        , channel_bulk_velocity_reynolds_number(pow(0.073, -4.0/7.0)*pow(2.0, 5.0/7.0)*pow(channel_friction_velocity_reynolds_number, 8.0/7.0))
        , channel_centerline_velocity_reynolds_number(1.28*pow(2.0, -0.0116)*pow(channel_bulk_velocity_reynolds_number,1.0-0.0116))
{
    // initialize zero tensor
    for (int d1=0; d1<dim; ++d1) {
        for (int d2=0; d2<dim; ++d2) {
            zero_tensor[d1][d2] = 0.0;
        }
    }
}

template <int dim, int nstate>
void ChannelFlow<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    // Update maximum local wave speed for adaptive time_step
    this->update_maximum_local_wave_speed(*dg);
    // get averaged wall shear stress
    const double average_wall_shear_stress = get_average_wall_shear_stress(*dg);
    set_bulk_flow_quantities(*dg);
    const double skin_friction_coefficient = get_skin_friction_coefficient_from_average_wall_shear_stress(average_wall_shear_stress);

    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        this->add_value_to_data_table(average_wall_shear_stress,"tau_w",unsteady_data_table);
        this->add_value_to_data_table(skin_friction_coefficient,"skin_friction_coefficient",unsteady_data_table);
        this->add_value_to_data_table(bulk_density,"bulk_density",unsteady_data_table);
        this->add_value_to_data_table(bulk_velocity,"bulk_velocity",unsteady_data_table);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }

    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Cf: " << skin_friction_coefficient
                << "    Ub: " << this->bulk_velocity
                << "    BulkMassFlow: " << this->bulk_mass_flow_rate
                << std::endl;

    // TO DO: print t/2pi and Re_b calculated to track the convergence of the flow; add these to the table

    // Abort if average_wall_shear_stress is nan
    if(std::isnan(average_wall_shear_stress)) {
        this->pcout << " ERROR: Wall shear stress at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }
}

template <int dim, int nstate>
void ChannelFlow<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    if(this->all_param.flow_solver_param.adaptive_time_step)
        this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    else
        this->pcout << "- - Constant time step: " << this->all_param.flow_solver_param.constant_time_step << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    this->pcout << "- - Reynolds number based on wall friction velocity: " << this->channel_friction_velocity_reynolds_number << std::endl;
    this->pcout << "- - Reynolds number based on bulk velocity: " << this->channel_bulk_velocity_reynolds_number << std::endl;
    this->pcout << "- - Reynolds number based on centerline velocity: " << this->channel_centerline_velocity_reynolds_number << std::endl;
    this->pcout << "- - Half channel height: " << this->half_channel_height << std::endl;
    this->display_grid_parameters();
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
    const double minimum_element_size = get_mesh_step_size_y_direction()[0]; // smallest spacing occurs next to wall (i.e. first/last element)
    this->minimum_approximate_grid_spacing = minimum_element_size/double(this->all_param.flow_solver_param.poly_degree+1);
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const double time_step = get_adaptive_time_step(dg);
    return time_step;
}

template <int dim, int nstate>
std::vector<double> ChannelFlow<dim,nstate>::get_mesh_step_size_y_direction() const 
{
    using turbulent_channel_mesh_stretching_function_enum = Parameters::FlowSolverParam::TurbulentChannelMeshStretchingFunctionType;
    const turbulent_channel_mesh_stretching_function_enum turbulent_channel_mesh_stretching_function_type = this->all_param.flow_solver_param.turbulent_channel_mesh_stretching_function_type;
    std::vector<double> step_size_y_direction;
    if(turbulent_channel_mesh_stretching_function_type == turbulent_channel_mesh_stretching_function_enum::gullbrand){
        step_size_y_direction = get_mesh_step_size_y_direction_Gullbrand();
    } else if(turbulent_channel_mesh_stretching_function_type == turbulent_channel_mesh_stretching_function_enum::hopw){
        step_size_y_direction = get_mesh_step_size_y_direction_HOPW();
    } else if(turbulent_channel_mesh_stretching_function_type == turbulent_channel_mesh_stretching_function_enum::carton_de_wiart_et_al){
        step_size_y_direction = get_mesh_step_size_y_direction_carton_de_wiart_et_al();
    } else {
        this->pcout << "ERROR: Invalid turbulent_channel_mesh_stretching_function_type. Aborting..." << std::endl;
        std::abort();
    }
    return step_size_y_direction;
}

template <int dim, int nstate>
std::vector<double> ChannelFlow<dim,nstate>::get_mesh_step_size_y_direction_HOPW() const 
{
    const int number_of_edges_y_direction = number_of_cells_y_direction+1;
    std::vector<double> element_edges_y_direction(number_of_edges_y_direction);
    // - Note: This stretching function comes from the structured GMSH .geo file obtained from https://how5.cenaero.be/content/ws2-les-plane-channel-ret550
    const double N_streching_param = 1.0;
    const double r_streching_param = pow(1.2,N_streching_param/2.0);
    const double num_cells_y = (double)number_of_cells_y_direction;
    const double h0_streching_param = 0.5*(1.0-r_streching_param)/(1.0-pow(r_streching_param,(num_cells_y/2.0)));
    const int max_loop_index = (int)((num_cells_y-2.0)/2.0);
    double h_streching_param = 0.0;
    element_edges_y_direction[0] = h_streching_param;
    for (int i=0; i<max_loop_index; i++) {
        h_streching_param += h0_streching_param*pow(r_streching_param,(double)i);
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
    return step_size_y_direction;
}

template <int dim, int nstate>
std::vector<double> ChannelFlow<dim,nstate>::get_mesh_step_size_y_direction_Gullbrand() const 
{
    // Domain lower bound in y-direction
    const double desired_domain_lower_bound_y = 0.0; // for convenient wall distance calculation
    const double domain_shift = desired_domain_lower_bound_y+1.0; // +1 since domain lower bound from original function is -1
    // - get stretched spacing for y-direction to capture boundary layer
    const int number_of_edges_y_direction = number_of_cells_y_direction+1;
    std::vector<double> element_edges_y_direction(number_of_edges_y_direction);
    /**
     * Reference: Gullbrand, "Grid-independent large-eddy simulation in turbulent channel flow using three-dimensional explicit filtering", 2003.
     **/
    const double num_cells_y = (double)number_of_cells_y_direction;
    const double stretching_parameter = 2.75;
    const double tanh_stretching_parameter = tanh(stretching_parameter);
    for (int j=0; j<number_of_edges_y_direction; j++) {
        element_edges_y_direction[j] = -1.0*tanh(stretching_parameter*(1.0 - 2.0*((double)j)/num_cells_y))/tanh_stretching_parameter;
    }
    // - now apply the domain shift since these are currently defined in the domain $y\in[-1,1]$
    //   (NOTE: this has no affect on the returned vector of step sizes, but is included for completeness)
    for (int j=0; j<number_of_edges_y_direction; j++) {
        element_edges_y_direction[j] += domain_shift;
        element_edges_y_direction[j] /= 2.0;
        element_edges_y_direction[j] *= domain_length_y;
    }
    // - compute the step size in y-direction as the difference between element edges in y-direction
    std::vector<double> step_size_y_direction(number_of_cells_y_direction);
    for (int j=0; j<number_of_cells_y_direction; j++) {
        step_size_y_direction[j] = element_edges_y_direction[j+1] - element_edges_y_direction[j];
    }
    return step_size_y_direction;
}

template <int dim, int nstate>
std::vector<double> ChannelFlow<dim,nstate>::get_mesh_step_size_y_direction_carton_de_wiart_et_al() const 
{
    // - get stretched spacing for y-direction to capture boundary layer
    const int number_of_edges_y_direction = number_of_cells_y_direction+1;
    std::vector<double> element_edges_y_direction(number_of_edges_y_direction);
    /**
     * Reference: C. CARTON DE WIARTET. AL, "Implicit LES of free and wall-bounded turbulent flows based onthe discontinuous Galerkin/symmetric interior penalty method", 2015.
     **/
    const double num_cells_y = (double)number_of_cells_y_direction;
    const double uniform_spacing = domain_length_y/num_cells_y;
    for (int j=0; j<(number_of_cells_y_direction/2+1); j++) {
        element_edges_y_direction[j] = 1.0 - cos(this->pi_val*((double)j)*uniform_spacing/2.0);
        element_edges_y_direction[number_of_cells_y_direction-j] = domain_length_y-element_edges_y_direction[j];
    }
    // - compute the step size in y-direction as the difference between element edges in y-direction
    std::vector<double> step_size_y_direction(number_of_cells_y_direction);
    for (int j=0; j<number_of_cells_y_direction; j++) {
        step_size_y_direction[j] = element_edges_y_direction[j+1] - element_edges_y_direction[j];
    }
    return step_size_y_direction;
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

    // define domain to be centered about x, y, and z axes
    const dealii::Point<dim> p1(-0.5*domain_length_x, -0.5*domain_length_y, -0.5*domain_length_z);
    const dealii::Point<dim> p2(0.5*domain_length_x, 0.5*domain_length_y, 0.5*domain_length_z);

    // get step size for each cell
    // - uniform spacing in x and z
    const double uniform_spacing_x = domain_length_x/double(number_of_cells_x_direction);
    const double uniform_spacing_z = domain_length_z/double(number_of_cells_z_direction);
    // - get stretched spacing for y-direction to capture boundary layer
    std::vector<double> step_size_y_direction = get_mesh_step_size_y_direction();

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

template<int dim, int nstate>
double ChannelFlow<dim, nstate>::get_average_wall_shear_stress(DGBase<dim, double> &dg) const
{
    /// Update flags needed at face points.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;
    double integral_value = 0.0;
    double integral_area_value = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim-1> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEFaceValues<dim,dim> fe_face_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra, 
                                                  face_update_flags);

    
    std::array<double,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_face_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        
        cell->get_dof_indices (dofs_indices);

        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){
                const unsigned int boundary_id = face->boundary_id();
                if(boundary_id==1001){
                    fe_face_values_extra.reinit (cell,iface);
                    const unsigned int n_quad_pts = fe_face_values_extra.n_quadrature_points;
                    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                        std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
                        for (int s=0; s<nstate; ++s) {
                            for (int d=0; d<dim; ++d) {
                                soln_grad_at_q[s][d] = 0.0;
                            }
                        }
                        for (unsigned int idof=0; idof<fe_face_values_extra.dofs_per_cell; ++idof) {
                            const unsigned int istate = fe_face_values_extra.get_fe().system_to_component_index(idof).first;
                            soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_face_values_extra.shape_value_component(idof, iquad, istate);
                            soln_grad_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_face_values_extra.shape_grad_component(idof,iquad,istate);
                        }
                        // const dealii::Point<dim> qpoint = (fe_face_values_extra.quadrature_point(iquad));
                        const dealii::Tensor<1,dim,double> normal_vector = -fe_face_values_extra.normal_vector(iquad); // minus for wall normal from face normal
                        double integrand_value = this->navier_stokes_physics->compute_wall_shear_stress(soln_at_q,soln_grad_at_q,normal_vector);
                        integral_value += integrand_value * fe_face_values_extra.JxW(iquad);
                        integral_area_value += fe_face_values_extra.JxW(iquad);
                    }
                }
            }
        }
    }
    const double mpi_sum_integral_value = dealii::Utilities::MPI::sum(integral_value, this->mpi_communicator);
    const double mpi_sum_integral_area_value = dealii::Utilities::MPI::sum(integral_area_value, this->mpi_communicator);
    const double averaged_value = mpi_sum_integral_value/mpi_sum_integral_area_value;
    return averaged_value;
}

template <int dim, int nstate>
double ChannelFlow<dim, nstate>::get_skin_friction_coefficient_from_average_wall_shear_stress(const double avg_wall_shear_stress) const
{
    // Reference: Reference: Equation 34 of Lodato G, Castonguay P, Jameson A. Discrete filter operators for large-eddy simulation using high-order spectral difference methods. International Journal for Numerical Methods in Fluids2013;72(2):231–258. 
    const double skin_friction_coefficient = 2.0*avg_wall_shear_stress/(this->bulk_density*this->bulk_velocity*this->bulk_velocity);
    return skin_friction_coefficient;
}

template <int dim, int nstate>
void ChannelFlow<dim, nstate>::set_bulk_flow_quantities(DGBase<dim, double> &dg)
{
    const int NUMBER_OF_INTEGRATED_QUANTITIES = 2;
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrated_quantities;
    /// List of possible integrated quantities over the domain
    enum IntegratedQuantitiesEnum {
        bulk_density,
        bulk_mass_flow_rate
    };
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integral_values;
    std::fill(integral_values.begin(), integral_values.end(), 0.0);

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10; // TO DO: could reduce this to reduce computational cost
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

        // double cellwise_integrand_value = 0.0;
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

            std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrand_values;
            std::fill(integrand_values.begin(), integrand_values.end(), 0.0);
            integrand_values[IntegratedQuantitiesEnum::bulk_density] = soln_at_q[0]; // density
            integrand_values[IntegratedQuantitiesEnum::bulk_mass_flow_rate] = soln_at_q[1]; // x-momentum

            // cellwise_integrand_value += integrand_value * fe_values_extra.JxW(iquad);

            for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
                integral_values[i_quantity] += integrand_values[i_quantity] * fe_values_extra.JxW(iquad);
            }
        }
        // // get cell index
        // const dealii::types::global_dof_index cell_index = cell->active_cell_index();
        // const double cellwise_average = cellwise_integrand_value/dg.pde_model_double->cellwise_volume[cell_index];
        // integral_value += cellwise_average;
    }
    // update integrated quantities
    for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
        integrated_quantities[i_quantity] = dealii::Utilities::MPI::sum(integral_values[i_quantity], this->mpi_communicator);
        integrated_quantities[i_quantity] /= this->domain_volume; // divide by total domain volume
    }
    // set the bulk density, mass flow rate, and velocity for the source term used to force the mass flow rate
    this->bulk_density = integrated_quantities[IntegratedQuantitiesEnum::bulk_density];
    this->bulk_mass_flow_rate = integrated_quantities[IntegratedQuantitiesEnum::bulk_mass_flow_rate];
    this->bulk_velocity = this->bulk_mass_flow_rate/this->bulk_density;
}

#if PHILIP_DIM==3
template class ChannelFlow <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace