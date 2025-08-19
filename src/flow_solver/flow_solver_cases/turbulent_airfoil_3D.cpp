#include "turbulent_airfoil_3D.h"
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include <deal.II/base/table_handler.h>
#include "mesh/grids/naca_airfoil_grid.hpp"
#include "mesh/gmsh_reader.hpp"
#include "functional/lift_drag.hpp"
#include "functional/extraction_functional.hpp"
#include "functional/amiet_model.hpp"

namespace PHiLiP {
namespace FlowSolver {
//=========================================================
// NACA0012
//=========================================================
template <int dim, int nstate>
Airfoil_3D_LES<dim, nstate>::Airfoil_3D_LES(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : FlowSolverCaseBase<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
{
    
    // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = PDE_enum::navier_stokes;
    this->navier_stokes_physics = std::dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_navier_stokes));

}

template <int dim, int nstate>
void Airfoil_3D_LES<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;
    const PDE_enum pde_type = this->all_param.pde_type;
    if (pde_type == PDE_enum::navier_stokes){
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
    }
    this->pcout << "- - Courant-Friedrichs-Lewy number: " << this->all_param.flow_solver_param.courant_friedrichs_lewy_number << std::endl;
    this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    const double pi = atan(1.0) * 4.0;
    this->pcout << "- - Angle of attack [deg]: " << this->all_param.euler_param.angle_of_attack*180/pi << std::endl;
    this->pcout << "- - Side-slip angle [deg]: " << this->all_param.euler_param.side_slip_angle*180/pi << std::endl;
    this->pcout << "- - Farfield conditions: " << std::endl;
    const dealii::Point<dim> dummy_point;
    for (int s=0;s<nstate;s++) {
        this->pcout << "- - - State " << s << "; Value: " << this->initial_condition_function->value(dummy_point, s) << std::endl;
    }
}

template <int dim, int nstate>
std::shared_ptr<Triangulation> Airfoil_3D_LES<dim,nstate>::generate_grid() const
{
    //Dummy triangulation
    if constexpr(dim==2) {
        std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>(
    #if PHILIP_DIM!=1
                this->mpi_communicator
    #endif
        );
        dealii::GridGenerator::Airfoil::AdditionalData airfoil_data;
        dealii::GridGenerator::Airfoil::create_triangulation(*grid, airfoil_data);
        grid->refine_global();
        return grid;
    } 
    else if constexpr(dim==3) {
        const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
        const bool use_mesh_smoothing = false;
        std::shared_ptr<HighOrderGrid<dim,double>> airfoil_mesh = read_gmsh<dim, dim> (mesh_filename, 
                this->all_param.flow_solver_param.use_periodic_BC_in_x, 
                this->all_param.flow_solver_param.use_periodic_BC_in_y, 
                this->all_param.flow_solver_param.use_periodic_BC_in_z, 
                this->all_param.flow_solver_param.x_periodic_id_face_1, 
                this->all_param.flow_solver_param.x_periodic_id_face_2, 
                this->all_param.flow_solver_param.y_periodic_id_face_1, 
                this->all_param.flow_solver_param.y_periodic_id_face_2, 
                this->all_param.flow_solver_param.z_periodic_id_face_1, 
                this->all_param.flow_solver_param.z_periodic_id_face_2,
                this->all_param.flow_solver_param.mesh_reader_verbose_output, 
                this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
        
        return airfoil_mesh->triangulation;
    }
    
    // TO DO: Avoid reading the mesh twice (here and in set_high_order_grid -- need a default dummy triangulation)
}

template <int dim, int nstate>
void Airfoil_3D_LES<dim,nstate>::set_higher_order_grid(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const std::string mesh_filename = this->all_param.flow_solver_param.input_mesh_filename+std::string(".msh");
    const bool use_mesh_smoothing = false;
    std::shared_ptr<HighOrderGrid<dim,double>> airfoil_mesh = read_gmsh<dim, dim> (mesh_filename, this->all_param.do_renumber_dofs, 0, use_mesh_smoothing);
    dg->set_high_order_grid(airfoil_mesh);
    for (int i=0; i<this->all_param.flow_solver_param.number_of_mesh_refinements; ++i) {
        dg->high_order_grid->refine_global();
    }
}

template <int dim, int nstate>
double Airfoil_3D_LES<dim,nstate>::compute_lift(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> lift_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::lift);
    const double lift = lift_functional.evaluate_functional();
    return lift;
}

template <int dim, int nstate>
double Airfoil_3D_LES<dim,nstate>::compute_drag(std::shared_ptr<DGBase<dim, double>> dg) const
{
    LiftDragFunctional<dim,dim+2,double,Triangulation> drag_functional(dg, LiftDragFunctional<dim,dim+2,double,Triangulation>::Functional_types::drag);
    const double drag = drag_functional.evaluate_functional();
    return drag;
}


std::string get_padded_mpi_rank_strings(const int mpi_rank_input) {
    // returns the mpi rank as a string with appropriate padding
    std::string mpi_rank_string = std::to_string(mpi_rank_input);
    const unsigned int length_of_mpi_rank_with_padding = 5;
    const int number_of_zeros = length_of_mpi_rank_with_padding - mpi_rank_string.length();
    mpi_rank_string.insert(0, number_of_zeros, '0');

    return mpi_rank_string;
}

template <int dim, int nstate>
void Airfoil_3D_LES<dim,nstate>::steady_state_postprocessing(std::shared_ptr<DGBase<dim, double>> dg) const
{
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg);

    this->pcout << " Resulting lift : " << lift << std::endl;
    this->pcout << " Resulting drag : " << drag << std::endl;
}

template <int dim, int nstate>
void Airfoil_3D_LES<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
            const std::shared_ptr<dealii::TableHandler> unsteady_data_table,
            const bool do_write_unsteady_data_table_file)
{
    // Compute aerodynamic values
    const double lift = this->compute_lift(dg);
    const double drag = this->compute_drag(dg); 
    
    if(terminal_counter == 9999){ //only write to terminal every 10000 time steps
        // Print to console
        this->pcout << "    Iter: " << current_iteration
                    << "    Time: " << current_time
                    << "    Lift: " << lift
                    << "    Drag: " << drag;
        this->pcout << std::endl;
        
        terminal_counter = 0;
    }else{
        // Add to counter
        terminal_counter += 1;
    }


    if(this->mpi_rank==0) {
        // Write to file
        if(do_write_unsteady_data_table_file) {
            std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
            unsteady_data_table->write_text(unsteady_data_table_file);
        }
    }
}

template <int dim, int nstate>
void Airfoil_3D_LES<dim, nstate>::compute_time_averaged_solution(
    const std::shared_ptr <ODE::ODESolverBase<dim, double>> ode_solver,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const double time_step)
{
    if((ode_solver->current_time <= this->all_param.flow_solver_param.time_to_start_averaging) && (ode_solver->current_time+time_step > this->all_param.flow_solver_param.time_to_start_averaging)){
        dg->time_averaged_solution =  dg->solution; //First time step, set time-averaged as equal to solution.
    }
    else if(ode_solver->current_time > this->all_param.flow_solver_param.time_to_start_averaging) {
        const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
        auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
            if (!current_cell->is_locally_owned()) continue;

            const int i_fele = current_cell->active_fe_index();
            const unsigned int poly_degree = i_fele;
            const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
            current_dofs_indices.resize(n_dofs_cell);
            current_cell->get_dof_indices (current_dofs_indices);             
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                dg->time_averaged_solution(current_dofs_indices[idof]) = dg->time_averaged_solution(current_dofs_indices[idof]) + (dg->solution(current_dofs_indices[idof]) - dg->time_averaged_solution(current_dofs_indices[idof]))/((ode_solver->current_time - this->all_param.flow_solver_param.time_to_start_averaging + time_step) / time_step); //Incremental average
            }
        }
    }
}

template <int dim, int nstate>
void Airfoil_3D_LES<dim, nstate>::compute_Reynolds_stress(
    const std::shared_ptr <ODE::ODESolverBase<dim, double>> ode_solver,
    const std::shared_ptr <DGBase<dim, double>> dg,
    const double time_step)
{
    if((ode_solver->current_time > this->all_param.flow_solver_param.time_to_start_averaging) && (ode_solver->current_time >= this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress)){
        const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
        std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
        auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
        for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell, ++metric_cell) {
            if (!current_cell->is_locally_owned()) continue;

            const int i_fele = current_cell->active_fe_index();
            const unsigned int poly_degree = i_fele;
            const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;                  
            current_dofs_indices.resize(n_dofs_cell);
            current_cell->get_dof_indices (current_dofs_indices);
            const unsigned int n_shape_fns = n_dofs_cell / nstate;
            dealii::Quadrature<1> vol_quad_equidistant_1D = dealii::QIterated<1>(dealii::QTrapez<1>(),poly_degree);
            const unsigned int n_quad_pts = pow(vol_quad_equidistant_1D.size(),dim);
            const unsigned int init_grid_degree = dg->high_order_grid->fe_system.tensor_degree();
            OPERATOR::basis_functions<dim,2*dim> soln_basis(1, dg->max_degree, init_grid_degree); 
            soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);
            soln_basis.build_1D_gradient_operator(dg->oneD_fe_collection_1state[dg->max_degree], vol_quad_equidistant_1D);                
            // Store solution coeffs for time-averaged flutuating quantitites
            std::array<std::vector<double>,nstate> soln_coeff;
            std::array<std::vector<double>,nstate> time_averaged_soln_coeff;
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                if(ishape == 0) {
                    soln_coeff[istate].resize(n_shape_fns);
                    time_averaged_soln_coeff[istate].resize(n_shape_fns);
                }
                soln_coeff[istate][ishape] = dg->solution(current_dofs_indices[idof]);
                time_averaged_soln_coeff[istate][ishape] = dg->time_averaged_solution(current_dofs_indices[idof]);
            }

            //Project solutin
            std::array<std::vector<double>,nstate> soln_at_q;
            std::array<std::vector<double>,nstate> time_averaged_soln_at_q;
            for(int istate=0; istate<nstate; istate++){
                soln_at_q[istate].resize(n_quad_pts);
                time_averaged_soln_at_q[istate].resize(n_quad_pts);
                // Interpolate soln coeff to volume cubature nodes.
                soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                                soln_basis.oneD_vol_operator);
                // Interpolate soln coeff to volume cubature nodes.
                soln_basis.matrix_vector_mult_1D(time_averaged_soln_coeff[istate], time_averaged_soln_at_q[istate],
                                                soln_basis.oneD_vol_operator);
            }
            // compute quantities at quad nodes (equisdistant)
            dealii::Tensor<1,dim,std::vector<double>> velocity_at_q;
            dealii::Tensor<1,dim,std::vector<double>> time_averaged_velocity_at_q;
            dealii::Tensor<1,dim,std::vector<double>> velocity_fluctuations_at_q;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                std::array<double,nstate> soln_state;
                std::array<double,nstate> time_averaged_soln_state;
                for(int istate=0; istate<nstate; istate++){
                    soln_state[istate] = soln_at_q[istate][iquad];
                    time_averaged_soln_state[istate] = time_averaged_soln_at_q[istate][iquad];
                }
                dealii::Tensor<1,dim,double> vel;// = this->navier_stokes_physics->compute_velocities(soln_state);
                dealii::Tensor<1,dim,double> time_averaged_vel;// = this->navier_stokes_physics->compute_velocities(time_averaged_soln_state);
                const double density = soln_state[0];
                const double time_averaged_density = time_averaged_soln_state[0];
                for (unsigned int d=0; d<dim; ++d) {
                    vel[d] = soln_state[1+d]/density;
                    time_averaged_vel[d] = time_averaged_soln_state[1+d]/time_averaged_density;
                }
                const dealii::Tensor<1,dim,double> velocity = vel;
                const dealii::Tensor<1,dim,double> time_averaged_velocity = time_averaged_vel;
                for(int idim=0; idim<dim; idim++){
                    if(iquad==0){
                        velocity_at_q[idim].resize(n_quad_pts);
                        time_averaged_velocity_at_q[idim].resize(n_quad_pts);
                        velocity_fluctuations_at_q[idim].resize(n_quad_pts);
                    }
                    velocity_at_q[idim][iquad] = velocity[idim];
                    time_averaged_velocity_at_q[idim][iquad] = time_averaged_velocity[idim];
                    velocity_fluctuations_at_q[idim][iquad] = velocity_at_q[idim][iquad] - time_averaged_velocity_at_q[idim][iquad];
                } 
            }
            for(unsigned int idof=0; idof<n_dofs_cell; idof++){
                const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
                const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
                if((ode_solver->current_time <= this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress) && (ode_solver->current_time+time_step > this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress)){
                    //First time step, simply compute Reynolds stress
                    if(istate == 0){//u'v'
                        dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[1][ishape];
                    }else if(istate == 1){//u'u'
                        dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[0][ishape];
                    }else if(istate == 2){//v'v'
                        dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[1][ishape]*velocity_fluctuations_at_q[1][ishape];
                    }else if(istate == 3){//w'w'
                        dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[2][ishape]*velocity_fluctuations_at_q[2][ishape];
                    }else if(istate == 4){//u'w'
                        dg->fluctuating_quantities(current_dofs_indices[idof]) = velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[2][ishape];
                    }
                }else{//Perform time-average
                    if(istate == 0){//<u'v'>
                        dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[1][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress + time_step) / time_step); //Incremental average
                    }else if(istate == 1){//<u'u'>
                        dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[0][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress + time_step) / time_step); //Incremental average
                    }else if(istate == 2){//<v'v'>
                        dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[1][ishape]*velocity_fluctuations_at_q[1][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress + time_step) / time_step); //Incremental average
                    }else if(istate == 3){//<w'w'>
                        dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[2][ishape]*velocity_fluctuations_at_q[2][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress + time_step) / time_step); //Incremental average
                    }else if(istate == 4){//<u'w'>
                        dg->fluctuating_quantities(current_dofs_indices[idof])= dg->fluctuating_quantities(current_dofs_indices[idof]) + (velocity_fluctuations_at_q[0][ishape]*velocity_fluctuations_at_q[2][ishape] - dg->fluctuating_quantities(current_dofs_indices[idof]))/((ode_solver->current_time - this->all_param.flow_solver_param.time_to_start_computing_Reynolds_stress + time_step) / time_step); //Incremental average
                    }
                }
            }
        }
    }
}



#if PHILIP_DIM!=1
    template class Airfoil_3D_LES<PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

