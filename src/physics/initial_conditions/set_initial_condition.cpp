#include "set_initial_condition.h"
#include "parameters/parameters_flow_solver.h"
#include <deal.II/numerics/vector_tools.h>
#include <string>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
// #include <deal.II/lac/affine_constraints.h>

namespace PHiLiP{

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::set_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > initial_condition_function_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> > dg_input,
        const Parameters::AllParameters *const parameters_input)
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);

    // Set initial condition depending on the method
    using ApplyInitialConditionMethodEnum = Parameters::FlowSolverParam::ApplyInitialConditionMethod;
    const ApplyInitialConditionMethodEnum apply_initial_condition_method = parameters_input->flow_solver_param.apply_initial_condition_method;
    
    pcout << "Initializing solution by " << std::flush;
    if(apply_initial_condition_method == ApplyInitialConditionMethodEnum::interpolate_initial_condition_function) {
        pcout << "interpolating the initial condition function... " << std::flush;
        // for non-curvilinear
        SetInitialCondition<dim,nstate,real>::interpolate_initial_condition(initial_condition_function_input, dg_input);
    } else if(apply_initial_condition_method == ApplyInitialConditionMethodEnum::project_initial_condition_function) {
        pcout << "projecting the initial condition function... " << std::flush;
        // for curvilinear
        SetInitialCondition<dim,nstate,real>::project_initial_condition(initial_condition_function_input, dg_input);
    } else if(apply_initial_condition_method == ApplyInitialConditionMethodEnum::read_values_from_file_and_project) {
        const std::string input_filename_prefix = parameters_input->flow_solver_param.input_flow_setup_filename_prefix;
        pcout << "reading values from file prefix  " << input_filename_prefix << " and projecting... " << std::flush;
        SetInitialCondition<dim,nstate,real>::read_values_from_file_and_project(dg_input,input_filename_prefix);
    }
    pcout << "done." << std::endl;
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::interpolate_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    dealii::VectorTools::interpolate(dg->dof_handler,*initial_condition_function,solution_no_ghost);
    dg->solution = solution_no_ghost;
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::project_initial_condition(
        std::shared_ptr< InitialConditionFunction<dim,nstate,double> > &initial_condition_function,
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg) 
{
    // Commented since this has not yet been tested
    // dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    // solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    // dealii::AffineConstraints affine_constraints(dof_handler.locally_owned_dofs());
    // dealii::VectorTools::project(*(dg->high_order_grid->mapping_fe_field),dg->dof_handler,affine_constraints,dg->volume_quadrature_collection,*initial_condition_function,solution_no_ghost);
    // dg->solution = solution_no_ghost;

    //Note that for curvilinear, can't use dealii interpolate since it doesn't project at the correct order.
    //Thus we interpolate it directly.
    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, 
                                dealii::update_quadrature_points);
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    OPERATOR::vol_projection_operator<dim,2*dim,real> vol_projection(1, dg->max_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
    
        const int i_fele = current_cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection.reinit (current_cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_shape_fns = n_dofs_cell/nstate;
        current_dofs_indices.resize(n_dofs_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for(int istate=0; istate<nstate; istate++){
            std::vector<double> exact_value(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                const dealii::Point<dim> qpoint = (fe_values.quadrature_point(iquad));
                exact_value[iquad] = initial_condition_function->value(qpoint, istate);
            }   
            std::vector<double> sol(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(exact_value, sol, vol_projection.oneD_vol_operator);
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                dg->solution[current_dofs_indices[ishape+istate*n_shape_fns]] = sol[ishape];
            }
        }
    }
}

std::string get_padded_mpi_rank_string(const int mpi_rank_input) {
    // returns the mpi rank as a string with appropriate padding
    std::string mpi_rank_string = std::to_string(mpi_rank_input);
    const unsigned int length_of_mpi_rank_with_padding = 5;
    const int number_of_zeros = length_of_mpi_rank_with_padding - mpi_rank_string.length();
    mpi_rank_string.insert(0, number_of_zeros, '0');

    return mpi_rank_string;
}

template<int dim, int nstate, typename real>
void SetInitialCondition<dim,nstate,real>::read_values_from_file_and_project(
        std::shared_ptr < PHiLiP::DGBase<dim,real> > &dg,
        const std::string input_filename_prefix) 
{
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    
    // (1) Get filename based on MPI rank
    //-------------------------------------------------------------
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    // -- Get padded mpi rank string
    const std::string mpi_rank_string = get_padded_mpi_rank_string(mpi_rank);
    // -- Assemble filename string
    const std::string filename_without_extension = input_filename_prefix + std::string("-") + mpi_rank_string;
    const std::string filename = filename_without_extension + std::string(".dat");
    //-------------------------------------------------------------

    // (2) Read file
    //-------------------------------------------------------------
    std::string line;
    std::string::size_type sz1;

    std::ifstream FILE (filename);
    std::getline(FILE, line); // read first line: DOFs
    
    // check that the file is not empty
    if (line.empty()) {
        pcout << "ERROR: Trying to read empty file named " << filename << std::endl;
        std::abort();
    } else {
        const unsigned int number_of_degrees_of_freedom_per_state_DG = dg->dof_handler.n_dofs()/nstate;
        const unsigned int number_of_degrees_of_freedom_per_state_file = std::stoi(line);
        if(number_of_degrees_of_freedom_per_state_file != number_of_degrees_of_freedom_per_state_DG) {
            pcout << "ERROR: Cannot read initial condition. "
                      << "Number of degrees of freedom per state do not match expected by DG in file: " 
                      << filename << "\n Aborting..." << std::endl;
            std::abort();
        }
    }

    std::getline(FILE, line); // read first line of data

    // check that there indeed is data to be read
    if (line.empty()) {
        pcout << "Error: File has no data to be read" << std::endl;
        std::abort();
    }

    // Commented since this has not yet been tested
    // dealii::LinearAlgebra::distributed::Vector<double> solution_no_ghost;
    // solution_no_ghost.reinit(dg->locally_owned_dofs, MPI_COMM_WORLD);
    // dealii::AffineConstraints affine_constraints(dof_handler.locally_owned_dofs());
    // dealii::VectorTools::project(*(dg->high_order_grid->mapping_fe_field),dg->dof_handler,affine_constraints,dg->volume_quadrature_collection,*initial_condition_function,solution_no_ghost);
    // dg->solution = solution_no_ghost;

    //Note that for curvilinear, can't use dealii interpolate since it doesn't project at the correct order.
    //Thus we interpolate it directly.
    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, 
                                dealii::update_quadrature_points);
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    OPERATOR::vol_projection_operator<dim,2*dim,real> vol_projection(1, dg->max_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[dg->max_degree], dg->oneD_quadrature_collection[dg->max_degree]);
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
    
        const int i_fele = current_cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection.reinit (current_cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        const unsigned int n_shape_fns = n_dofs_cell/nstate;
        current_dofs_indices.resize(n_dofs_cell);
        current_cell->get_dof_indices (current_dofs_indices);
        for(int istate=0; istate<nstate; istate++){
            std::vector<double> exact_value(n_quad_pts);
            for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
                const dealii::Point<dim> qpoint = (fe_values.quadrature_point(iquad));
                
                // -- get point
                dealii::Point<dim> current_point_read_from_file;
                std::string dummy_line = line;
                current_point_read_from_file[0] = std::stod(dummy_line,&sz1);
                for(int i=1; i<dim; ++i) {
                    dummy_line = dummy_line.substr(sz1);
                    sz1 = 0;
                    current_point_read_from_file[i] = std::stod(dummy_line,&sz1);
                }
                if(qpoint.distance(current_point_read_from_file) > 1.0e-14) {
                    pcout << "ERROR: Distance between points is " << qpoint.distance(current_point_read_from_file)
                          << ".\n Aborting..." << std::endl;
                    std::abort();
                }

                // -- get state
                dummy_line = dummy_line.substr(sz1); sz1 = 0; 
                const int current_state_read_from_file = (int) std::stod(dummy_line,&sz1);
                if(istate != current_state_read_from_file) {
                    pcout << "ERROR: Expecting to read state " << istate << " but reading state " 
                          << current_state_read_from_file << ".\n Aborting..." << std::endl;
                    std::abort();
                }
                // -- get initial condition value
                dummy_line = dummy_line.substr(sz1); sz1 = 0; 
                const double initial_condition_value = std::stod(dummy_line,&sz1);

                exact_value[iquad] = initial_condition_value; // store value for projection

                std::getline(FILE, line); // read next line
            }   
            std::vector<double> sol(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(exact_value, sol, vol_projection.oneD_vol_operator);
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                dg->solution[current_dofs_indices[ishape+istate*n_shape_fns]] = sol[ishape];
            }
        }
    }
    if(!line.empty()) {
        pcout << "ERROR: Line is not empty:\n" << line << std::endl;
        pcout << "Aborting..." << std::endl;
    }
}

template class SetInitialCondition<PHILIP_DIM, 1, double>;
template class SetInitialCondition<PHILIP_DIM, 2, double>;
template class SetInitialCondition<PHILIP_DIM, 3, double>;
template class SetInitialCondition<PHILIP_DIM, 4, double>;
template class SetInitialCondition<PHILIP_DIM, 5, double>;
template class SetInitialCondition<PHILIP_DIM, 6, double>;

}//end of namespace PHILIP
