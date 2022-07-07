#include <vector>
#include <iostream>
#include <fstream>

#include <Epetra_RowMatrixTransposer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include "parameters/all_parameters.h"

#include "dg/dg.h"
#include "mesh_error_estimate.h"
#include "functional/functional.h"
#include "physics/physics.h"
#include "linear_solver/linear_solver.h"
#include "post_processor/physics_post_processor.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
MeshErrorEstimateBase<dim, real, MeshType> :: ~MeshErrorEstimateBase(){}

template <int dim, typename real, typename MeshType>
MeshErrorEstimateBase<dim, real, MeshType> :: MeshErrorEstimateBase(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : dg(dg_input)
    {}

template <int dim, typename real, typename MeshType>
ResidualErrorEstimate<dim, real, MeshType> :: ResidualErrorEstimate(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim, real, MeshType> (dg_input)
    {}

template <int dim, typename real, typename MeshType>
dealii::Vector<real> ResidualErrorEstimate<dim, real, MeshType> :: compute_cellwise_errors()
{
    std::vector<dealii::types::global_dof_index> dofs_indices;
    dealii::Vector<real> cellwise_errors (this->dg->high_order_grid->triangulation->n_active_cells());

    for (const auto &cell : this->dg->dof_handler.active_cell_iterators()) 
    {
         if (!cell->is_locally_owned()) 
         continue;

         const int i_fele = cell->active_fe_index();
         const dealii::FESystem<dim,dim> &fe_ref = this->dg->fe_collection[i_fele];
         const unsigned int n_dofs_cell = fe_ref.n_dofs_per_cell();
         dofs_indices.resize(n_dofs_cell);
         cell->get_dof_indices (dofs_indices);
         real max_residual = 0;
         for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) 
         {
            const unsigned int index = dofs_indices[idof];
            const real res = std::abs(this->dg->right_hand_side[index]);
            if (res > max_residual) 
                max_residual = res;
         }
         cellwise_errors[cell->active_cell_index()] = max_residual;
     }

     return cellwise_errors;
}

// constructor
template <int dim, int nstate, typename real, typename MeshType>
DualWeightedResidualError<dim, nstate, real, MeshType>::DualWeightedResidualError(std::shared_ptr< DGBase<dim, real, MeshType> > dg_input)
    : MeshErrorEstimateBase<dim,real,MeshType> (dg_input) 
    , solution_coarse(this->dg->solution)
    , solution_refinement_state(SolutionRefinementStateEnum::coarse)
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    // storing the original FE degree distribution
    coarse_fe_index.reinit(this->dg->triangulation->n_active_cells());

    // create functional
    functional = FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(this->dg->all_parameters->functional_param, this->dg);

    // looping over the cells
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
    {
        if(cell->is_locally_owned())
        {
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
real DualWeightedResidualError<dim, nstate, real, MeshType>::total_dual_weighted_residual_error()
{
    dealii::Vector<real> cellwise_errors = compute_cellwise_errors();
    real error_sum = cellwise_errors.l1_norm();
    return dealii::Utilities::MPI::sum(error_sum, mpi_communicator);
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::compute_cellwise_errors()
{
    dealii::Vector<real> cellwise_errors(this->dg->triangulation->n_active_cells());
    reinit();
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);
    fine_grid_adjoint();
    cellwise_errors = dual_weighted_residual();
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse);

    return cellwise_errors;
}


template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::reinit()
{
    // reinitilizing all variables after triangulation in the constructor
    solution_coarse = this->dg->solution;
    solution_refinement_state = SolutionRefinementStateEnum::coarse;

    // storing the original FE degree distribution
    coarse_fe_index.reinit(this->dg->triangulation->n_active_cells());
    
    // looping over the cells
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
    {
        if(cell->is_locally_owned())
        {
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();
        }
    }

    // for remaining, clearing the values
    derivative_functional_wrt_solution_fine      = dealii::LinearAlgebra::distributed::Vector<real>();
    derivative_functional_wrt_solution_coarse    = dealii::LinearAlgebra::distributed::Vector<real>();
    adjoint_fine   = dealii::LinearAlgebra::distributed::Vector<real>();
    adjoint_coarse = dealii::LinearAlgebra::distributed::Vector<real>();

    dual_weighted_residual_fine = dealii::Vector<real>();
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum required_refinement_state)
{   
    // checks if conversion is needed
    if(solution_refinement_state == required_refinement_state)
    {
        return;
    }
    // calls corresponding function for state conversions
    else if(solution_refinement_state == SolutionRefinementStateEnum::coarse && required_refinement_state == SolutionRefinementStateEnum::fine)
    {
        coarse_to_fine();
    }
    
    else if(solution_refinement_state == SolutionRefinementStateEnum::fine && required_refinement_state == SolutionRefinementStateEnum::coarse)
    {
        fine_to_coarse();
    }
    else
    {
        pcout<<"Invalid state. Aborting.."<<std::endl;
        std::abort();
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::coarse_to_fine()
{
    if (this->dg->get_max_fe_degree() >= this->dg->max_degree) 
    {
        pcout<<"Polynomial degree of DG will exceed the maximum allowable after refinement. Update max_degree in dg"<<std::endl;
        std::abort();
    }

    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs =  this->dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(this->dg->dof_handler, locally_relevant_dofs);

    // dealii::LinearAlgebra::distributed::Vector<double> solution_coarse(this->dg->solution);
    solution_coarse.update_ghost_values();
    
    // Solution Transfer to fine grid
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);

    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    this->dg->triangulation->prepare_coarsening_and_refinement();

    for (auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
    {
        if (cell->is_locally_owned()) 
        {
            cell->set_future_fe_index(cell->active_fe_index()+1);
        }
    }

    this->dg->triangulation->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();

    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>, 
                                 decltype(solution_transfer)>) {
        solution_transfer.interpolate(solution_coarse, this->dg->solution);
    } else {
        solution_transfer.interpolate(this->dg->solution);
    }
    
    this->dg->solution.update_ghost_values();

    solution_refinement_state = SolutionRefinementStateEnum::fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::fine_to_coarse()
{
    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    this->dg->triangulation->prepare_coarsening_and_refinement();

    for (auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
    {
        if (cell->is_locally_owned()) 
        {
            cell->set_future_fe_index(coarse_fe_index[cell->active_cell_index()]);
        }
    }

    this->dg->triangulation->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();

    this->dg->solution = solution_coarse;

    solution_refinement_state = SolutionRefinementStateEnum::coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::fine_grid_adjoint()
{
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);

    adjoint_fine = compute_adjoint(derivative_functional_wrt_solution_fine, adjoint_fine);

    return adjoint_fine;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::coarse_grid_adjoint()
{
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::coarse);

    adjoint_coarse = compute_adjoint(derivative_functional_wrt_solution_coarse, adjoint_coarse);

    return adjoint_coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>
::compute_adjoint(dealii::LinearAlgebra::distributed::Vector<real> &derivative_functional_wrt_solution, 
                  dealii::LinearAlgebra::distributed::Vector<real> &adjoint_variable)
{
    derivative_functional_wrt_solution.reinit(this->dg->solution);
    adjoint_variable.reinit(this->dg->solution);
    
    const bool compute_derivative_functional_wrt_solution = true, compute_derivative_functional_wrt_grid_dofs = false;
    const real functional_value = functional->evaluate_functional(compute_derivative_functional_wrt_solution, compute_derivative_functional_wrt_grid_dofs);
    (void) functional_value;
    derivative_functional_wrt_solution = functional->dIdw;


    this->dg->assemble_residual(true);
    this->dg->system_matrix *= -1.0;

    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
    Epetra_CrsMatrix *system_matrix_transpose_tril;

    Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&this->dg->system_matrix.trilinos_matrix()));
    epmt.CreateTranspose(false, system_matrix_transpose_tril);
    system_matrix_transpose.reinit(*system_matrix_transpose_tril, true);
    solve_linear(system_matrix_transpose, derivative_functional_wrt_solution, adjoint_variable, this->dg->all_parameters->linear_solver_param);

    return adjoint_variable;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> DualWeightedResidualError<dim, nstate, real, MeshType>::dual_weighted_residual()
{
    convert_dgsolution_to_coarse_or_fine(SolutionRefinementStateEnum::fine);

    // allocate 
    dual_weighted_residual_fine.reinit(this->dg->triangulation->n_active_cells());

    const unsigned int max_dofs_per_cell = this->dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);

    // compute the error indicator cell-wise by taking the dot product over the DOFs with the residual vector
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
    {
        if(!cell->is_locally_owned()) 
            continue;
        
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = this->dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        real dwr_cell = 0;
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof)
        {
            dwr_cell += this->dg->right_hand_side[current_dofs_indices[idof]]*adjoint_fine[current_dofs_indices[idof]];
        }

        dual_weighted_residual_fine[cell->active_cell_index()] = std::abs(dwr_cell);
    }

    return dual_weighted_residual_fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void DualWeightedResidualError<dim, nstate, real, MeshType>::output_results_vtk(const unsigned int cycle)
{
    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler(this->dg->dof_handler);

    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(this->dg->all_parameters);
    data_out.add_data_vector(this->dg->solution, *post_processor);

    dealii::Vector<float> subdomain(this->dg->triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) 
    {
        subdomain(i) = this->dg->triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    this->dg->dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector(active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }

    data_out.add_data_vector(this->dg->right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    // set names of data to be output in the vtu file.
    std::vector<std::string> derivative_functional_wrt_solution_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "derivative_functional_wrt_solution" + dealii::Utilities::int_to_string(s,1);
        derivative_functional_wrt_solution_names.push_back(varname);
    }

    std::vector<std::string> adjoint_names;
    for(int s=0;s<nstate;++s) 
    {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1);
        adjoint_names.push_back(varname);
    }

    // add the data structures specific to this class, check if currently fine or coarse
    if(solution_refinement_state == SolutionRefinementStateEnum::fine) {
        data_out.add_data_vector(derivative_functional_wrt_solution_fine, derivative_functional_wrt_solution_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
        data_out.add_data_vector(adjoint_fine, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

        data_out.add_data_vector(dual_weighted_residual_fine, "DWR", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    } else if(solution_refinement_state == SolutionRefinementStateEnum::coarse) {
        data_out.add_data_vector(derivative_functional_wrt_solution_coarse, derivative_functional_wrt_solution_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
        data_out.add_data_vector(adjoint_coarse, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
    }

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    //data_out.build_patches (mapping_collection[mapping_collection.size()-1]);
    data_out.build_patches();
    // data_out.build_patches(*(this->dg->high_order_grid.mapping_fe_field), this->dg->max_degree, dealii::DataOut<dim, dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells);
    //data_out.build_patches(*(high_order_grid.mapping_fe_field), fe_collection.size(), dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
    std::string filename = "adjoint-" ;
    if(solution_refinement_state == SolutionRefinementStateEnum::fine)
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (iproc == 0) 
    {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) 
        {
            std::string fn = "adjoint-";
            if(solution_refinement_state == SolutionRefinementStateEnum::fine)
                fn += "fine-";
            else if(solution_refinement_state == SolutionRefinementStateEnum::coarse)
                fn += "coarse-";
            fn += dealii::Utilities::int_to_string(dim, 1) + "D-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "adjoint-";
        if(solution_refinement_state == SolutionRefinementStateEnum::fine)
            master_fn += "fine-";
        else if(solution_refinement_state == SolutionRefinementStateEnum::coarse)
            master_fn += "coarse-";
        master_fn += dealii::Utilities::int_to_string(dim, 1) +"D-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }
}

template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class MeshErrorEstimateBase<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif


template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM != 1
template class ResidualErrorEstimate<PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

template class DualWeightedResidualError <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class DualWeightedResidualError <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DualWeightedResidualError <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DualWeightedResidualError <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
