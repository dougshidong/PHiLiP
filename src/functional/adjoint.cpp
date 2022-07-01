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
#include "adjoint.h"
#include "functional.h"
#include "physics/physics.h"
#include "linear_solver/linear_solver.h"
#include "post_processor/physics_post_processor.h"

namespace PHiLiP {

// constructor
template <int dim, int nstate, typename real, typename MeshType>
Adjoint<dim, nstate, real, MeshType>::Adjoint(
    std::shared_ptr< DGBase<dim,real,MeshType> > _dg, 
    std::shared_ptr< Functional<dim, nstate, real, MeshType> > _functional,
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> > _physics):
    dg(_dg),
    functional(_functional),
    physics(_physics),
    triangulation(dg->triangulation),
    solution_coarse(dg->solution),
    adjoint_state(AdjointStateEnum::coarse),
    mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    // storing the original FE degree distribution
    coarse_fe_index.reinit(dg->triangulation->n_active_cells());

    // looping over the cells
    for(auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();
}

// destructor
template <int dim, int nstate, typename real, typename MeshType>
Adjoint<dim, nstate, real, MeshType>::~Adjoint(){}

template <int dim, int nstate, typename real, typename MeshType>
void Adjoint<dim, nstate, real, MeshType>::reinit()
{
    // assuming that all pointers are still valid
    // reinitilizing all variables after triangulation in the constructor
    solution_coarse = dg->solution;
    adjoint_state = AdjointStateEnum::coarse;

    // storing the original FE degree distribution
    coarse_fe_index.reinit(dg->triangulation->n_active_cells());

    // looping over the cells
    for(auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            coarse_fe_index[cell->active_cell_index()] = cell->active_fe_index();

    // for remaining, clearing the values
    dIdw_fine      = dealii::LinearAlgebra::distributed::Vector<real>();
    dIdw_coarse    = dealii::LinearAlgebra::distributed::Vector<real>();
    adjoint_fine   = dealii::LinearAlgebra::distributed::Vector<real>();
    adjoint_coarse = dealii::LinearAlgebra::distributed::Vector<real>();

    dual_weighted_residual_fine = dealii::Vector<real>();
}

template <int dim, int nstate, typename real, typename MeshType>
void Adjoint<dim, nstate, real, MeshType>::convert_to_state(AdjointStateEnum state)
{   
    // checks if conversion is needed
    if(adjoint_state == state) 
        return;

    // then calls corresponding function for state conversions
    if(adjoint_state == AdjointStateEnum::coarse && state == AdjointStateEnum::fine) 
        coarse_to_fine();
    
    if(adjoint_state == AdjointStateEnum::fine && state == AdjointStateEnum::coarse)
        fine_to_coarse();
}

template <int dim, int nstate, typename real, typename MeshType>
void Adjoint<dim, nstate, real, MeshType>::coarse_to_fine()
{
    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs =  dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dg->dof_handler, locally_relevant_dofs);

    // dealii::LinearAlgebra::distributed::Vector<double> solution_coarse(dg->solution);
    solution_coarse.update_ghost_values();
    
    // Solution Transfer to fine grid
    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_coarse);

    dg->high_order_grid->prepare_for_coarsening_and_refinement();
    dg->triangulation->prepare_coarsening_and_refinement();

    for (auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell)
        if (cell->is_locally_owned()) 
            cell->set_future_fe_index(cell->active_fe_index()+1);

    dg->triangulation->execute_coarsening_and_refinement();
    dg->high_order_grid->execute_coarsening_and_refinement();

    dg->allocate_system();
    dg->solution.zero_out_ghosts();

    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>, 
                                 decltype(solution_transfer)>){
        solution_transfer.interpolate(solution_coarse, dg->solution);
    }else{
        solution_transfer.interpolate(dg->solution);
    }
    
    dg->solution.update_ghost_values();

    adjoint_state = AdjointStateEnum::fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void Adjoint<dim, nstate, real, MeshType>::fine_to_coarse()
{
    dg->high_order_grid->prepare_for_coarsening_and_refinement();
    dg->triangulation->prepare_coarsening_and_refinement();

    for (auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell)
        if (cell->is_locally_owned()) 
            cell->set_future_fe_index(coarse_fe_index[cell->active_cell_index()]);

    dg->triangulation->execute_coarsening_and_refinement();
    dg->high_order_grid->execute_coarsening_and_refinement();

    dg->allocate_system();
    dg->solution.zero_out_ghosts();

    dg->solution = solution_coarse;

    adjoint_state = AdjointStateEnum::coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> Adjoint<dim, nstate, real, MeshType>::fine_grid_adjoint()
{
    convert_to_state(AdjointStateEnum::fine);

    // dIdw_fine.reinit(dg->solution);
    // dIdw_fine = functional.evaluate_dIdw(dg, physics);
    const bool compute_dIdW = true, compute_dIdX = false;
    const real functional_value = functional->evaluate_functional(compute_dIdW,compute_dIdX);
    (void) functional_value;
    dIdw_fine = functional->dIdw;

    adjoint_fine.reinit(dg->solution);
    
    dg->assemble_residual(true);
    dg->system_matrix *= -1.0;

    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
    Epetra_CrsMatrix *system_matrix_transpose_tril;

    Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->system_matrix.trilinos_matrix()));
    epmt.CreateTranspose(false, system_matrix_transpose_tril);
    system_matrix_transpose.reinit(*system_matrix_transpose_tril,true);
    delete system_matrix_transpose_tril;
    solve_linear(system_matrix_transpose, dIdw_fine, adjoint_fine, dg->all_parameters->linear_solver_param);
    // solve_linear(dg.system_matrix, dIdw_fine, adjoint_fine, dg.all_parameters->linear_solver_param);

    return adjoint_fine;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> Adjoint<dim, nstate, real, MeshType>::coarse_grid_adjoint()
{
    convert_to_state(AdjointStateEnum::coarse);

    dIdw_coarse.reinit(dg->solution);
    //dIdw_coarse = functional.evaluate_dIdw(dg, physics);
    const bool compute_dIdW = true, compute_dIdX = false;
    const real functional_value = functional->evaluate_functional(compute_dIdW,compute_dIdX);
    (void) functional_value;
    dIdw_coarse = functional->dIdw;

    adjoint_coarse.reinit(dg->solution);

    dg->assemble_residual(true);
    dg->system_matrix *= -1.0;

    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;
    Epetra_CrsMatrix *system_matrix_transpose_tril;

    Epetra_RowMatrixTransposer epmt(const_cast<Epetra_CrsMatrix *>(&dg->system_matrix.trilinos_matrix()));
    epmt.CreateTranspose(false, system_matrix_transpose_tril);
    system_matrix_transpose.reinit(*system_matrix_transpose_tril);
    solve_linear(system_matrix_transpose, dIdw_coarse, adjoint_coarse, dg->all_parameters->linear_solver_param);
    // solve_linear(dg->system_matrix, dIdw_coarse, adjoint_coarse, dg->all_parameters->linear_solver_param);

    return adjoint_coarse;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::Vector<real> Adjoint<dim, nstate, real, MeshType>::dual_weighted_residual()
{
    convert_to_state(AdjointStateEnum::fine);

    // allocating 
    dual_weighted_residual_fine.reinit(dg->triangulation->n_active_cells());

    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);

    // computing the error indicator cell-wise by taking the dot product over the DOFs with the residual vector
    for(auto cell = dg->dof_handler.begin_active(); cell != dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;
        
        const unsigned int fe_index_curr_cell = cell->active_fe_index();
        const dealii::FESystem<dim,dim> &current_fe_ref = dg->fe_collection[fe_index_curr_cell];
        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();

        current_dofs_indices.resize(n_dofs_curr_cell);
        cell->get_dof_indices(current_dofs_indices);

        real dwr_cell = 0;
        for(unsigned int idof = 0; idof < n_dofs_curr_cell; ++idof){
            dwr_cell += dg->right_hand_side[current_dofs_indices[idof]]*adjoint_fine[current_dofs_indices[idof]];
        }

        dual_weighted_residual_fine[cell->active_cell_index()] = std::abs(dwr_cell);
    }

    return dual_weighted_residual_fine;
}

template <int dim, int nstate, typename real, typename MeshType>
void Adjoint<dim, nstate, real, MeshType>::output_results_vtk(const unsigned int cycle)
{
    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler(dg->dof_handler);

    const std::unique_ptr< dealii::DataPostprocessor<dim> > post_processor = Postprocess::PostprocessorFactory<dim>::create_Postprocessor(dg->all_parameters);
    data_out.add_data_vector(dg->solution, *post_processor);

    dealii::Vector<float> subdomain(dg->triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain(i) = dg->triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output the polynomial degree in each cell
    std::vector<unsigned int> active_fe_indices;
    dg->dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    dealii::Vector<double> cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector(active_fe_indices_dealiivector, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    std::vector<std::string> residual_names;
    for(int s=0;s<nstate;++s) {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }

    data_out.add_data_vector(dg->right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    // setting up the naming
    std::vector<std::string> dIdw_names;
    for(int s=0;s<nstate;++s) {
        std::string varname = "dIdw" + dealii::Utilities::int_to_string(s,1);
        dIdw_names.push_back(varname);
    }

    std::vector<std::string> adjoint_names;
    for(int s=0;s<nstate;++s) {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1);
        adjoint_names.push_back(varname);
    }

    // adding the data structures specific to this particular class, checking if currently fine or coarse
    if(adjoint_state == AdjointStateEnum::fine){
        data_out.add_data_vector(dIdw_fine, dIdw_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
        data_out.add_data_vector(adjoint_fine, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

        data_out.add_data_vector(dual_weighted_residual_fine, "DWR", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    }else if(adjoint_state == AdjointStateEnum::coarse){
        data_out.add_data_vector(dIdw_coarse, dIdw_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
        data_out.add_data_vector(adjoint_coarse, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
    }

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    //data_out.build_patches (mapping_collection[mapping_collection.size()-1]);
    data_out.build_patches();
    // data_out.build_patches(*(dg->high_order_grid.mapping_fe_field), dg->max_degree, dealii::DataOut<dim, dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells);
    //data_out.build_patches(*(high_order_grid.mapping_fe_field), fe_collection.size(), dealii::DataOut<dim>::CurvedCellRegion::curved_inner_cells);
    std::string filename = "adjoint-" ;
    if(adjoint_state == AdjointStateEnum::fine)
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            std::string fn = "adjoint-";
            if(adjoint_state == AdjointStateEnum::fine)
                fn += "fine-";
            else if(adjoint_state == AdjointStateEnum::coarse)
                fn += "coarse-";
            fn += dealii::Utilities::int_to_string(dim, 1) + "D-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "adjoint-";
        if(adjoint_state == AdjointStateEnum::fine)
            master_fn += "fine-";
        else if(adjoint_state == AdjointStateEnum::coarse)
            master_fn += "coarse-";
        master_fn += dealii::Utilities::int_to_string(dim, 1) +"D-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }
}

template class Adjoint <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class Adjoint <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class Adjoint <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Adjoint <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
