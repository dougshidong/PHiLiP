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

#include "dg/dg_base.hpp"
#include "acoustic_adjoint.hpp"
#include "functional.h"
#include "physics/physics.h"
#include "linear_solver/linear_solver.h"
#include "post_processor/physics_post_processor.h"
#include "mesh/meshmover_linear_elasticity.hpp"

#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "optimization/design_parameterization/ffd_parameterization.hpp"

namespace PHiLiP {

//================================================================
// Acoustic adjoint
//================================================================
template <int dim, int nstate, typename real, typename MeshType>
AcousticAdjoint<dim, nstate, real, MeshType>::AcousticAdjoint(
    std::shared_ptr< DGBase<dim,real,MeshType> > dg_input,
    std::shared_ptr< Functional<dim, nstate, real, MeshType>> functional_input):
    dg(dg_input),
    functional(functional_input),
    triangulation(dg->triangulation),
    mpi_communicator(MPI_COMM_WORLD),
    pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
{
    this->dIdw = functional->dIdw;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
AcousticAdjoint<dim, nstate, real, MeshType>::~AcousticAdjoint(){}
//----------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
void AcousticAdjoint<dim, nstate, real, MeshType>::compute_adjoint()
{
    this->adjoint.reinit(dg->solution);
    this->dg->assemble_residual(true,false,false);

    AssertDimension(this->functional->dIdw.size(), this->adjoint.size());
    AssertDimension(this->dg->system_matrix_transpose.n(), this->adjoint.size());

    solve_linear(this->dg->system_matrix_transpose, this->functional->dIdw, this->adjoint, this->dg->all_parameters->linear_solver_param);
    this->adjoint *= -1.0;

    this->adjoint.compress(dealii::VectorOperation::add);
    this->adjoint.update_ghost_values();
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
void AcousticAdjoint<dim, nstate, real, MeshType>::compute_dIdXv()
{
    this->dg->assemble_residual(false,true,false);

    this->dIdXv = this->functional->dIdX;

    this->dg->dRdXv.Tvmult(this->dIdXv,this->adjoint);

    this->dIdXv.compress(dealii::VectorOperation::add);
    this->dIdXv.update_ghost_values();
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
void AcousticAdjoint<dim, nstate, real, MeshType>::compute_dXvdXs(std::shared_ptr<HighOrderGrid<dim,real>> high_order_grid)
{
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    dealii::LinearAlgebra::distributed::Vector<double> surface_node_displacements(high_order_grid->surface_nodes);
    // MeshMover::LinearElasticity<dim, double> 
    //MeshMover::LinearElasticity<dim, double, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>> 
        // meshmover( 
        //   *(high_order_grid->triangulation),
        //   high_order_grid->initial_mapping_fe_field,
        //   high_order_grid->dof_handler_grid,
        //   high_order_grid->surface_to_volume_indices,
        //   surface_node_displacements);
    VectorType dIdXs = high_order_grid->surface_nodes;
    MeshMover::LinearElasticity<dim, double> meshmover(*high_order_grid, dIdXs);

    // dealii::TrilinosWrappers::SparseMatrix dXvdXs_matrix;     
    // dealii::LinearAlgebra::distributed::Vector<double> dXvdXs_matrix;
    meshmover.evaluate_dXvdXs();      
    // this->dXvdXs = this->meshmover->dXvdXs;
    //evaluate_dXvdXs will store derivative in dXvdXs_matrix
    //this->dXvdXs = meshmover.dXvdXs_matrix;
}
//---------------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
void AcousticAdjoint<dim, nstate, real, MeshType>::compute_dXsdXd(std::shared_ptr<HighOrderGrid<dim,real>> high_order_grid){


//create ffd box
    const dealii::Point<dim> ffd_origin(-0.1,-0.1);
    const std::array<double,dim> ffd_rectangle_lengths = {{0.6,0.2}};
    const std::array<unsigned int,dim> ffd_ndim_control_pts = {{30,10}};
    FreeFormDeformation<dim> ffd(ffd_origin, ffd_rectangle_lengths, ffd_ndim_control_pts);

    unsigned int n_design_variables = 0;
    // Vector of ijk indices and dimension.
    // Each entry in the vector points to a design variable's ijk ctl point and its acting dimension.
    std::vector< std::pair< unsigned int, unsigned int > > ffd_design_variables_indices_dim;
    for (unsigned int i_ctl = 0; i_ctl < ffd.n_control_pts; ++i_ctl) {

        const std::array<unsigned int,dim> ijk = ffd.global_to_grid ( i_ctl );
        for (unsigned int d_ffd = 0; d_ffd < dim; ++d_ffd) {

            if (   ijk[0] == 0 // Constrain first column of FFD points.
                || ijk[0] == ffd_ndim_control_pts[0] - 1  // Constrain last column of FFD points.
                || ijk[1] == 1 // Constrain middle row of FFD points.
                || d_ffd == 0 // Constrain x-direction of FFD points.
               ) {
                continue;
            }
            ++n_design_variables;
            ffd_design_variables_indices_dim.push_back(std::make_pair(i_ctl, d_ffd));
        }
    }

    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    dealii::LinearAlgebra::distributed::Vector<double> ffd_design_variables(row_part,ghost_row_part,MPI_COMM_WORLD);

    ffd.get_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_design_variables);
 
    ffd.get_dXvsdXp(*high_order_grid, ffd_design_variables_indices_dim, this->dXsdXd);
    //initializing dIdXd
    this->dIdXd.reinit(ffd_design_variables);

}
//--------------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
void AcousticAdjoint<dim, nstate, real, MeshType>::compute_dIdXd(std::shared_ptr<HighOrderGrid<dim,real>> high_order_grid){
    
    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;
    // computing dXv_dXs
    VectorType surface_node_displacements(high_order_grid->surface_nodes);
    this->dIdXs = high_order_grid->surface_nodes;
    MeshMover::LinearElasticity<dim, double> meshmover(*high_order_grid, this->dIdXs);
    meshmover.evaluate_dXvdXs();    
    this->pcout << "dXvdXs computed" << std::endl;

    // assembling dI_dXs
    meshmover.dXvdXs_matrix.Tvmult(this->dIdXs,this->dIdXv);
    this->pcout << "dIdXs computed" << std::endl;

    // computing dXs_dXd
    compute_dXsdXd(high_order_grid);
    this->pcout << "dXsdXd computed" << std::endl;
    // assembling dI_dXd
    this->dXsdXd.Tvmult(this->dIdXd,this->dIdXs);
    this->pcout << "dIdXd computed" << std::endl;

    this->dIdXs.compress(dealii::VectorOperation::add);
    this->dIdXs.update_ghost_values();

    this->dIdXd.compress(dealii::VectorOperation::add);
    this->dIdXd.update_ghost_values();
}
//---------------------------------------------------------------------
template <int dim, int nstate, typename real, typename MeshType>
void AcousticAdjoint<dim, nstate, real, MeshType>::output_results_vtk(const unsigned int cycle)
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

    data_out.add_data_vector(dIdw, dIdw_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
    data_out.add_data_vector(dIdXv, "dIdXv", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    data_out.add_data_vector(dIdXs, "dIdXs", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
     this->pcout << "dIdXs outputted" << std::endl;
    data_out.add_data_vector(dIdXd, "dIdXd", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
     this->pcout << "dIdXd outputted" << std::endl;
    data_out.add_data_vector(adjoint, adjoint_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    data_out.build_patches();
    std::string filename = "acoustic_adjoint-" ;
    filename += dealii::Utilities::int_to_string(cycle, 4) + ".";
    filename += dealii::Utilities::int_to_string(iproc, 4);
    filename += ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    if (iproc == 0) {
        std::vector<std::string> filenames;
        for (unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc) {
            std::string fn = "acoustic_adjoint-";
            fn += dealii::Utilities::int_to_string(cycle, 4) + ".";
            fn += dealii::Utilities::int_to_string(iproc, 4);
            fn += ".vtu";
            filenames.push_back(fn);
        }
        std::string master_fn = "acoustic_adjoint-";
        master_fn += dealii::Utilities::int_to_string(cycle, 4) + ".pvtu";
        std::ofstream master_output(master_fn);
        data_out.write_pvtu_record(master_output, filenames);
    }
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- AcousticAdjoint
template class AcousticAdjoint <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class AcousticAdjoint <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class AcousticAdjoint <PHILIP_DIM, PHILIP_DIM+2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class AcousticAdjoint <PHILIP_DIM, PHILIP_DIM+3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace