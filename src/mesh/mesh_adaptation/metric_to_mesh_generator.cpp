#include "metric_to_mesh_generator.h"
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <ostream>
#include <fstream>
#include "grid_refinement/gmsh_out.h"
#include  <cstdlib>

namespace PHiLiP {

template<int dim, int nstate, typename real, typename MeshType>
MetricToMeshGenerator<dim, nstate, real, MeshType> :: MetricToMeshGenerator(
    std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> _volume_nodes_mapping,
    std::shared_ptr<MeshType> _triangulation)
    : volume_nodes_mapping(_volume_nodes_mapping)
    , triangulation(_triangulation)
    , dof_handler_vertices(*triangulation)
    , fe_q(1)
    , fe_system(dealii::FESystem<dim>(fe_q,1)) // setting nstate = 1 to index vertices
    , mpi_communicator(MPI_COMM_WORLD)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0)
    , filename("mesh_to_be_generated")
    , filename_pos(filename + ".pos")
    , filename_geo(filename + ".geo")
    , filename_msh(filename + ".msh")
{
    MPI_Comm_rank(mpi_communicator, &mpi_rank);
    MPI_Comm_size(mpi_communicator, &n_mpi);
    
    if(dim != 2)
    {
        pcout<<"This class is currently hardcoded for dim = 2. To be updated later when required."<<std::endl<<std::flush;
        std::abort();
    }

    reinit();
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: reinit()
{
    dof_handler_vertices.initialize(*triangulation, fe_system);
    dof_handler_vertices.distribute_dofs(fe_system);
    dealii::DoFRenumbering::Cuthill_McKee(dof_handler_vertices);
    
    locally_owned_vertex_dofs = dof_handler_vertices.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_vertices, locally_relevant_vertex_dofs);
    ghost_vertex_dofs = locally_relevant_vertex_dofs;
    ghost_vertex_dofs.subtract_set(locally_owned_vertex_dofs);

    const unsigned int n_vertices = dof_handler_vertices.n_dofs();
    
    for(unsigned int i=0; i<dim*dim; ++i)
    {
        optimal_metric_at_vertices[i].reinit(locally_owned_vertex_dofs, ghost_vertex_dofs, mpi_communicator);
    } 

    all_vertices.clear();
    all_vertices.resize(n_vertices);
}
    
template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: interpolate_metric_to_vertices(
    const std::vector<dealii::Tensor<2, dim, real>> &cellwise_optimal_metric)
{
    reinit();
    AssertDimension(cellwise_optimal_metric.size(), triangulation->n_active_cells());

    dealii::Quadrature<dim> quadrature = fe_system.get_unit_support_points();
    dealii::FEValues<dim, dim> fe_values_vertices(*volume_nodes_mapping, fe_system, quadrature, dealii::update_quadrature_points);
    const unsigned int n_dofs_cell = fe_system.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> dof_indices(n_dofs_cell);
    AssertDimension(n_dofs_cell, dealii::GeometryInfo<dim>::vertices_per_cell)

    // Store vertices in each cell with global dof.
    for(const auto &cell: dof_handler_vertices.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        fe_values_vertices.reinit(cell);
        cell->get_dof_indices(dof_indices);

        const std::vector<dealii::Point<dim>> &cell_vertices = fe_values_vertices.get_quadrature_points();

        for(unsigned int idof = 0; idof < cell_vertices.size(); ++idof)
        {
            const unsigned int iquad = idof;
            all_vertices[dof_indices[idof]] = cell_vertices[iquad];
        }
        
        AssertDimension(cell_vertices.size(), dealii::GeometryInfo<dim>::vertices_per_cell); 
        
        // Check if vertices are in the same order.
        for(unsigned int idof = 0; idof < cell_vertices.size(); ++idof)
        {
            const dealii::Point<dim> &ref_vertex = dealii::GeometryInfo<dim>::unit_cell_vertex(idof);
            const dealii::Point<dim> expected_vertex = volume_nodes_mapping->transform_unit_to_real_cell(cell, ref_vertex);
            const real error_in_vertex = expected_vertex.distance(all_vertices[dof_indices[idof]]);
            if(error_in_vertex > 1.0e-10)
            {
                std::cout<<"Expected vertex = "<<expected_vertex<<"     "<<"Actual vertex = "<<all_vertices[dof_indices[idof]]<<std::endl;
                std::cout<<"Error in ordering of vertices. Aborting.."<<std::flush;
                std::abort();
            }   
        }
    } // cell loop ends

//================================ Interpolate metric field ================================================
//  Using Eq. 2.56 in Dolejší, Vít, and Georg May (2022). Anisotropic hp-Mesh Adaptation Methods: Theory, implementation and applications.
    dealii::LinearAlgebra::distributed::Vector<real> metric_count_at_vertices(locally_owned_vertex_dofs, ghost_vertex_dofs, mpi_communicator);
    metric_count_at_vertices = 0;

    for(const auto &cell: dof_handler_vertices.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}

        cell->get_dof_indices(dof_indices);
        const unsigned int cell_index = cell->active_cell_index();
        //std::vector<real> local_ones(n_dofs_cell);
        //std::fill(local_ones.begin(),local_ones.end(),1.0);
        //metric_count_at_vertices.add(dof_indices, local_ones);
        
        for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
        {
            const unsigned int idof_global = dof_indices[idof];
            std::cout<<"Processor# = "<<mpi_rank<<" |  dof_index_global = "<<idof_global<<" |  Vertex = "<<all_vertices[idof_global]<<std::endl;
            metric_count_at_vertices(idof_global) += 1;
            for(unsigned int i=0; i<dim; ++i)
            {
                for(unsigned int j=0; j<dim; ++j)
                {
                    optimal_metric_at_vertices[i*dim + j](idof_global) += cellwise_optimal_metric[cell_index][i][j];
                }
            }
        }
    } // cell loop ends
    
    if((mpi_rank ==2) || (mpi_rank==0) )
    {
        std::cout<<"Metric count at 65 before = "<<metric_count_at_vertices(65)<<"  from processor "<<mpi_rank<<std::endl;
    }

    // Add contribution from all processsors at common idofs_global.
    metric_count_at_vertices.compress(dealii::VectorOperation::add);
    metric_count_at_vertices.update_ghost_values();;
    for(unsigned int i=0; i<dim*dim; ++i)
    {
       optimal_metric_at_vertices[i].compress(dealii::VectorOperation::add);
       optimal_metric_at_vertices[i].update_ghost_values();
    }
    
    if((mpi_rank ==2) || (mpi_rank==0) )
    {
        std::cout<<"Metric count at 65 after = "<<metric_count_at_vertices(65)<<"  from processor "<<mpi_rank<<std::endl;
    }

    // Compute average
    for(unsigned int idof=0; idof<locally_owned_vertex_dofs.n_elements(); ++idof)
    {
        const unsigned int idof_global = locally_owned_vertex_dofs.nth_index_in_set(idof);
        for(unsigned int i=0; i<dim*dim; ++i)
        {
            optimal_metric_at_vertices[i](idof_global) /= metric_count_at_vertices(idof_global);
        }
    }

    // Output optimal metric at vertices for verifying.
    pcout<<"Optimal metric at vertices = "<<std::endl;
    for(const auto &cell : dof_handler_vertices.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        if(mpi_rank != 2) {continue;}
        cell->get_dof_indices(dof_indices);
        
        std::cout<<"Cell index = "<<cell->active_cell_index()<<std::endl;
        std::cout<<"Vertices in this cell = "<<std::endl;
        for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
        {
            const unsigned int idof_global = dof_indices[idof];
            std::cout<<"Vertex = "<<all_vertices[idof_global]<<std::endl;
            std::cout<<"metric count = "<<metric_count_at_vertices[idof_global]<<std::endl;
        }
    
        for(unsigned int idof = 0; idof<n_dofs_cell; ++idof)
        {
            const unsigned int idof_global = dof_indices[idof];
            std::cout<<"Metric at vertex "<<all_vertices[idof_global]<<" : idof_global = "<<idof_global<<std::endl;
        
            for(unsigned int i = 0; i<dim; ++i)
            {
                for(unsigned int j=0; j<dim; ++j)
                {
                    std::cout<<optimal_metric_at_vertices[i*dim+j][idof_global]<<" ";
                }
                std::cout<<std::endl;
            }
        }

        std::cout<<"Metric at cell = "<<std::endl;
        for(unsigned int i = 0; i<dim; ++i)
        {
            for(unsigned int j=0; j<dim; ++j)
            {
                std::cout<<cellwise_optimal_metric[cell->active_cell_index()][i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        
        std::cout<<std::flush<<std::endl;
    
    } // cell loop ends
    MPI_Barrier(mpi_communicator);
    std::abort();
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: write_pos_file() const
{
    AssertDimension(fe_system.dofs_per_cell, dealii::GeometryInfo<dim>::vertices_per_cell);
    
    // Based on gmsh/tutorials/t17_bgmesh.pos in GMSH4.11.1.
    // Adapted from GridRefinement::Gmsh_Out::write_pos_anisotropic() and modified to use metric field at nodes.
    // Might need to figure out a way to link GridRefinement::Gmsh_Out and this class in future.
    const std::string quotes = "\"";
    const std::string filename_this_processor = filename + std::to_string(mpi_rank) + ".pos";
    std::ofstream outfile(filename_this_processor);
    
    //  DEAL.II's default quad numbering: 
    // 2 * * 3
    // *     *
    // *     *
    // 0 * * 1

    // Split into 2 triangles, [0,2,3] and [0,3,1].
    
    // Write .pos file for each triangle using Tensor Triangle (TT) structure.
    // TT(x1,y1,z1,x2,y2,z2,x3,y3,z3){M1, M2, M3};
    // where Mi = m11, m12, m13, m21, m22, m23, m31, m32, m33;
    std::vector<dealii::types::global_dof_index> dof_indices(fe_system.dofs_per_cell);

    for(const auto &cell : dof_handler_vertices.active_cell_iterators())
    {
        if(! cell->is_locally_owned()) {continue;}
        cell->get_dof_indices(dof_indices);
        
        for(const auto &tri :
                    std::array<std::array<int,3>,2>{{
                    {{0,2,3}},
                    {{0,3,1}} }} )
        {
            outfile<<"TT(";
            // Write vertices
            for(unsigned int i = 0; i < tri.size(); ++i)
            {
                if(i != 0) {outfile << ",";}
                
                const unsigned int idof = tri[i];
                const unsigned int idof_global = dof_indices[idof];
                // x
                const dealii::Point<dim> &pos = all_vertices[idof_global];
                if(dim >= 1){outfile << pos[0] << ",";}
                else        {outfile << 0      << ",";}
                // y
                if(dim >= 2){outfile << pos[1] << ",";}
                else        {outfile << 0      << ",";}
                // z
                if(dim >= 3){outfile << pos[2];}
                else        {outfile << 0;}
            }
            outfile<<"){";
            // Write metric
            bool flag = false;
            
            for(unsigned int i = 0; i < tri.size(); ++i)
            {
                const unsigned int idof = tri[i];
                const unsigned int idof_global = dof_indices[idof];
                
                dealii::Tensor<2,dim,real> vertex_metric; 
                for(unsigned int idim =0; idim < dim; ++idim)
                {
                    for(unsigned int jdim =0; jdim < dim; ++jdim)
                    {
                        vertex_metric[idim][jdim] = optimal_metric_at_vertices[idim*dim + jdim][idof_global];
                    }
                }

                for(unsigned int idim =0; idim < 3; ++idim)
                {
                    for(unsigned int jdim =0; jdim < 3; ++jdim)
                    {
                        if(flag) {outfile<<",";}
                        else     {flag = true;}

                        if( (idim < dim) && (jdim < dim) )
                        { 
                            outfile<<vertex_metric[idim][jdim]; 
                        }
                        else
                        {
                            // output 1 along diagonal
                            if(idim == jdim) {outfile<<1.0;}
                            else             {outfile<<0;}
                        }
                    }
                }
            }

            outfile<<"};"<<'\n';

        } //loop on tri : std::array<std::array>> ends
    } // cell loop ends
    outfile<<std::flush;
    outfile.close();
    MPI_Barrier(mpi_communicator);
    
    // Now that all processors have output their files, processor#0 can access and recombine them all in a single file. 
    if(mpi_rank == 0)
    {
        std::ofstream outfile0(filename_pos);

        outfile0<<"// Background mesh containing optimal metric field at nodes."<<'\n'; 
        outfile0<< "View "<<quotes<<"nodalMetric"<<quotes<<" {"<<'\n';

        for(int iproc = 0; iproc < n_mpi; ++iproc)
        {
            const std::string input_filename = filename + std::to_string(iproc) + ".pos"; 
            std::ifstream inputfile (input_filename);
            outfile0<<inputfile.rdbuf();
        }

        outfile0<<"};"<<'\n';
        outfile0.close();
    }

    // Wait for proc#0 to be done.
    MPI_Barrier(mpi_communicator);
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: write_geo_file() const
{
    // Based on gmsh/tutorials/t17.geo in GMSH 4.11.1.
    std::ofstream outfile(filename_geo);
    // Header
    outfile<<" // GEO file "<<'\n'<<'\n';
    const std::string quotes = "\"";

    outfile<<"SetFactory("<<quotes<<"OpenCASCADE"<<quotes<<");"<<'\n'<<'\n';

    // Geo file of rectangle
    GridRefinement::GmshOut<dim, real>::write_geo_hyper_cube(0.0, 1.0, outfile, true);

    // Merge .pos view
    outfile<<"Merge "<<quotes<<filename_pos<<quotes<<";"<<'\n';

    // Apply the view as the current background mesh
    outfile<<"Background Mesh View[0];"<<'\n';
    
    // Use BAMG (Algorithm 7) to generate mesh.
    outfile<<"Mesh.SmoothRatio = 0;"<<'\n'; // No smoothing for now.
    outfile<<"Mesh.AnisoMax = 1e30;"<<'\n';
    outfile<<"Mesh.Algorithm = 7;"<<'\n';

    // Recombine triangles into quads.
    outfile<<"Mesh.RecombinationAlgorithm = 2;"<<'\n';
    //outfile<<"RecombineMesh;"<<'\n';
    outfile<<"Mesh.RecombineAll = 1;"<<'\n';
    //outfile<<"Mesh.RecombinationAlgorithm = 0;"<<'\n';
    //outfile<<"Mesh.SubdivisionAlgorithm = 1;"<<'\n';
    outfile<<std::flush;
    outfile.close();
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: generate_mesh_from_cellwise_metric(
    const std::vector<dealii::Tensor<2, dim, real>> &cellwise_optimal_metric)
{
    interpolate_metric_to_vertices(cellwise_optimal_metric);
    write_pos_file();
    if(mpi_rank == 0)
    {
        write_geo_file();
        // Running this on command line:
        // For 2D: gmsh -dim filename.geo -o filename.msh
        // For 3D: gmsh -dim -algo mmg3d filename.geo -o filename.msh
        std::string command_str = "gmsh -2 " + filename_geo + " -o " + filename_msh;
        int val = std::system(command_str.c_str());
        (void) val;
        pcout<<"Generated new mesh."<<std::endl;
    }
    MPI_Barrier(mpi_communicator);
}

template<int dim, int nstate, typename real, typename MeshType>
std::string MetricToMeshGenerator<dim, nstate, real, MeshType> :: get_generated_mesh_filename() const
{
    return filename_msh;
}

template<int dim, int nstate, typename real, typename MeshType>
void MetricToMeshGenerator<dim, nstate, real, MeshType> :: delete_generated_files() const
{
/*
    std::string command_str = "rm " + filename_geo + " " + filename_pos + " " + filename_msh;
    int val = std::system(command_str.c_str());
    (void) val;
*/
}

// Instantiations
#if PHILIP_DIM!=1
template class MetricToMeshGenerator <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class MetricToMeshGenerator <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif
} // PHiLiP namespace
