#ifndef __METRIC_TO_MESH_GENERATOR_H__
#define __METRIC_TO_MESH_GENERATOR_H__

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/tria.h>

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.templates.h>

#include <deal.II/dofs/dof_handler.h>

namespace PHiLiP {
/// Class to convert metric field to mesh using BAMG.
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class MetricToMeshGenerator {

    using VectorType = dealii::LinearAlgebra::distributed::Vector<double>; ///< Alias for dealii's parallel distributed vector.
    using DoFHandlerType = dealii::DoFHandler<dim>; ///< Alias for declaring DofHandler

public:
    /// Constructor.
    MetricToMeshGenerator(
        std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> _volume_nodes_mapping,
        std::shared_ptr<MeshType> _triangulation);

    /// Destructor
    ~MetricToMeshGenerator() = default;
    
    /// Generates mesh from the input cellwise metric.
    void generate_mesh_from_cellwise_metric(const std::vector<dealii::Tensor<2, dim, real>> &cellwise_optimal_metric);

    /// Returns name of the .msh file
    std::string get_generated_mesh_filename() const;

private:
    /// Reinitialize dof handler vertices after updating triangulation.
    void reinit();
    
    /// Interpolates cellwise metric field to vertices.
    void interpolate_metric_to_vertices(const std::vector<dealii::Tensor<2, dim, real>> &cellwise_optimal_metric);

    /// Writes .geo file
    void write_geo_file() const;

    /// Writes .pos file.
    void write_pos_file() const;
    
    
    /// Mapping field to update physical quadrature points, jacobians etc with the movement of volume nodes.
    std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> volume_nodes_mapping;

    /// Stores triangulation currently in use.
    std::shared_ptr<MeshType> triangulation;
    
    /// DoFHandler to get global index of vertices.
    dealii::DoFHandler<dim> dof_handler_vertices;

    /// Continuous FE of degree 1. 
    const dealii::FE_Q<dim> fe_q;

    /// FESystem for vertices, created with nstate = 1 to relate an entire vertex of size dim by a single dof.
    const dealii::FESystem<dim> fe_system;
    
    /// Stores optimal metric at vertices. Accessed by optimal_metric_at_vertices[position_in_metric_tensor][global_vertex_dof].
    std::array<VectorType, dim*dim> optimal_metric_at_vertices;

    /// Stores all vertices
    std::vector<dealii::Point<dim>> all_vertices;

    /// Alias for MPI_COMM_WORLD
    MPI_Comm mpi_communicator;
    
    /// std::cout only on processor #0.
    dealii::ConditionalOStream pcout;

    /// Processor# of current processor.
    int mpi_rank;

    /// Total no. of processors
    int n_mpi;

    /// Name of the file.
    const std::string filename;
    /// .pos file.
    const std::string filename_pos;
    /// .geo file
    const std::string filename_geo;
    /// .geo file
    const std::string filename_msh;
    
    dealii::IndexSet locally_owned_vertex_dofs; ///< Dofs owned by current processor.
    dealii::IndexSet ghost_vertex_dofs; ///< Dofs not owned by current processor (but it can still access them).
    dealii::IndexSet locally_relevant_vertex_dofs; ///< Union of locally owned degrees of freedom and ghost degrees of freedom.
    
}; // class ends

} // PHiLiP namespace
#endif
