#ifndef __HIGHORDERGRID_H__
#define __HIGHORDERGRID_H__

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/la_parallel_vector.templates.h>

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include "parameters/all_parameters.h"

namespace PHiLiP {

// determines the associated typenames for HO grid from the meshType
template <typename MeshType>
class MeshTypeHelper{};

// specializations
template <>
class MeshTypeHelper<dealii::Triangulation<PHILIP_DIM>>
{
public:
    using VectorType     = dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>;

    template <int dim = PHILIP_DIM, typename Vector = VectorType, typename DoFHandler = DoFHandlerType>
    using SolutionTransfer = dealii::SolutionTransfer<dim, Vector, DoFHandler>;

    // reinitialize vector based on parralel Dofhandler
    static void reinit_vector(
        VectorType             &vector, 
        DoFHandlerType const   &/* dof_handler */,
        dealii::IndexSet const &locally_owned_dofs,
        dealii::IndexSet const &ghost_dofs,
        MPI_Comm const          mpi_communicator)
    {
        vector.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    }
};

template <>
class MeshTypeHelper<dealii::parallel::distributed::Triangulation<PHILIP_DIM>>
{
public:
    using VectorType     = dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>;
    
    template <int dim = PHILIP_DIM, typename Vector = VectorType, typename DoFHandler = DoFHandlerType>
    using SolutionTransfer = dealii::parallel::distributed::SolutionTransfer<dim, Vector, DoFHandler>;

    // reinitialize vector based on parralel Dofhandler
    static void reinit_vector(
        VectorType             &vector, 
        DoFHandlerType const   &/* dof_handler */,
        dealii::IndexSet const &locally_owned_dofs,
        dealii::IndexSet const &ghost_dofs,
        MPI_Comm const          mpi_communicator)
    {
        vector.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    }
};

#if PHILIP_DIM1!=1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <>
class MeshTypeHelper<dealii::parallel::shared::Triangulation<PHILIP_DIM>>
{
public:
    using VectorType     = dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>;

    template <int dim = PHILIP_DIM, typename Vector = VectorType, typename DoFHandler = DoFHandlerType>
    using SolutionTransfer = dealii::SolutionTransfer<dim, Vector, DoFHandler>;

    // reinitialize vector based on parralel Dofhandler
    static void reinit_vector(
        VectorType             &vector, 
        DoFHandlerType const   &/* dof_handler */,
        dealii::IndexSet const &locally_owned_dofs,
        dealii::IndexSet const &ghost_dofs,
        MPI_Comm const          mpi_communicator)
    {
        vector.reinit(locally_owned_dofs, ghost_dofs, mpi_communicator);
    }
};
#endif

/** This HighOrderGrid class basically contains all the different part necessary to generate
 *  a dealii::MappingFEField that corresponds to the current Triangulation and attached Manifold.
 *  Once the high order grid is generated, the mesh can be deformed by assigning different values to the
 *  volume_nodes vector. The dof_handler_grid is used to access and loop through those volume_nodes.
 *  This will especially be useful when performing shape optimization, where the surface and volume
 *  volume_nodes will need to be displaced.
 *  Note that there are a lot of pre-processor statements, and that is because the SolutionTransfer class
 *  and the Vector class act quite differently between serial and parallel implementation. Hopefully,
 *  deal.II will change this one day such that we have one interface for both.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int      dim            = PHILIP_DIM, 
          typename real           = double, 
          typename MeshType       = dealii::Triangulation<dim>, 
          typename VectorType     = typename MeshTypeHelper<MeshType>::VectorType, 
          typename DoFHandlerType = typename MeshTypeHelper<MeshType>::DoFHandlerType>
#else
template <int      dim            = PHILIP_DIM, 
          typename real           = double, 
          typename MeshType       = dealii::parallel::distributed::Triangulation<dim>, 
          typename VectorType     = typename MeshTypeHelper<MeshType>::VectorType, 
          typename DoFHandlerType = typename MeshTypeHelper<MeshType>::DoFHandlerType>
#endif
class HighOrderGrid
{
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;
public:
    /// Principal constructor that will call delegated constructor.
    HighOrderGrid(
        const unsigned int              max_degree, 
        const std::shared_ptr<MeshType> triangulation_input);

    /// Reinitialize high_order_grid after a change in triangulation
    void reinit();

    /// Update the MappingFEField
    /** Note that this rarely needs to be called since MappingFEField stores a
     *  pointer to the DoFHandler and to the node Vector.
     */
    void update_mapping_fe_field();

    /// Ensures that hanging nodes are updated for a conforming mesh.
    void ensure_conforming_mesh();

    /// Sets the volume_nodes to the interpolated position of the Manifold associated to the triangulation.
    void initialize_with_triangulation_manifold(const bool output_mesh = false);

    /// Needed to allocate the correct number of volume_nodes when initializing and after the mesh is refined
    void allocate();

    /// Return a MappingFEField that corresponds to the current node locations
    dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> get_MappingFEField();

    /// Maximum degree of the geometry polynomial representing the grid.
    const unsigned int max_degree;

    const std::shared_ptr<MeshType> triangulation; ///< Mesh

    /// Degrees of freedom handler for the high-order grid
    dealii::DoFHandler<dim> dof_handler_grid;

    /// Current nodal coefficients of the high-order grid.
    /** Note that this contains all \<dim\> directions.
     *  By convention, the DoF representing the z-direction follows the DoF representing
     *  the y-direction, which follows the one representing the x-direction such that
     *  the integer division "idof_index / dim" gives the coordinates related to the same
     *  point.
     */
    VectorType volume_nodes;


    /** Distributed ghosted vector of surface nodes.
     */
    VectorType surface_nodes;

    /// Initial nodal coefficients of the high-order grid.
    /** Used for deformation schemes that always deform the mesh from the initial mesh.
     */
    VectorType initial_volume_nodes;
    /// Distributed ghosted vector of initial surface nodes.
    /** Used for deformation schemes that always deform the mesh from the initial mesh.
     */
    VectorType initial_surface_nodes;

    /// Sets the initial_volume_nodes and initial_surface_nodes to the current volume_nodes and surface_nodes.
    void reset_initial_nodes();

    /** Distributed ghosted vector of surface indices.
     *  Ordering matches the surface_nodes.
     */
    dealii::LinearAlgebra::distributed::Vector<int> surface_to_volume_indices;

    /** Maps a vector of surface_nodes to a vector of volume_nodes.
     *  This is equivalent to using surface_to_volume_indices to assign the surface_nodes values
     *  into a vector of size volume_nodes.size(), except that it uses a matrix-vector multiplication
     *  to easily obtain the mapping.
     */
    dealii::TrilinosWrappers::SparseMatrix map_nodes_surf_to_vol;

    /// Sets the initial_volume_nodes and initial_surface_nodes to the current volume_nodes and surface_nodes.
    void update_map_nodes_surf_to_vol();

    /** Locally owned surface nodes dealii::IndexSet
     */
    dealii::IndexSet locally_owned_surface_nodes_indexset;
    /** Ghost surface nodes dealii::IndexSet
     */
    dealii::IndexSet ghost_surface_nodes_indexset;

    /** Number of locally_relevant_surface_nodes per process */
    std::vector<unsigned int> n_locally_owned_surface_nodes_per_mpi;

    /// List of surface nodes.
    /** Note that this contains all \<dim\> directions.
     *  By convention, the DoF representing the z-direction follows the DoF representing
     *  the y-direction, which follows the one representing the x-direction such that
     *  the integer division "idof_index / dim" gives the coordinates related to the same
     *  point.
     */
    std::vector<real> locally_relevant_surface_nodes;

    /// List of surface node indices
    std::vector<dealii::types::global_dof_index> locally_relevant_surface_nodes_indices;
    /// List of surface node boundary IDs, corresponding to locally_relevant_surface_nodes_indices
    std::vector<dealii::types::global_dof_index> locally_relevant_surface_nodes_boundary_id;

    /// List of surface nodes.
    /** Note that this contains all \<dim\> directions.
     *  By convention, the DoF representing the z-direction follows the DoF representing
     *  the y-direction, which follows the one representing the x-direction such that
     *  the integer division "idof_index / dim" gives the coordinates related to the same
     *  point.
     */
    std::vector<real> locally_owned_surface_nodes;
    /// List of all surface nodes
    /** Same as locally_owned_surface_nodes except that it stores a global vector of all the
     *  surface nodes that will be needed to evaluate the A matrix in the RBF
     *  deformation dxv = A * coeff = A * (Minv*dxs)
     */
    std::vector<real> all_surface_nodes;
    /** List of global surface node indices including those across different processors.
     *  Ordering corresponds to all_surface_nodes.
     *  TODO: Might want to make this a std::pair.
     */
    std::vector<dealii::types::global_dof_index> all_surface_indices;

    /// List of surface node indices
    std::vector<dealii::types::global_dof_index> locally_owned_surface_nodes_indices;

    // /// List of cells associated with locally_relevant_surface_nodes_indices
    // std::vector<dealii::types::global_dof_index> locally_relevant_surface_nodes_cells;
    // /// List of points associated with locally_relevant_surface_nodes_indices
    // std::vector<dealii::types::global_dof_index> locally_relevant_surface_nodes_points;
    // /// List of direction associated with locally_relevant_surface_nodes_indices
    // std::vector<dealii::types::global_dof_index> locally_relevant_surface_nodes_direction;

    /// Locally relevant surface points.
    /** Might be useful if the transformation operates on a Point.
     *  TODO: However, might want to remove this and use the surface_nodes vector
     *  for consistency and create a function to retrieve the Point.
     */
    std::vector<dealii::Point<dim>> locally_relevant_surface_points;

    /// Initial locally relevant surface points.
    /** In the optimization framework, we often displace the mesh from the initial mesh.
     */
    std::vector<dealii::Point<dim>> initial_locally_relevant_surface_points;

    /** Given a global DoF index, this will return the Point index
     *  within the locally_relevant_surface_points and its component.
     *  This is the inverse map of point_and_axis_to_global_index.
     */
    std::map<dealii::types::global_dof_index, std::pair<unsigned int, unsigned int>> global_index_to_point_and_axis;
    /** Given the Point index within the locally_relevant_surface_points and its component,
     *  this will return the global DoF index.
     *  This is the inverse map of global_index_to_point_and_axis.
     */
    std::map<std::pair<unsigned int, unsigned int>, dealii::types::global_dof_index> point_and_axis_to_global_index;

    // /** Given the Point index within the locally_relevant_surface_points and its component,
    //  *  this will return the index of the surface_nodes' index.
    //  *  This allows us to obtain a deformation vector matching the locally_relevant_surface_points, and then copy
    //  *  its results to the surface_nodes.
    //  */
    // std::map<std::pair<unsigned int, unsigned int>, dealii::types::global_dof_index> point_and_axis_to_surface_nodes_index;



    /// Update list of surface nodes (all_locally_relevant_surface_nodes).
    void update_surface_nodes();

    /** Transforms the surface_nodes vector using a std::function tranformation.
     */
    VectorType transform_surface_nodes(std::function<dealii::Point<dim>(dealii::Point<dim>)> transformation) const;

    /// RBF mesh deformation  -  To be done
    //void deform_mesh(VectorType surface_displacements);
    void deform_mesh(std::vector<real> local_surface_displacements);

    void test_jacobian(); ///< Test metric Jacobian

    /// Evaluates Jacobian of a cell given some points using a global solution vector
    std::vector<real> evaluate_jacobian_at_points(
        const VectorType &solution,
        const typename DoFHandlerType::cell_iterator &cell,
        const std::vector<dealii::Point<dim>> &points) const;

    /// Evaluates Jacobian given some DoF, associated FE, and some points.
    /** Calls evaluate_jacobian_at_point() on the vector of points.
     *  Can be used in conjunction with AD since the type is templated.
     */
    template <typename real2>
    void evaluate_jacobian_at_points(
        const std::vector<real2> &dofs,
        const dealii::FESystem<dim> &fe,
        const std::vector<dealii::Point<dim>> &points,
        std::vector<real2> &jacobian_determinants) const;
    /// Evaluates Jacobian given some DoF, associated FE, and some point.
    /** Can be used in conjunction with AD since the type is templated.
     */
    template <typename real2>
    real2 evaluate_jacobian_at_point(
        const std::vector<real2> &dofs,
        const dealii::FESystem<dim> &fe,
        const dealii::Point<dim> &point) const;

    /// Evaluate exact Jacobian determinant polynomial and uses Bernstein polynomials to determine positivity
    bool check_valid_cell(const typename DoFHandlerType::cell_iterator &cell) const;

    /// Evaluate exact Jacobian determinant polynomial and uses Bernstein polynomials to determine positivity
    bool fix_invalid_cell(const typename DoFHandlerType::cell_iterator &cell);

    /// Used to transform coefficients from a Lagrange basis to a Bernstein basis
    dealii::FullMatrix<double> lagrange_to_bernstein_operator;
    /// Evaluates the operator to obtain Bernstein coefficients from a set of Lagrange coefficients
    /** This is used in the evaluation of the Jacobian positivity by checking the convex hull of the
     *  resulting Bezier curve.
     */
    void evaluate_lagrange_to_bernstein_operator(const unsigned int order);

    void output_results_vtk (const unsigned int cycle) const; ///< Output mesh with metric informations


    // /// Evaluate cell metric Jacobian
    // /** The metric Jacobian is given by the gradient of the physical location
    //  *  with respect to the reference locations
    //  */
    //  dealii::Tensor<2,dim,real> cell_jacobian (const typename dealii::Triangulation<dim,spacedim>::cell_iterator &cell, const dealii::Point<dim> &point) const override
    //  {
    //  }

    /// Globally refines the high-order mesh.
    void refine_global();

    /// Prepares the solution transfer such that the curved refined grid is on top of the curved coarse grid.
    /** This function needs to be called before dealii::Triangulation::execute_coarsening_and_refinement() or dealii::Triangulation::refine_global()
     *  and this->execute_coarsening_and_refinement().
     */
    void prepare_for_coarsening_and_refinement();
    /// Executes the solution transfer such that the curved refined grid is on top of the curved coarse grid.
    /** This function needs to be after this->prepare_for_coarsening_and_refinement() and
     *  dealii::Triangulation::execute_coarsening_and_refinement() or dealii::Triangulation::refine_global().
     */
    void execute_coarsening_and_refinement(const bool output_mesh = false);
    
    /// Returns the coordinates/vertex of the smallest cell.
    dealii::Point<dim> smallest_cell_coordinates();

    /// Use Lagrange polynomial to represent the spatial location.
    const dealii::FE_Q<dim>     fe_q;
    /// Using system of polynomials to represent the x, y, and z directions.
    const dealii::FESystem<dim> fe_system;


    /// MappingFEField that will provide the polynomial-based grid.
    /** It is a shared smart pointer because the constructor requires the dof_handler_grid to be properly initialized.
     *  See discussion in the following
     *  <a href=" https://stackoverflow.com/questions/7557153/defining-an-object-without-calling-its-constructor-in-c">thread</a>.
     */
    std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> mapping_fe_field;

    /// MappingFEField that will provide the polynomial-based grid for the initial volume_nodes.
    /** Will likely be used for deformations based on initial grids.
     *  It is a shared smart pointer because the constructor requires the dof_handler_grid to be properly initialized.
     *  See discussion in the following
     *  <a href=" https://stackoverflow.com/questions/7557153/defining-an-object-without-calling-its-constructor-in-c">thread</a>.
     */
    std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> initial_mapping_fe_field;

    dealii::IndexSet locally_owned_dofs_grid; ///< Locally own degrees of freedom for the grid
    dealii::IndexSet ghost_dofs_grid; ///< Locally relevant ghost degrees of freedom for the grid
    dealii::IndexSet locally_relevant_dofs_grid; ///< Union of locally owned degrees of freedom and relevant ghost degrees of freedom for the grid

    static unsigned int nth_refinement; ///< Used to name the various files outputted.
protected:
    int n_mpi; ///< Number of MPI processes.
    int mpi_rank; ///< This processor's MPI rank.
    /// Update list of surface indices (locally_relevant_surface_nodes_indices and locally_relevant_surface_nodes_boundary_id)
    void update_surface_indices();

    /// Used for the SolutionTransfer when performing grid adaptation.
    VectorType old_volume_nodes;

    /** Transfers the coarse curved curve onto the fine curved grid.
     *  Used in prepare_for_coarsening_and_refinement() and execute_coarsening_and_refinement()
     */
    SolutionTransfer solution_transfer;

    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

    /// Evaluate the determinant of a matrix given in the format of a std::array<dealii::Tensor<1,dim,real2>,dim>.
    /** The indices of the array represent the matrix rows, and the indices of the Tensor represents its columns.
     */
    template <typename real2>
    real2 determinant(const std::array< dealii::Tensor<1,dim,real2>, dim > jacobian) const;

    /// A stripped down copy of dealii::VectorTools::get_position_vector()
    void get_position_vector(const DoFHandlerType &dh, VectorType &vector, const dealii::ComponentMask &mask);

    /// Another version of dealii::VectorTools::get_position_vector() that projects the points instead
    /// of interpolating them.
    void get_projected_position_vector(const DoFHandlerType &dh, VectorType &vector, const dealii::ComponentMask &mask);

};

/// Postprocessor used to output the grid.
template <int dim>
class GridPostprocessor : public dealii::DataPostprocessor<dim>
{
public:
    // /// Constructor
    // GridPostprocessor();

    /// Evaluates the values of interest to output.
    virtual void evaluate_vector_field (const dealii::DataPostprocessorInputs::Vector<dim> &inputs, std::vector<dealii::Vector<double>> &computed_quantities) const override;
    /// Returns the names associated with the output data.
    virtual std::vector<std::string> get_names () const override;
    /// Returns the DCI associated with the output data.
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> get_data_component_interpretation () const override;
    /// Returns the update flags required to evaluate the output data.
    virtual dealii::UpdateFlags get_needed_update_flags () const override;
};


} // namespace PHiLiP

#endif
