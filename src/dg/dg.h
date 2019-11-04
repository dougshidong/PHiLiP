#ifndef __DISCONTINUOUSGALERKIN_H__
#define __DISCONTINUOUSGALERKIN_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h> 


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <Sacado.hpp>

#include "dg/high_order_grid.h"
#include "physics/physics.h"
#include "numerical_flux/numerical_flux.h"
#include "parameters/all_parameters.h"

// Template specialization of MappingFEField
//extern template class dealii::MappingFEField<PHILIP_DIM,PHILIP_DIM,dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<PHILIP_DIM> >;
namespace PHiLiP {

//#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
//    using Triangulation = dealii::Triangulation<dim>;
//#else
//    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
//#endif
//namespace PHiLiP {
//#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
//    template <int dim> using Triangulation = dealii::Triangulation<dim>;
//#else
//    template <int dim> using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
//#endif

/// DGBase is independent of the number of state variables.
/**  This base class allows the use of arrays to efficiently allocate the data structures
  *  through std::array in the derived class DG.
  *  This class is the one being returned by the DGFactory and is the main
  *  interface for a user to call its main functions such as "assemble_residual".
  *
  *  Discretizes the problem
  *  \f[
  *      \frac{\partial \mathbf{u}}{\partial t} 
  *      + \boldsymbol\nabla \cdot
  *      ( \mathbf{F}_{conv}(\mathbf{u})
  *      + \mathbf{F}_{diss}(\mathbf{u},\boldsymbol\nabla\mathbf{u}) )
  *      = \mathbf{q}
  *  \f]
  *  
  *  Also defines the main loop of the DGWeak class which is assemble_residual
  */
template <int dim, typename real>
class DGBase 
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
public:
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Number of state variables.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const int nstate;

    /// Maximum degree used for p-refinement.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const unsigned int max_degree;

    /// Principal constructor that will call delegated constructor.
    /** Will initialize mapping, fe_dg, all_parameters, volume_quadrature, and face_quadrature
     *  from DGBase. The it will new some FEValues that will be used to retrieve the
     *  finite element values at physical locations.
     *
     *  Passes create_collection_tuple() to the delegated constructor.
     */
    DGBase(const int nstate_input,
           const Parameters::AllParameters *const parameters_input,
           const unsigned int max_degree,
           Triangulation *const triangulation_input);


    /// Makes for cleaner doxygen documentation
    using MassiveCollectionTuple = std::tuple<
        //dealii::hp::MappingCollection<dim>, // Mapping
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::QCollection<1>, // 1D quadrature for strong form
        dealii::hp::FECollection<dim> >;  // Lagrange polynomials for strong form

    /// Delegated constructor that initializes collections.
    /** Since a function is used to generate multiple different objects, a delegated
     *  constructor is used to unwrap the tuple and initialize the collections.
     *  
     *  The tuple is built from create_collection_tuple(). */
    DGBase( const int nstate_input,
            const Parameters::AllParameters *const parameters_input,
            const unsigned int max_degree_input,
            Triangulation *const triangulation_input,
            const MassiveCollectionTuple collection_tuple);

    virtual ~DGBase(); ///< Destructor.

    Triangulation *const triangulation; ///< Mesh
    /// Sets the triangulation for 2D and 3D. Should be done before allocate system
    //void set_triangulation(Triangulation *triangulation_input);

    /// Refers to a collection Mappings, which represents the high-order grid.
    /** Since we are interested in performing mesh movement for optimization purposes,
     *  this is not a constant member variables.
     */
    //dealii::hp::MappingCollection<dim> mapping_collection;

    void set_all_cells_fe_degree ( const unsigned int degree );

    /// Allocates the system.
    /** Must be done after setting the mesh and before assembling the system. */
    virtual void allocate_system ();

    /// Allocates and evaluates the mass matrices for the entire grid
    /*  Although straightforward, this has not been tested yet.
     *  Will be required for accurate time-stepping or nonlinear problems
     */
    void evaluate_mass_matrices (bool do_inverse_mass_matrix = false);

    /// Evaluates the maximum stable time step
    /*  If exact_time_stepping = true, use the same time step for the entire solution
     *  NOT YET IMPLEMENTED
     */
    std::vector<real> evaluate_time_steps (const bool exact_time_stepping);

    /// Add mass matrices to the system scaled by a factor (likely time-step)
    /*  Although straightforward, this has not been tested yet.
     *  Will be required for accurate time-stepping or nonlinear problems
     */
    void add_mass_matrices (const real scale);

    double get_residual_l2norm () const; ///< Returns the L2-norm of the right_hand_side vector

    unsigned int n_dofs() const; ///< Number of degrees of freedom


    /// Sparsity pattern used on the system_matrix
    /** Not sure we need to store it.  */
    dealii::SparsityPattern sparsity_pattern;

    /// Sparsity pattern used on the system_matrix
    /** Not sure we need to store it.  */
    dealii::SparsityPattern mass_sparsity_pattern;

    /// Global mass matrix
    /** Should be block diagonal where each block contains the mass matrix of each cell.  */
    dealii::TrilinosWrappers::SparseMatrix global_mass_matrix;
    /// Global inverser mass matrix
    /** Should be block diagonal where each block contains the inverse mass matrix of each cell.  */
    dealii::TrilinosWrappers::SparseMatrix global_inverse_mass_matrix;
    /// System matrix corresponding to the derivative of the right_hand_side with
    /// respect to the solution
    dealii::TrilinosWrappers::SparseMatrix system_matrix;

    /// Residual of the current solution
    /** Weak form.
     * 
     *  The right-hand side sends all the term to the side of the source term.
     * 
     *  Given
     *  \f[
     *      \frac{\partial \mathbf{u}}{\partial t} 
     *      + \boldsymbol\nabla \cdot
     *      ( \mathbf{F}_{conv}(\mathbf{u})
     *      + \mathbf{F}_{diss}(\mathbf{u},\boldsymbol\nabla\mathbf{u}) )
     *      = \mathbf{q}
     *  \f]
     *  The right-hand side is given by
     *  \f[
     *      \mathbf{\text{rhs}} = - \boldsymbol\nabla \cdot
     *            ( \mathbf{F}_{conv}(\mathbf{u})
     *            + \mathbf{F}_{diss}(\mathbf{u},\boldsymbol\nabla\mathbf{u}) )
     *            + \mathbf{q}
     *  \f]
     *
     *  It is important to note that the \f$\mathbf{F}_{diss}\f$ is positive in the DG
     *  formulation. Therefore, the PhysicsBase class should have a negative when
     *  considering stable applications of diffusion.
     * 
     */
    dealii::LinearAlgebra::distributed::Vector<double> right_hand_side;

    dealii::IndexSet locally_owned_dofs; ///< Locally own degrees of freedom
    dealii::IndexSet ghost_dofs; ///< Locally relevant ghost degrees of freedom
    dealii::IndexSet locally_relevant_dofs; ///< Union of locally owned degrees of freedom and relevant ghost degrees of freedom

    dealii::IndexSet locally_owned_dofs_grid; ///< Locally own degrees of freedom for the grid
    dealii::IndexSet ghost_dofs_grid; ///< Locally relevant ghost degrees of freedom for the grid
    dealii::IndexSet locally_relevant_dofs_grid; ///< Union of locally owned degrees of freedom and relevant ghost degrees of freedom for the grid
    /// Current modal coefficients of the solution
    /** Note that the current processor has read-access to all locally_relevant_dofs
     *  and has write-access to all locally_owned_dofs
     */
    dealii::LinearAlgebra::distributed::Vector<double> solution;

    /// Evaluate SparsityPattern of dRdX
    /*  Where R represents the residual and X represents the grid degrees of freedom stored as high_order_grid.nodes.
     */
    dealii::SparsityPattern get_dRdX_sparsity_pattern ();

    /// Evaluate dRdX using finite-differences
    /*  Where R represents the residual and X represents the grid degrees of freedom stored as high_order_grid.nodes.
     */
    dealii::TrilinosWrappers::SparseMatrix get_dRdX_finite_differences (dealii::SparsityPattern dRdX_sparsity_pattern);

    void initialize_manufactured_solution (); ///< Virtual function defined in DG

    void output_results_vtk (const unsigned int ith_grid); ///< Output solution
    void output_paraview_results (std::string filename); ///< Outputs a paraview file to view the solution

    /// Main loop of the DG class.
    /** Evaluates the right-hand-side \f$ \mathbf{R(\mathbf{u}}) \f$ of the system
     *
     *  \f[
     *      \frac{\partial \mathbf{u}}{\partial t} = \mathbf{R(\mathbf{u}}) = 
     *      - \boldsymbol\nabla \cdot
     *      ( \mathbf{F}_{conv}(\mathbf{u})
     *      + \mathbf{F}_{diss}(\mathbf{u},\boldsymbol\nabla\mathbf{u}) )
     *      + \mathbf{q}
     *  \f]
     *
     *  As well as sets the
     *  \f[
     *  \mathbf{\text{system_matrix}} = \frac{\partial \mathbf{R}}{\partial \mathbf{u}}
     *  \f]
     *
     * It loops over all the cells, evaluates the volume contributions,
     * then loops over the faces of the current cell. Four scenarios may happen
     *
     * 1. Boundary condition.
     *
     * 2. Current face has children. Therefore, neighbor is finer. In that case,
     * loop over neighbor faces to compute its face contributions.
     *
     * 3. Neighbor has same coarseness. Cell with lower global index will be used
     * to compute the face contribution.
     *
     * 4. Neighbor is coarser. Therefore, the current cell is the finer one.
     * Do nothing since this cell will be taken care of by scenario 2.
     *    
     */
    //void assemble_residual_dRdW ();
    void assemble_residual (const bool compute_dRdW=false);

    /// Used in assemble_residual(). 
    /** IMPORTANT: This does not fully compute the cell residual since it might not
     *  perform the work on all the faces.
     *  All the active cells must be traversed to ensure that the right hand side is correct.
     */
    template<typename DoFCellAccessorType>
    void assemble_cell_residual (
        const DoFCellAccessorType &current_cell,
        const bool compute_dRdW,
        dealii::hp::FEValues<dim,dim>        &fe_values_collection_volume,
        dealii::hp::FEFaceValues<dim,dim>    &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>    &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim> &fe_values_collection_subface,
        dealii::hp::FEValues<dim,dim>        &fe_values_collection_volume_lagrange,
        dealii::LinearAlgebra::distributed::Vector<double> &rhs);

    /// Finite Element Collection for p-finite-element to represent the solution
    /** This is a collection of FESystems */
    const dealii::hp::FECollection<dim>    fe_collection;

    /// Finite Element Collection to represent the high-order grid
    /** This is a collection of FESystems.
     *  Unfortunately, deal.II doesn't have a working hp Mapping FE field.
     *  Therefore, every grid/cell will use the maximal polynomial mapping regardless of the solution order.
     */
    //const dealii::hp::FECollection<dim>    fe_collection_grid;
    //const dealii::FESystem<dim>    fe_grid;

    /// Quadrature used to evaluate volume integrals.
    dealii::hp::QCollection<dim>     volume_quadrature_collection;
    /// Quadrature used to evaluate face integrals.
    dealii::hp::QCollection<dim-1>   face_quadrature_collection;
    /// 1D quadrature to generate Lagrange polynomials for the sake of flux interpolation.
    dealii::hp::QCollection<1>       oned_quadrature_collection;

protected:
    /// Lagrange basis used in strong form
    /** This is a collection of scalar Lagrange bases */
    const dealii::hp::FECollection<dim>  fe_collection_lagrange;

public:
    /// Degrees of freedom handler
    /*  Allows us to iterate over the finite elements' degrees of freedom.
     *  Note that since we are not using FESystem, we need to multiply
     *  the index by a factor of "nstate"
     *
     *  Must be defined after fe_dg since it is a subscriptor of fe_dg.
     *  Destructor are called in reverse order in which they appear in class definition. 
     */ 
    dealii::hp::DoFHandler<dim> dof_handler;

    /// High order grid that will provide the MappingFEField
    HighOrderGrid<dim,real> high_order_grid;

protected:

    /// Evaluate the integral over the cell volume.
    /** Compute both the right-hand side and the block of the Jacobian */
    virtual void assemble_volume_terms_implicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange) = 0;
    /// Evaluate the integral over the cell edges that are on domain boundaries
    /** Compute both the right-hand side and the block of the Jacobian */
    virtual void assemble_boundary_term_implicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs) = 0;
    /// Evaluate the integral over the internal cell edges
    /** Compute both the right-hand side and the block of the Jacobian.
     *  This adds the contribution to both cell's residual and effectively 
     *  computes 4 block contributions to the Jacobian. */
    virtual void assemble_face_term_implicit(
        const dealii::FEValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs) = 0;

    /// Evaluate the integral over the cell volume
    virtual void assemble_volume_terms_explicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange) = 0;
    /// Evaluate the integral over the cell edges that are on domain boundaries
    virtual void assemble_boundary_term_explicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs) = 0;
    /// Evaluate the integral over the internal cell edges
    virtual void assemble_face_term_explicit(
        const dealii::FEValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs) = 0;

    // /// Lagrange polynomial basis
    // /** Refer to deal.II documentation for the various polynomial types
    //  *  Note that only tensor-product polynomials recover optimal convergence
    //  *  since the mapping from the reference to physical element is a bilnear mapping.
    //  *
    //  *  As a result, FE_DGP does not give optimal convergence orders.
    //  *  See [discussion](https://groups.google.com/d/msg/dealii/f9NzCp8dnyU/aAdO6I9JCwAJ)
    //  *  on deal.II group forum]
    //  */
    // const dealii::FE_DGQ<dim> fe_dg;
    // //const dealii::FE_DGQLegendre<dim> fe_dg;

    // /// Finite Element System used for vector-valued problems
    // /** Note that we will use the same set of polynomials for all state equations
    //  *  therefore, FESystem is only used for the ease of obtaining sizes and 
    //  *  global indexing.
    //  *
    //  *  When evaluating the function values, we will still be using fe_dg
    //  */
    // const dealii::FESystem<dim,dim> fe_system;
    //

    // /// QGauss is Gauss-Legendre quadrature nodes
    // dealii::QGauss<1>     oned_quadrature; // For the strong form
    // dealii::QGauss<dim>   volume_quadrature;
    // dealii::QGauss<dim-1> face_quadrature;
    // // const dealii::QGaussLobatto<dim>   volume_quadrature;
    // // const dealii::QGaussLobatto<dim-1> face_quadrature;

    /// Update flags needed at volume points.
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;
    /// Update flags needed at face points.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;
    /// Update flags needed at neighbor' face points. 
    /** NOTE: With hp-adaptation, might need to query neighbor's quadrature points depending on the order of the cells. */
    const dealii::UpdateFlags neighbor_face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;



protected:
    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
private:

    /// Evaluate the average penalty term at the face
    /** For a cell with solution of degree p, and Hausdorff measure h,
     *  which represents the element dimension orthogonal to the face,
     *  the penalty term is given by p*(p+1)/h .
     */
    template<typename DoFCellAccessorType>
    real evaluate_penalty_scaling (
        const DoFCellAccessorType &cell,
        const int iface,
        const dealii::hp::FECollection<dim> fe_collection) const;

    /// In the case that two cells have the same coarseness, this function decides if the current cell should perform the work.
    /** In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face.
     *  We cannot use the cell->index() because the index is relative to the distributed triangulation.
     *  Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell.
     *
     *  For a locally owned neighbor cell, cell with lower index does work or if both cells have same index, then cell at the lower level does the work
     *  See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
     */
    template<typename DoFCellAccessorType1, typename DoFCellAccessorType2>
    bool current_cell_should_do_the_work (const DoFCellAccessorType1 &current_cell, const DoFCellAccessorType2 &neighbor_cell) const;

    /// Used in the delegated constructor
    /** The main reason we use this weird function is because all of the above objects
     *  need to be looped with the various p-orders. This function allows us to do this in a
     *  single function instead of having like 6 different functions to initialize each of them.
     */
    MassiveCollectionTuple create_collection_tuple(const unsigned int max_degree, const int nstate, const Parameters::AllParameters *const parameters_input) const;

}; // end of DGBase class

/// DGWeak class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
template <int dim, int nstate, typename real>
class DGWeak : public DGBase<dim, real>
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
public:
    /// Constructor.
    DGWeak(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree,
        Triangulation *const triangulation_input);

    ~DGWeak(); ///< Destructor.

private:

    /// Contains the physics of the PDE
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, Sacado::Fad::DFad<real> > > pde_physics;
    /// Convective numerical flux
    NumericalFlux::NumericalFluxConvective<dim, nstate, Sacado::Fad::DFad<real> > *conv_num_flux;
    /// Dissipative numerical flux
    NumericalFlux::NumericalFluxDissipative<dim, nstate, Sacado::Fad::DFad<real> > *diss_num_flux;

    /// Contains the physics of the PDE
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, real > > pde_physics_double;
    /// Convective numerical flux
    NumericalFlux::NumericalFluxConvective<dim, nstate, real > *conv_num_flux_double;
    /// Dissipative numerical flux
    NumericalFlux::NumericalFluxDissipative<dim, nstate, real > *diss_num_flux_double;

    /// Evaluate the integral over the cell volume
    /** Compute both the right-hand side and the block of the Jacobian */
    void assemble_volume_terms_implicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    /** Compute both the right-hand side and the block of the Jacobian */
    void assemble_boundary_term_implicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
    /** Compute both the right-hand side and the block of the Jacobian.
     *  This adds the contribution to both cell's residual and effectively 
     *  computes 4 block contributions to the Jacobian. */
    void assemble_face_term_implicit(
        const dealii::FEValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs);

    /// Evaluate the integral over the cell volume
    void assemble_volume_terms_explicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs, 
        const dealii::FEValues<dim,dim> &fe_values_lagrange);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    void assemble_boundary_term_explicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
    void assemble_face_term_explicit(
        const dealii::FEValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs);

    using DGBase<dim,real>::mpi_communicator; ///< MPI communicator
    using DGBase<dim,real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // end of DGWeak class

/// DGStrong class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
template <int dim, int nstate, typename real>
class DGStrong : public DGBase<dim, real>
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
public:
    /// Constructor
    DGStrong(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree,
        Triangulation *const triangulation_input);

    /// Destructor
    ~DGStrong();

private:
    /// Contains the physics of the PDE
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, Sacado::Fad::DFad<real> > > pde_physics;
    /// Convective numerical flux
    NumericalFlux::NumericalFluxConvective<dim, nstate, Sacado::Fad::DFad<real> > *conv_num_flux;
    /// Dissipative numerical flux
    NumericalFlux::NumericalFluxDissipative<dim, nstate, Sacado::Fad::DFad<real> > *diss_num_flux;


    /// Contains the physics of the PDE
    std::shared_ptr < Physics::PhysicsBase<dim, nstate, real > > pde_physics_double;
    /// Convective numerical flux
    NumericalFlux::NumericalFluxConvective<dim, nstate, real > *conv_num_flux_double;
    /// Dissipative numerical flux
    NumericalFlux::NumericalFluxDissipative<dim, nstate, real > *diss_num_flux_double;

    /// Evaluate the integral over the cell volume
    /** Compute both the right-hand side and the block of the Jacobian */
    void assemble_volume_terms_implicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    /** Compute both the right-hand side and the block of the Jacobian */
    void assemble_boundary_term_implicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
    /** Compute both the right-hand side and the block of the Jacobian.
     *  This adds the contribution to both cell's residual and effectively 
     *  computes 4 block contributions to the Jacobian. */
    void assemble_face_term_implicit(
        const dealii::FEValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs);

    /// Evaluate the integral over the cell volume
    void assemble_volume_terms_explicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    void assemble_boundary_term_explicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
    void assemble_face_term_explicit(
        const dealii::FEValuesBase<dim,dim>     &fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     &fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs);

    using DGBase<dim,real>::all_parameters; ///< Pointer to all parameters
    using DGBase<dim,real>::mpi_communicator; ///< MPI communicator
    using DGBase<dim,real>::pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

}; // end of DGStrong class

/// This class creates a new DGBase object
/** This allows the DGBase to not be templated on the number of state variables
  * while allowing DG to be template on the number of state variables */
template <int dim, typename real>
class DGFactory
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::Triangulation<dim>;
#else
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
public:
    /// Creates a derived object DG, but returns it as DGBase.
    /** That way, the caller is agnostic to the number of state variables */
    static std::shared_ptr< DGBase<dim,real> >
        create_discontinuous_galerkin(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree,
        Triangulation *const triangulation_input);
};

} // PHiLiP namespace

#endif
