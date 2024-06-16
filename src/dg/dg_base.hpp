#ifndef PHILIP_DG_BASE_HPP
#define PHILIP_DG_BASE_HPP

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/qprojector.h>

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

#include <Epetra_RowMatrixTransposer.h>
#include <AztecOO.h>

#include "ADTypes.hpp"
#include <Sacado.hpp>
#include <CoDiPack/include/codi.hpp>

#include "mesh/high_order_grid.h"
#include "physics/physics.h"
#include "physics/model.h"
#include "numerical_flux/numerical_flux_factory.hpp"
#include "numerical_flux/convective_numerical_flux.hpp"
#include "numerical_flux/viscous_numerical_flux.hpp"
#include "parameters/all_parameters.h"
#include "operators/operators.h"
#include "artificial_dissipation_factory.h"

#include <time.h>
#include <deal.II/base/timer.h>

// Template specialization of MappingFEField
//extern template class dealii::MappingFEField<PHILIP_DIM,PHILIP_DIM,dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<PHILIP_DIM> >;
namespace PHiLiP {

/// Get the coefficients of a function projected onto a set of basis (to be replaced with operators->projection_operator). 
template<int dim, typename real>
std::vector< real > project_function(
    const std::vector< real > &function_coeff,
    const dealii::FESystem<dim,dim> &fe_input,
    const dealii::FESystem<dim,dim> &fe_output,
    const dealii::QGauss<dim> &projection_quadrature);


/// DGBase is independent of the number of state variables.
/**  This base class allows the use of arrays to efficiently allocate the data structures
  *  through std::array in the derived class DGBaseState.
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
  */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGBase 
{
public:
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = MeshType;

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Number of state variables.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const int nstate;

    /// Initial polynomial degree assigned during constructor
    const unsigned int initial_degree;

    /// Maximum degree used for p-refi1nement.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const unsigned int max_degree;

    /// Maximum grid degree used for hp-refi1nement.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const unsigned int max_grid_degree;

    /// Destructor
    virtual ~DGBase() = default;

    /// Principal constructor that will call delegated constructor.
    /** Will initialize mapping, fe_dg, all_parameters, volume_quadrature, and face_quadrature
     *  from DGBase. The it will new some FEValues that will be used to retrieve the
     *  finite element values at physical locations.
     *
     *  Passes create_collection_tuple() to the delegated constructor.
     */
    DGBase(const int nstate_input,
           const Parameters::AllParameters *const parameters_input,
           const unsigned int degree,
           const unsigned int max_degree_input,
           const unsigned int grid_degree_input,
           const std::shared_ptr<Triangulation> triangulation_input);


    /// Reinitializes the DG object after a change of triangulation
    /** Calls respective function for high-order-grid and initializes dof_handler
     *  again. Also resets all fe_degrees to intial_degree set during constructor.
     */
    void reinit();

    /// Makes for cleaner doxygen documentation
    using MassiveCollectionTuple = std::tuple<
        //dealii::hp::MappingCollection<dim>, // Mapping
        dealii::hp::FECollection<dim>, // Solution FE
        dealii::hp::QCollection<dim>,  // Volume quadrature
        dealii::hp::QCollection<dim-1>, // Face quadrature
        dealii::hp::FECollection<dim>,  // Lagrange polynomials for strong form
        dealii::hp::FECollection<1>,  // Solution FE 1D
        dealii::hp::FECollection<1>,  // Solution FE 1D for a single state
        dealii::hp::FECollection<1>,   // Collocated flux basis 1D for strong form
        dealii::hp::QCollection<1> >; // 1D quadrature for strong form

    /// Delegated constructor that initializes collections.
    /** Since a function is used to generate multiple different objects, a delegated
     *  constructor is used to unwrap the tuple and initialize the collections.
     *
     *  The tuple is built from create_collection_tuple(). */
    DGBase( const int nstate_input,
            const Parameters::AllParameters *const parameters_input,
            const unsigned int degree,
            const unsigned int max_degree_input,
            const unsigned int grid_degree_input,
            const std::shared_ptr<Triangulation> triangulation_input,
            const MassiveCollectionTuple collection_tuple);

    std::shared_ptr<Triangulation> triangulation; ///< Mesh


    /// Sets the associated high order grid with the provided one.
    void set_high_order_grid(std::shared_ptr<HighOrderGrid<dim,real,MeshType>> new_high_order_grid);

    /// Refers to a collection Mappings, which represents the high-order grid.
    /** Since we are interested in performing mesh movement for optimization purposes,
     *  this is not a constant member variables.
     */
    //dealii::hp::MappingCollection<dim> mapping_collection;
    void set_all_cells_fe_degree ( const unsigned int degree );

    /// Gets the maximum value of currently active FE degree
    unsigned int get_max_fe_degree();

    /// Gets the minimum value of currently active FE degree
    unsigned int get_min_fe_degree();
    
    /// Returns the coordinates of the most refined cell.
    dealii::Point<dim> coordinates_of_highest_refined_cell(bool check_for_p_refined_cell = false);

    /// Allocates the system.
    /** Must be done after setting the mesh and before assembling the system. */
    virtual void allocate_system (const bool compute_dRdW = true, 
                                  const bool compute_dRdX = true, 
                                  const bool compute_d2R = true);

private:
    /// Allocates the second derivatives.
    /** Is called when assembling the residual's second derivatives, and is currently empty
     *  due to being cleared by the allocate_system().
     */
    virtual void allocate_second_derivatives ();

    /// Allocates the residual derivatives w.r.t the volume nodes.
    /** Is called when assembling the residual's second derivatives, and is currently empty
     *  due to being cleared by the allocate_system().
     */
    virtual void allocate_dRdX ();

    /// Allocates variables of artificial dissipation.
    /** It is called by allocate_system() when artificial dissipation is set
     *  to true in the parameters file.
     */
    void allocate_artificial_dissipation();

public:

    /// Scales a solution update with the appropriate maximum time step.
    /** Used for steady state solutions using the explicit ODE solver.
     */
    void time_scale_solution_update ( dealii::LinearAlgebra::distributed::Vector<double> &solution_update, const real CFL ) const;

    /// Evaluate the time_scaled_global_mass_matrix such that the maximum time step
    /// cell-wise is taken into account.
    void time_scaled_mass_matrices(const real scale);

    /// Builds needed operators for cell residual loop.
    void reinit_operators_for_cell_residual_loop(
        const unsigned int                                poly_degree_int, 
        const unsigned int                                poly_degree_ext, 
        const unsigned int                                grid_degree,
        OPERATOR::basis_functions<dim,2*dim,real>         &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>         &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim,real>         &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>         &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>   &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real> &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real> &soln_basis_projection_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim,real> &mapping_basis);

    /// Builds needed operators to compute mass matrices/inverses efficiently.
    void reinit_operators_for_mass_matrix(
        const bool                                                       Cartesian_element,
        const unsigned int                                               poly_degree, 
        const unsigned int                                               grid_degree,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>                &mapping_basis,
        OPERATOR::basis_functions<dim,2*dim,real>                        &basis,
        OPERATOR::local_mass<dim,2*dim,real>                             &reference_mass_matrix,
        OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim,real>     &reference_FR,
        OPERATOR::local_Flux_Reconstruction_operator_aux<dim,2*dim,real> &reference_FR_aux,
        OPERATOR::derivative_p<dim,2*dim,real>                           &deriv_p);

    /// Allocates and evaluates the mass matrices for the entire grid
    void evaluate_mass_matrices (bool do_inverse_mass_matrix = false);

    /// Evaluates the metric dependent local mass matrices and inverses, then sets them in the global matrices.
    void evaluate_local_metric_dependent_mass_matrix_and_set_in_global_mass_matrix(
        const bool                                                       Cartesian_element,//Flag if cell is Cartesian
        const bool                                                       do_inverse_mass_matrix, 
        const unsigned int                                               poly_degree, 
        const unsigned int                                               curr_grid_degree, 
        const unsigned int                                               n_quad_pts, 
        const unsigned int                                               n_dofs_cell, 
        const std::vector<dealii::types::global_dof_index>               dofs_indices, 
        OPERATOR::metric_operators<real,dim,2*dim>                       &metric_oper,
        OPERATOR::basis_functions<dim,2*dim,real>                        &basis,
        OPERATOR::local_mass<dim,2*dim,real>                             &reference_mass_matrix,
        OPERATOR::local_Flux_Reconstruction_operator<dim,2*dim,real>     &reference_FR,
        OPERATOR::local_Flux_Reconstruction_operator_aux<dim,2*dim,real> &reference_FR_aux,
        OPERATOR::derivative_p<dim,2*dim,real>                           &deriv_p);

    /// Applies the inverse of the local metric dependent mass matrices when the global is not stored.
    /** We use matrix-free methods to apply the inverse of the local mass matrix on-the-fly 
    *   in each cell using sum-factorization techniques.
    */
    void apply_inverse_global_mass_matrix(
        const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
        dealii::LinearAlgebra::distributed::Vector<double> &output_vector,
        const bool use_auxiliary_eq = false);

    /// Applies the local metric dependent mass matrices when the global is not stored.
    /** We use matrix-free methods to apply the local mass matrix on-the-fly 
    *   in each cell using sum-factorization techniques.
    *   use_unmodified_mass_matrix flag allows the unmodified mass matrix to be used 
    *   for FR, i.e., use M rather than M+K.
    *   For example, if this function is used for an inner product <a,b>,
    *       setting `use_unmodified_mass_matrix = true` will result in an M norm (L2) inner product,
    *       whereas setting `use_unmodified_mass_matrix = false` will result in an M+K norm (broken Sobolev) inner product.
    */
    void apply_global_mass_matrix(
        const dealii::LinearAlgebra::distributed::Vector<double> &input_vector,
        dealii::LinearAlgebra::distributed::Vector<double> &output_vector,
        const bool use_auxiliary_eq = false,
        const bool use_unmodified_mass_matrix = false);

    /// Evaluates the maximum stable time step
    /** If exact_time_stepping = true, use the same time step for the entire solution
     *  NOT YET IMPLEMENTED
     */
    std::vector<real> evaluate_time_steps (const bool exact_time_stepping);

    /// Add mass matrices to the system scaled by a factor (likely time-step)
    /**  Although straightforward, this has not been tested yet.
     *  Will be required for accurate time-stepping or nonlinear problems
     */
    void add_mass_matrices (const real scale);

    /// Add time scaled mass matrices to the system.
    /** For pseudotime-stepping where the scaling depends on wavespeed and cell-size.
     */
    void add_time_scaled_mass_matrices();

    double get_residual_l2norm () const; ///< Returns the L2-norm of the right_hand_side vector

    double get_residual_linfnorm () const; ///< Returns the Linf-norm of the right_hand_side vector

    unsigned int n_dofs() const; ///< Number of degrees of freedom

    /// Set anisotropic flags based on jump indicator.
    /** Some cells must have already been tagged for refinement through some other indicator
     */
    void set_anisotropic_flags();

    /// Sparsity pattern used on the system_matrix
    /** Not sure we need to store it.  */
    dealii::SparsityPattern sparsity_pattern;

    /// Sparsity pattern used on the system_matrix
    /** Not sure we need to store it.  */
    dealii::SparsityPattern mass_sparsity_pattern;

    /// Global mass matrix divided by the time scales.
    /** Should be block diagonal where each block contains the scaled mass matrix of each cell.  */
    dealii::TrilinosWrappers::SparseMatrix time_scaled_global_mass_matrix;

    /// Global mass matrix
    /** Should be block diagonal where each block contains the mass matrix of each cell.  */
    dealii::TrilinosWrappers::SparseMatrix global_mass_matrix;
    /// Global inverser mass matrix
    /** Should be block diagonal where each block contains the inverse mass matrix of each cell.  */
    dealii::TrilinosWrappers::SparseMatrix global_inverse_mass_matrix;

    /// Global auxiliary mass matrix. 
    /** Note that it has a mass matrix in each dimension since the auxiliary variable is a tensor of size dim. We use the same matrix in each dim.*/
    dealii::TrilinosWrappers::SparseMatrix global_mass_matrix_auxiliary;

    /// Global inverse of the auxiliary mass matrix
    dealii::TrilinosWrappers::SparseMatrix global_inverse_mass_matrix_auxiliary;

    /// System matrix corresponding to the derivative of the right_hand_side with
    /// respect to the solution
    dealii::TrilinosWrappers::SparseMatrix system_matrix;

    /// System matrix corresponding to the derivative of the right_hand_side with
    /// respect to the solution TRANSPOSED.
    dealii::TrilinosWrappers::SparseMatrix system_matrix_transpose;

    /// Epetra_RowMatrixTransposer used to transpose the system_matrix.
    std::unique_ptr<Epetra_RowMatrixTransposer> epetra_rowmatrixtransposer_dRdW;

    //AztecOO dRdW_preconditioner_builder;

    /// System matrix corresponding to the derivative of the right_hand_side with
    /// respect to the volume volume_nodes Xv
    dealii::TrilinosWrappers::SparseMatrix dRdXv;

    /// System matrix corresponding to the second derivatives of the right_hand_side with
    /// respect to the solution
    dealii::TrilinosWrappers::SparseMatrix d2RdWdW;

    /// System matrix corresponding to the second derivatives of the right_hand_side with
    /// respect to the volume volume_nodes
    dealii::TrilinosWrappers::SparseMatrix d2RdXdX;
    //
    /// System matrix corresponding to the mixed second derivatives of the right_hand_side with
    /// respect to the solution and the volume volume_nodes
    dealii::TrilinosWrappers::SparseMatrix d2RdWdX;

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

    ///The auxiliary equations' right hand sides.
    std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> auxiliary_right_hand_side;

    ///The auxiliary equations' solution.
    std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> auxiliary_solution;
private:
    /// Modal coefficients of the solution used to compute dRdW last
    /// Will be used to avoid recomputing dRdW.
    dealii::LinearAlgebra::distributed::Vector<double> solution_dRdW;
    /// Modal coefficients of the grid nodes used to compute dRdW last
    /// Will be used to avoid recomputing dRdW.
    dealii::LinearAlgebra::distributed::Vector<double> volume_nodes_dRdW;

    /// CFL used to add mass matrix in the optimization FlowConstraints class
    double CFL_mass_dRdW;

    /// Modal coefficients of the solution used to compute dRdX last
    /// Will be used to avoid recomputing dRdX.
    dealii::LinearAlgebra::distributed::Vector<double> solution_dRdX;
    /// Modal coefficients of the grid nodes used to compute dRdX last
    /// Will be used to avoid recomputing dRdX.
    dealii::LinearAlgebra::distributed::Vector<double> volume_nodes_dRdX;

    /// Modal coefficients of the solution used to compute d2R last
    /// Will be used to avoid recomputing d2R.
    dealii::LinearAlgebra::distributed::Vector<double> solution_d2R;
    /// Modal coefficients of the grid nodes used to compute d2R last
    /// Will be used to avoid recomputing d2R.
    dealii::LinearAlgebra::distributed::Vector<double> volume_nodes_d2R;
    /// Dual variables to compute d2R last
    /// Will be used to avoid recomputing d2R.
    dealii::LinearAlgebra::distributed::Vector<double> dual_d2R;
public:

    /// Time it takes for the maximum wavespeed to cross the cell domain.
    /** Uses evaluate_CFL() which would be defined in the subclasses.
     *  This is because DGBase isn't templated on nstate and therefore, can't use
     *  the Physics to compute maximum wavespeeds.
     */
    dealii::Vector<double> cell_volume;

    /// Time it takes for the maximum wavespeed to cross the cell domain.
    /** Uses evaluate_CFL() which would be defined in the subclasses.
     *  This is because DGBase isn't templated on nstate and therefore, can't use
     *  the Physics to compute maximum wavespeeds.
     */
    dealii::Vector<double> max_dt_cell;

    /// Artificial dissipation in each cell.
    dealii::Vector<double> artificial_dissipation_coeffs;

    /// Artificial dissipation error ratio sensor in each cell.
    dealii::Vector<double> artificial_dissipation_se;

    template <typename real2>
    /** Discontinuity sensor with 4 parameters, based on projecting to p-1. */
    real2 discontinuity_sensor(
        const dealii::Quadrature<dim> &volume_quadrature,
        const std::vector< real2 > &soln_coeff_high,
        const dealii::FiniteElement<dim,dim> &fe_high,
        const std::vector<real2> &jac_det);

    /// Current optimization dual variables corresponding to the residual constraints also known as the adjoint
    /** This is used to evaluate the dot-product between the dual and the 2nd derivatives of the residual
     *  since storing the 2nd order partials of the residual is a very large 3rd order tensor.
     */
    dealii::LinearAlgebra::distributed::Vector<real> dual;

    /// Sets the stored dual variables used to compute the dual dotted with the residual Hessians
    void set_dual(const dealii::LinearAlgebra::distributed::Vector<real> &dual_input);

    /// Evaluate SparsityPattern of dRdX
    /*  Where R represents the residual and X represents the grid degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::SparsityPattern get_dRdX_sparsity_pattern ();

    /// Evaluate SparsityPattern of dRdW
    /*  Where R represents the residual and W represents the solution degrees of freedom.
     */
    dealii::SparsityPattern get_dRdW_sparsity_pattern ();

    /// Evaluate SparsityPattern of the residual Hessian dual.d2RdWdW
    /*  Where R represents the residual and W represents the solution degrees of freedom.
     */
    dealii::SparsityPattern get_d2RdWdW_sparsity_pattern ();

    /// Evaluate SparsityPattern of the residual Hessian dual.d2RdXdX
    /*  Where R represents the residual and X represents the grid degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::SparsityPattern get_d2RdXdX_sparsity_pattern ();

    /// Evaluate SparsityPattern of the residual Hessian dual.d2RdXdW
    /*  Where R represents the residual, W the solution DoF, and X represents the grid degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::SparsityPattern get_d2RdWdX_sparsity_pattern ();

    /// Evaluate SparsityPattern of dRdXs
    /*  Where R represents the residual and Xs represents the grid surface degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::SparsityPattern get_dRdXs_sparsity_pattern ();
    /// Evaluate SparsityPattern of the residual Hessian dual.d2RdXsdXs
    /*  Where R represents the residual and Xs represents the grid surface degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::SparsityPattern get_d2RdXsdXs_sparsity_pattern ();

    /// Evaluate SparsityPattern of the residual Hessian dual.d2RdXsdW
    /*  Where R represents the residual, W the solution DoF, and Xs represents the grid surface degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::SparsityPattern get_d2RdWdXs_sparsity_pattern ();

    /// Evaluate dRdX using finite-differences
    /*  Where R represents the residual and X represents the grid degrees of freedom stored as high_order_grid.volume_nodes.
     */
    dealii::TrilinosWrappers::SparseMatrix get_dRdX_finite_differences (dealii::SparsityPattern dRdX_sparsity_pattern);

    void initialize_manufactured_solution (); ///< Virtual function defined in DG

    void output_results_vtk (const unsigned int cycle, const double current_time=0.0); ///< Output solution
    void output_face_results_vtk (const unsigned int cycle, const double current_time=0.0); ///< Output Euler face solution

    bool update_artificial_diss;
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
    void assemble_residual (const bool compute_dRdW=false, const bool compute_dRdX=false, const bool compute_d2R=false, const double CFL_mass = 0.0);

    /// Used in assemble_residual().
    /** IMPORTANT: This does not fully compute the cell residual since it might not
     *  perform the work on all the faces.
     *  All the active cells must be traversed to ensure that the right hand side is correct.
     */
    template<typename DoFCellAccessorType1, typename DoFCellAccessorType2> // To be deleted
    void assemble_cell_residual (
        const DoFCellAccessorType1 &current_cell,
        const DoFCellAccessorType2 &current_metric_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
        dealii::hp::FEValues<dim,dim>                                      &fe_values_collection_volume,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim>                               &fe_values_collection_subface,
        dealii::hp::FEValues<dim,dim>                                      &fe_values_collection_volume_lagrange,
        OPERATOR::basis_functions<dim,2*dim,real>                          &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>                          &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim,real>                          &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>                          &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>                    &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real>                  &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real>                  &soln_basis_projection_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>                  &mapping_basis,
        const bool                                                         compute_auxiliary_right_hand_side,//flag on whether computing the Auxiliary variable's equations' residuals
        dealii::LinearAlgebra::distributed::Vector<double>                 &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux);
    
    template<typename adtype>
    void assemble_cell_residual_and_ad_derivatives (
        const dealii::TriaActiveIterator<dealii::DoFCellAccessor<dim, dim, false>> &current_cell,
        const dealii::TriaActiveIterator<dealii::DoFCellAccessor<dim, dim, false>> &current_metric_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
        dealii::hp::FEValues<dim,dim>                                      &fe_values_collection_volume,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                                  &fe_values_collection_face_ext,
        dealii::hp::FESubfaceValues<dim,dim>                               &fe_values_collection_subface,
        dealii::hp::FEValues<dim,dim>                                      &fe_values_collection_volume_lagrange,
        OPERATOR::basis_functions<dim,2*dim,real>                          &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>                          &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim,real>                          &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>                          &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>                    &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real>                  &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real>                  &soln_basis_projection_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>                  &mapping_basis,
        const bool                                                         compute_auxiliary_right_hand_side,//flag on whether computing the Auxiliary variable's equations' residuals
        dealii::LinearAlgebra::distributed::Vector<double>                 &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux);

    template <typename adtype>
    void assemble_volume_codi_taped_derivatives_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::vector<dealii::types::global_dof_index>     &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*soln_basis*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*flux_basis*/,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>             &/*flux_basis_stiffness*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_int*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_ext*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim>                        &fe_soln,
        dealii::Vector<real>                                   &local_rhs_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &/*local_auxiliary_RHS*/,
        std::vector<adtype>               &local_metric_int,
        const bool                                             /*compute_auxiliary_right_hand_side*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    template <typename adtype>
    void assemble_face_codi_taped_derivatives_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     /*poly_degree_int*/,
        const unsigned int                                     /*poly_degree_ext*/,
        const unsigned int                                     /*grid_degree_int*/,
        const unsigned int                                     /*grid_degree_ext*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*soln_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*soln_basis_ext*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*flux_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*flux_basis_ext*/,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>             &/*flux_basis_stiffness*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_int*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_ext*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_int*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_ext*/,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_ext,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &/*current_cell_rhs_aux*/,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &/*rhs_aux*/,
        std::vector<adtype>                  &local_metric,
        const bool                                             /*compute_auxiliary_right_hand_side*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    template <typename adtype>
    void assemble_boundary_codi_taped_derivatives_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     boundary_id,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     /*poly_degree*/,
        const unsigned int                                     /*grid_degree*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*soln_basis*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*flux_basis*/,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>             &/*flux_basis_stiffness*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_int*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_ext*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper*/,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                        &fe_soln,
        dealii::Vector<real>                                   &local_rhs_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &/*local_auxiliary_RHS*/,
        std::vector<adtype>                  &local_metric,
        const bool                                             /*compute_auxiliary_right_hand_side*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    template <typename adtype>
    void assemble_subface_codi_taped_derivatives_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const unsigned int                                     neighbor_i_subface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     /*poly_degree_int*/,
        const unsigned int                                     /*poly_degree_ext*/,
        const unsigned int                                     /*grid_degree_int*/,
        const unsigned int                                     /*grid_degree_ext*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*soln_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*soln_basis_ext*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*flux_basis_int*/,
        OPERATOR::basis_functions<dim,2*dim,real>                   &/*flux_basis_ext*/,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>             &/*flux_basis_stiffness*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_int*/,
        OPERATOR::vol_projection_operator<dim,2*dim,real>           &/*soln_basis_projection_oper_ext*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_int*/,
        OPERATOR::metric_operators<real,dim,2*dim>             &/*metric_oper_ext*/,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>           &/*mapping_basis*/,
        std::array<std::vector<real>,dim>                      &/*mapping_support_points*/,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FESubfaceValues<dim,dim>                   &fe_values_collection_subface,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &/*current_cell_rhs_aux*/,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &/*rhs_aux*/,
        std::vector<adtype>                  &local_metric,
        const bool                                             /*compute_auxiliary_right_hand_side*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);

    template <typename adtype>
    void assemble_face_subface_codi_taped_derivatives_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dofs_indices_int,
        const std::vector<dealii::types::global_dof_index> &metric_dofs_indices_ext,
        const std::vector<dealii::types::global_dof_index> &soln_dofs_indices_int,
        const std::vector<dealii::types::global_dof_index> &soln_dofs_indices_ext,
        dealii::Vector<real>          &local_rhs_int_cell,
        dealii::Vector<real>          &local_rhs_ext_cell,
        std::vector<adtype>                  &metric_int,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R);
    
    virtual void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<double> &soln_coeffs,
        const std::vector<double> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<double> &rhs, 
        double &dual_dot_residual)=0;
    
    virtual void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<codi_JacobianComputationType> &soln_coeffs,
        const std::vector<codi_JacobianComputationType> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<codi_JacobianComputationType> &rhs, 
        codi_JacobianComputationType &dual_dot_residual)=0;
    
    virtual void assemble_volume_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<codi_HessianComputationType> &soln_coeffs,
        const std::vector<codi_HessianComputationType> &metric_coeffs,
        const std::vector<real> &local_dual,
        const std::vector<dealii::types::global_dof_index>  &soln_dofs_indices,
        const std::vector<dealii::types::global_dof_index>  &metric_dofs_indices,
        const unsigned int  poly_degree,
        const unsigned int  grid_degree,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>  &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim> &fe_soln,
        std::vector<codi_HessianComputationType> &rhs, 
        codi_HessianComputationType &dual_dot_residual)=0;
    
    virtual void assemble_boundary_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<double> &soln_coeffs,
        const std::vector<double> &metric_coeffs,
        const std::vector< real > &local_dual,
        const unsigned int face_number,
        const unsigned int boundary_id,
        dealii::hp::FEFaceValues<dim,dim> &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim> &fe_soln,
        const real penalty,
        std::vector<double> &rhs,
        double &dual_dot_residual)=0;
    
    virtual void assemble_boundary_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<codi_JacobianComputationType> &soln_coeffs,
        const std::vector<codi_JacobianComputationType> &metric_coeffs,
        const std::vector< real > &local_dual,
        const unsigned int face_number,
        const unsigned int boundary_id,
        dealii::hp::FEFaceValues<dim,dim> &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim> &fe_soln,
        const real penalty,
        std::vector<codi_JacobianComputationType> &rhs,
        codi_JacobianComputationType &dual_dot_residual)=0;
    
    virtual void assemble_boundary_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const std::vector<codi_HessianComputationType> &soln_coeffs,
        const std::vector<codi_HessianComputationType> &metric_coeffs,
        const std::vector< real > &local_dual,
        const unsigned int face_number,
        const unsigned int boundary_id,
        dealii::hp::FEFaceValues<dim,dim> &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim> &fe_soln,
        const real penalty,
        std::vector<codi_HessianComputationType> &rhs,
        codi_HessianComputationType &dual_dot_residual)=0;
    
    virtual void assemble_face_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        std::vector<double> &soln_int,
        std::vector<double> &soln_ext,
        std::vector<double> &metric_int,
        std::vector<double> &metric_ext,
        const std::vector< double > &dual_int,
        const std::vector< double > &dual_ext,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const real penalty,
        const dealii::Quadrature<dim-1> &face_quadrature,
        std::vector<double> &rhs_int,
        std::vector<double> &rhs_ext,
        double &dual_dot_residual,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)=0;
    
    virtual void assemble_face_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        std::vector<codi_JacobianComputationType> &soln_int,
        std::vector<codi_JacobianComputationType> &soln_ext,
        std::vector<codi_JacobianComputationType> &metric_int,
        std::vector<codi_JacobianComputationType> &metric_ext,
        const std::vector< double > &dual_int,
        const std::vector< double > &dual_ext,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const real penalty,
        const dealii::Quadrature<dim-1> &face_quadrature,
        std::vector<codi_JacobianComputationType> &rhs_int,
        std::vector<codi_JacobianComputationType> &rhs_ext,
        codi_JacobianComputationType &dual_dot_residual,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)=0;
    
    virtual void assemble_face_term_and_build_operators_ad(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        std::vector<codi_HessianComputationType> &soln_int,
        std::vector<codi_HessianComputationType> &soln_ext,
        std::vector<codi_HessianComputationType> &metric_int,
        std::vector<codi_HessianComputationType> &metric_ext,
        const std::vector< double > &dual_int,
        const std::vector< double > &dual_ext,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const real penalty,
        const dealii::Quadrature<dim-1> &face_quadrature,
        std::vector<codi_HessianComputationType> &rhs_int,
        std::vector<codi_HessianComputationType> &rhs_ext,
        codi_HessianComputationType &dual_dot_residual,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)=0;

    /// Returns the value from a CoDiPack variable.
    /** The recursive calling allows to retrieve nested CoDiPack types.
     */
    template <typename real2>
    double getValue(const real2 &x);
   
    /// Derivative indexing when only 1 cell is concerned.
    /// Derivatives are ordered such that x comes first with index 0, then w.
    /// If derivatives with respect to x are not needed, then derivatives
    /// with respect to w will start at index 0. This function is for a single
    /// cell's DoFs.
    void automatic_differentiation_indexing_1(
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
        const unsigned int n_soln_dofs, const unsigned int n_metric_dofs,
        unsigned int &w_start, unsigned int &w_end,
        unsigned int &x_start, unsigned int &x_end);
    
    /// Derivative indexing when 2 cells are concerned.
    /// Derivatives are ordered such that x comes first with index 0, then w.
    /// If derivatives with respect to x are not needed, then derivatives
    /// with respect to w will start at index 0. This function is for a single
    /// cell's DoFs.
    void automatic_differentiation_indexing_2(
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
        const unsigned int n_soln_dofs_int, const unsigned int n_soln_dofs_ext, const unsigned int n_metric_dofs,
        unsigned int &w_int_start, unsigned int &w_int_end, unsigned int &w_ext_start, unsigned int &w_ext_end,
        unsigned int &x_int_start, unsigned int &x_int_end, unsigned int &x_ext_start, unsigned int &x_ext_end);
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

    /// Lagrange basis used in strong form
    /** This is a collection of scalar Lagrange bases */
    const dealii::hp::FECollection<dim>  fe_collection_lagrange;

public:

    /// 1D Finite Element Collection for p-finite-element to represent the solution
    /** This is a collection of FESystems for 1D. */
    const dealii::hp::FECollection<1>    oneD_fe_collection;

    /// 1D Finite Element Collection for p-finite-element to represent the solution for a single state.
    /** This is a collection of FESystems for 1D. 
    * Since each state is represented by the same polynomial degree, for the RHS,
    * we only need to store the 1D basis functions for a single state.
    */
    const dealii::hp::FECollection<1>    oneD_fe_collection_1state;
    /// 1D collocated flux basis used in strong form
    /** This is a collection of collocated Lagrange bases for 1D.*/
    const dealii::hp::FECollection<1>  oneD_fe_collection_flux;
    /// 1D quadrature to generate Lagrange polynomials for the sake of flux interpolation.
    dealii::hp::QCollection<1>       oneD_quadrature_collection;
    /// 1D surface quadrature is always one single point for all poly degrees.
    dealii::QGauss<0>                oneD_face_quadrature;

    /// Finite Element Collection to represent the high-order grid
    /** This is a collection of FESystems.
     *  Unfortunately, deal.II doesn't have a working hp Mapping FE field.
     *  Therefore, every grid/cell will use the maximal polynomial mapping regardless of the solution order.
     */
    //const dealii::hp::FECollection<dim>    fe_collection_grid;
    //const dealii::FESystem<dim>    fe_grid;

    /// Degrees of freedom handler
    /*  Allows us to iterate over the finite elements' degrees of freedom.
     *  Note that since we are not using FESystem, we need to multiply
     *  the index by a factor of "nstate"
     *
     *  Must be defined after fe_dg since it is a subscriptor of fe_dg.
     *  Destructor are called in reverse order in which they appear in class definition.
     */
    dealii::DoFHandler<dim> dof_handler;

    /// High order grid that will provide the MappingFEField
    std::shared_ptr<HighOrderGrid<dim,real,MeshType>> high_order_grid;

    /// Sets the current time within DG to be used for unsteady source terms.
    void set_current_time(const real current_time_input);

    /// Computational time for assembling residual.
    double assemble_residual_time;

protected:
    /// The current time set in set_current_time()
    real current_time;
    /// Continuous distribution of artificial dissipation.
    const dealii::FE_Q<dim> fe_q_artificial_dissipation;

    /// Degrees of freedom handler for C0 artificial dissipation.
    dealii::DoFHandler<dim> dof_handler_artificial_dissipation;

    /// Artificial dissipation coefficients
    dealii::LinearAlgebra::distributed::Vector<double> artificial_dissipation_c0;

    /// Builds the necessary operators/fe values and assembles volume residual.
    virtual void assemble_volume_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis,
        OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume,
        dealii::hp::FEValues<dim,dim>                          &fe_values_collection_volume_lagrange,
        const dealii::FESystem<dim,dim>                        &current_fe_ref,
        dealii::Vector<real>                                   &local_rhs_int_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;

    /// Builds the necessary operators/fe values and assembles boundary residual.
    virtual void assemble_boundary_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     boundary_id,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &cell_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &metric_dof_indices,
        const unsigned int                                     poly_degree,
        const unsigned int                                     grid_degree,
        OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis,
        OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        const dealii::FESystem<dim,dim>                        &current_fe_ref,
        dealii::Vector<real>                                   &local_rhs_int_cell,
        std::vector<dealii::Tensor<1,dim,real>>                &local_auxiliary_RHS,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;

    /// Builds the necessary operators/fe values and assembles face residual.
    virtual void assemble_face_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     poly_degree_int,
        const unsigned int                                     poly_degree_ext,
        const unsigned int                                     grid_degree_int,
        const unsigned int                                     grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_ext,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &current_cell_rhs_aux,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;

    /// Builds the necessary operators/fe values and assembles subface residual.
    virtual void assemble_subface_term_and_build_operators(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        typename dealii::DoFHandler<dim>::active_cell_iterator neighbor_cell,
        const dealii::types::global_dof_index                  current_cell_index,
        const dealii::types::global_dof_index                  neighbor_cell_index,
        const unsigned int                                     iface,
        const unsigned int                                     neighbor_iface,
        const unsigned int                                     neighbor_i_subface,
        const real                                             penalty,
        const std::vector<dealii::types::global_dof_index>     &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &current_metric_dofs_indices,
        const std::vector<dealii::types::global_dof_index>     &neighbor_metric_dofs_indices,
        const unsigned int                                     poly_degree_int,
        const unsigned int                                     poly_degree_ext,
        const unsigned int                                     grid_degree_int,
        const unsigned int                                     grid_degree_ext,
        OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>              &soln_basis_ext,
        OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_int,
        OPERATOR::basis_functions<dim,2*dim,real>              &flux_basis_ext,
        OPERATOR::local_basis_stiffness<dim,2*dim,real>        &flux_basis_stiffness,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_int,
        OPERATOR::vol_projection_operator<dim,2*dim,real>      &soln_basis_projection_oper_ext,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_int,
        OPERATOR::metric_operators<real,dim,2*dim>             &metric_oper_ext,
        OPERATOR::mapping_shape_functions<dim,2*dim,real>      &mapping_basis,
        std::array<std::vector<real>,dim>                      &mapping_support_points,
        dealii::hp::FEFaceValues<dim,dim>                      &fe_values_collection_face_int,
        dealii::hp::FESubfaceValues<dim,dim>                   &fe_values_collection_subface,
        dealii::Vector<real>                                   &current_cell_rhs,
        dealii::Vector<real>                                   &neighbor_cell_rhs,
        std::vector<dealii::Tensor<1,dim,real>>                &current_cell_rhs_aux,
        dealii::LinearAlgebra::distributed::Vector<double>     &rhs,
        std::array<dealii::LinearAlgebra::distributed::Vector<double>,dim> &rhs_aux,
        const bool                                             compute_auxiliary_right_hand_side,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;

protected:
    /// Evaluate the integral over the cell volume and the specified derivatives.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. */
    virtual void assemble_volume_term_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::FEValues<dim,dim> &,//fe_values_vol,
        const dealii::FESystem<dim,dim> &fe,
        const dealii::Quadrature<dim> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        dealii::Vector<real> &local_rhs_cell,
        const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;
    /// Evaluate the integral over the cell edges that are on domain boundaries and the specified derivatives.
    /** Compute both the right-hand side and the corresponding block of dRdW, dRdX, and/or d2R. */
    virtual void assemble_boundary_term_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const unsigned int face_number,
        const unsigned int boundary_id,
        const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe,
        const dealii::Quadrature<dim-1> &quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
        dealii::Vector<real> &local_rhs_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;
    /// Evaluate the integral over the internal cell edges and its specified derivatives.
    /** Compute both the right-hand side and the block of the Jacobian.
     *  This adds the contribution to both cell's residual and effectively
     *  computes 4 block contributions to dRdX blocks. */
    virtual void assemble_face_term_derivatives(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::types::global_dof_index neighbor_cell_index,
        const std::pair<unsigned int, int> face_subface_int,
        const std::pair<unsigned int, int> face_subface_ext,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_int,
        const typename dealii::QProjector<dim>::DataSetDescriptor face_data_set_ext,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_int,
        const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_ext,
        const real penalty,
        const dealii::FESystem<dim,dim> &fe_int,
        const dealii::FESystem<dim,dim> &fe_ext,
        const dealii::Quadrature<dim-1> &face_quadrature,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
        const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
        dealii::Vector<real>          &local_rhs_int_cell,
        dealii::Vector<real>          &local_rhs_ext_cell,
        const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R) = 0;

    /// Evaluate the integral over the cell volume
    virtual void assemble_volume_term_explicit(
        typename dealii::DoFHandler<dim>::active_cell_iterator cell,
        const dealii::types::global_dof_index current_cell_index,
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
        const unsigned int poly_degree,
        const unsigned int grid_degree,
        dealii::Vector<real> &current_cell_rhs,
        const dealii::FEValues<dim,dim> &fe_values_lagrange) = 0;

    /// Update flags needed at volume points.
    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values
        | dealii::update_inverse_jacobians;
    /// Update flags needed at face points.
    const dealii::UpdateFlags face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors
        | dealii::update_jacobians;
    /// Update flags needed at neighbor' face points.
    /** NOTE: With hp-adaptation, might need to query neighbor's quadrature points depending on the order of the cells. */
    const dealii::UpdateFlags neighbor_face_update_flags = dealii::update_values | dealii::update_gradients | dealii::update_quadrature_points | dealii::update_JxW_values;


public:
    /// Allocates the auxiliary equations' variables and right hand side (primarily for Strong form diffusive)
    void allocate_auxiliary_equation ();

    /// Asembles the auxiliary equations' residuals and solves.
    virtual void assemble_auxiliary_residual () = 0;

    /// Allocate the dual vector for optimization.
    /** Currently only used in weak form.
    */
    virtual void allocate_dual_vector () = 0;

protected:
    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
private:

    /** Evaluate the average penalty term at the face.
     *  For a cell with solution of degree p, and Hausdorff measure h,
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

public:
    /// Flag to freeze artificial dissipation.
    bool freeze_artificial_dissipation;
    /// Stores maximum artificial dissipation while assembling the residual.
    double max_artificial_dissipation_coeff;
    /// Update discontinuity sensor.
    void update_artificial_dissipation_discontinuity_sensor();
    /// Allocate the necessary variables declared in src/physics/model.h
    virtual void allocate_model_variables() = 0;
    /// Update the necessary variables declared in src/physics/model.h
    virtual void update_model_variables() = 0;
    /// Flag for using the auxiliary equation
    bool use_auxiliary_eq;
    /// Set use_auxiliary_eq flag
    virtual void set_use_auxiliary_eq() = 0;

}; // end of DGBase class

} // PHiLiP namespace

#endif
