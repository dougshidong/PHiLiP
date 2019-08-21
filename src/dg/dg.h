#ifndef __DISCONTINUOUSGALERKIN_H__
#define __DISCONTINUOUSGALERKIN_H__

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>


#include <deal.II/dofs/dof_handler.h>

#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/fe_values.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>

#include <Sacado.hpp>

#include "physics/physics.h"
#include "numerical_flux/numerical_flux.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {

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
public:
    /// Number of state variables.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const int nstate;

    /// Maximum degree used for p-refinement
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.  */
    const unsigned int max_degree;

    /// Pointer to all parameters
    const Parameters::AllParameters *const all_parameters;

    /// Principal constructor.
    /** Will initialize mapping, fe_dg, all_parameters, volume_quadrature, and face_quadrature
     *  from DGBase. The it will new some FEValues that will be used to retrieve the
     *  finite element values at physical locations.
     */
    DGBase(const int nstate_input, const Parameters::AllParameters *const parameters_input, const unsigned int max_degree);

    /// Delegated constructor that initializes collections
    /** Since a function is used to generate multiple different objects, a delegated
     *  constructor is used to unwrap the tuple and initialize the collections */
    DGBase( const int nstate_input,
            const Parameters::AllParameters *const parameters_input,
            const unsigned int max_degree_input,
            const std::tuple< dealii::hp::MappingCollection<dim>, dealii::hp::FECollection<dim>,
                              dealii::hp::QCollection<dim>, dealii::hp::QCollection<dim-1>, dealii::hp::QCollection<1>,
                              dealii::hp::FECollection<dim> > collection_tuple);

    virtual ~DGBase(); ///< Destructor.

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Triangulation = dealii::Triangulation<dim>;
#else
    using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif
    Triangulation *triangulation; ///< Mesh
    /// Sets the triangulation for 2D and 3D. Should be done before allocate system
    void set_triangulation(Triangulation *triangulation_input);

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

    unsigned int n_dofs() const; /// Number of degrees of freedom


    /// Degrees of freedom handler
    /*  Allows us to iterate over the finite elements' degrees of freedom.
     *  Note that since we are not using FESystem, we need to multiply
     *  the index by a factor of "nstate"
     *
     *  Must be defined after fe_dg since it is a subscriptor of fe_dg.
     *  Destructor are called in reverse order in which they appear in class definition. 
     */ 
    dealii::hp::DoFHandler<dim> dof_handler;

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
    /// Current modal coefficients of the solution
    /** Note that the current processor has read-access to all locally_relevant_dofs
     *  and has write-access to all locally_owned_dofs
     */
    dealii::LinearAlgebra::distributed::Vector<double> solution;

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

    /// Finite Element Collection for p-finite-element
    /** This is a collection of FESystems */
    const dealii::hp::FECollection<dim>    fe_collection;

    /// Mapping is currently MappingQ.
    /**  Refer to deal.II documentation for the various mapping types */
    //const dealii::MappingQ<dim> mapping;
    const dealii::hp::MappingCollection<dim> mapping_collection;

    dealii::hp::QCollection<dim>     volume_quadrature_collection;
    dealii::hp::QCollection<dim-1>   face_quadrature_collection;
    dealii::hp::QCollection<1>       oned_quadrature_collection;



protected:

    std::tuple< dealii::hp::MappingCollection<dim>, dealii::hp::FECollection<dim>,
                dealii::hp::QCollection<dim>, dealii::hp::QCollection<dim-1>, dealii::hp::QCollection<1>,
                dealii::hp::FECollection<dim> >
                create_collection_tuple(const unsigned int max_degree, const int nstate) const;

    /// Evaluate the integral over the cell volume
    virtual void assemble_volume_terms_implicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs) = 0;
    /// Evaluate the integral over the cell edges that are on domain boundaries
    virtual void assemble_boundary_term_implicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs) = 0;
    /// Evaluate the integral over the internal cell edges
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
        dealii::Vector<real> &current_cell_rhs) = 0;
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

    /// Create mapping collection for initializer list
    dealii::hp::MappingCollection<dim> create_mapping_collection(const unsigned int max_degree) const;

    /// Create FECollection for initializer list
    dealii::hp::FECollection<dim> create_fe_collection(const unsigned int max_degree) const;


    const dealii::UpdateFlags volume_update_flags = dealii::update_values | dealii::update_gradients |
                                                    dealii::update_quadrature_points | dealii::update_JxW_values;
    const dealii::UpdateFlags face_update_flags =   dealii::update_values | dealii::update_gradients |
                                                    dealii::update_quadrature_points | dealii::update_JxW_values | dealii::update_normal_vectors;
    const dealii::UpdateFlags neighbor_face_update_flags = dealii::update_values | dealii::update_gradients;

    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume;
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int;
    dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext;
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface;

    /// Lagrange basis used in strong form
    /** This is a collection of scalar Lagrange bases */
    const dealii::hp::FECollection<dim>  fe_collection_lagrange;
    dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange;

    MPI_Comm mpi_communicator;
    dealii::ConditionalOStream pcout;

}; // end of DGBase class

/// DGWeak class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
template <int dim, int nstate, typename real>
class DGWeak : public DGBase<dim, real>
{
public:
    /// Constructor
    DGWeak(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree);

    /// Destructor
    ~DGWeak();

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
    void assemble_volume_terms_implicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    void assemble_boundary_term_implicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
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
        dealii::Vector<real> &current_cell_rhs);
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

    using DGBase<dim,real>::mpi_communicator;
    using DGBase<dim,real>::pcout;

}; // end of DGWeak class

/// DGStrong class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 */
template <int dim, int nstate, typename real>
class DGStrong : public DGBase<dim, real>
{
public:
    /// Constructor
    DGStrong(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree);

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
    void assemble_volume_terms_implicit(
        const dealii::FEValues<dim,dim> &fe_values_volume,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    void assemble_boundary_term_implicit(
        const unsigned int boundary_id,
        const dealii::FEFaceValues<dim,dim> &fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
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
        dealii::Vector<real> &current_cell_rhs);
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

    using DGBase<dim,real>::mpi_communicator;
    using DGBase<dim,real>::pcout;

}; // end of DGStrong class

/// This class creates a new DGBase object
/** This allows the DGBase to not be templated on the number of state variables
  * while allowing DG to be template on the number of state variables
 */
template <int dim, typename real>
class DGFactory
{
public:
    /// Creates a derived object DG, but returns it as DGBase.
    /** That way, the called is agnostic to the number of state variables
     */
    static std::shared_ptr< DGBase<dim,real> >
        create_discontinuous_galerkin(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree);
};

} // PHiLiP namespace

#endif
