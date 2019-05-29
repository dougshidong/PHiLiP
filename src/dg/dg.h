#ifndef __DISCONTINUOUSGALERKIN_H__
#define __DISCONTINUOUSGALERKIN_H__

#include <deal.II/base/parameter_handler.h>

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/mapping_q1.h> // Might need mapping_q
#include <deal.II/fe/mapping_q.h> // Might need mapping_q

#include <deal.II/dofs/dof_handler.h>


#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <Sacado.hpp>

#include "physics/physics.h"
#include "numerical_flux/numerical_flux.h"
#include "parameters/all_parameters.h"


namespace PHiLiP {

/// DGBase is independent of the number of state variables.
/**  This base class allows the use of arrays to efficiently allocate the data structures
  *  through std::array in the derived class DG.
  *  This class is the one being returned by the DGFactory and is the main
  *  interface for a user to call its main functions such as "assemble_system".
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
  *  Note that it does not currently handle nonlinear or time-accurate problems
  *  since mass matrices have not been used/verified yet.
  */
template <int dim, typename real>
class DGBase 
{
public:
    /// Number of state variables.
    /** This is known through the constructor parameters.
     *  DGBase cannot use nstate as a compile-time known.
     */
    const int nstate;

    /// Constructor. Deleted the default constructor since it should not be used
    DGBase () = delete;
    /// Principal constructor.
    /** Will initialize mapping, fe_dg, all_parameters, quadrature, and face_quadrature
     *  from DGBase. The it will new some FEValues that will be used to retrieve the
     *  finite element values at physical locations.
     */
    DGBase(
        const int nstate_input,
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree);

    virtual ~DGBase(); ///< Destructor. Also calls delete_fe_values().
    void delete_fe_values (); ///< Delete the new-ed pointers to FEValues

    /// Sets the triangulation. Should be done before allocate system
    void set_triangulation(dealii::Triangulation<dim> *triangulation_input)
    { triangulation = triangulation_input; } ;

    virtual void allocate_system () = 0; ///< Virtual function defined in DG
    virtual void assemble_system () = 0; ///< Virtual function defined in DG

    /// Allocates and evaluates the inverse mass matrices for the entire grid
    /*  Although straightforward, this has not been tested yet.
     *  Will be required for accurate time-stepping or nonlinear problems
     */
    void evaluate_inverse_mass_matrices ();
    /// Vector of inverse mass matrices.
    /** Contains the inverse mass matrices of each cell at the quadrature points.  */
    std::vector<dealii::FullMatrix<real>> inv_mass_matrix;

    double get_residual_l2norm (); ///< Returns the L2-norm of the right_hand_side vector

    dealii::Triangulation<dim>   *triangulation; ///< Mesh


    /// Sparsity pattern used on the system_matrix
    /** Not sure we need to store it.  */
    dealii::DynamicSparsityPattern sparsity_pattern;

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
    dealii::Vector<real> right_hand_side;

    dealii::Vector<real> solution; ///< Current modal coefficients of the solution

    void output_results (const unsigned int ith_grid); ///< Output solution

    /// Mapping is currently MappingQ.
    /*  Refer to deal.II documentation for the various mapping types */
    const dealii::MappingQ<dim> mapping;

    /// Lagrange polynomial basis
    /** Refer to deal.II documentation for the various polynomial types
     *  Note that only tensor-product polynomials recover optimal convergence
     *  since the mapping from the reference to physical element is a bilnear mapping.
     *
     *  As a result, FE_DGP does not give optimal convergence orders.
     *  See 
     *  [discussion on deal.II group forum]
     *  (https://groups.google.com/d/msg/dealii/f9NzCp8dnyU/aAdO6I9JCwAJ)
     *  .
     */
    const dealii::FE_DGQ<dim> fe_dg;
    //const dealii::FE_DGQLegendre<dim> fe_dg;

    /// Finite Element System used for vector-valued problems
    /** Note that we will use the same set of polynomials for all state equations
     *  therefore, FESystem is only used for the ease of obtaining sizes and 
     *  global indexing.
     *
     *  When evaluating the function values, we will still be using fe_dg
     */
    const dealii::FESystem<dim,dim> fe_system;

    /// Pointer to all parameters
    const Parameters::AllParameters *const all_parameters;


    /// Degrees of freedom handler
    /*  Allows us to iterate over the finite elements' degrees of freedom.
     *  Note that since we are not using FESystem, we need to multiply
     *  the index by a factor of "nstate"
     *
     *  Must be defined after fe_dg since it is a subscriptor of fe_dg.
     *  Destructor are called in reverse order in which they appear in class definition. 
     */ 
    dealii::DoFHandler<dim> dof_handler;


protected:

    // QGauss is Gauss-Legendre quadrature nodes
    const dealii::QGauss<dim>   quadrature;
    const dealii::QGauss<dim-1> face_quadrature;

    dealii::FEValues<dim,dim>         *fe_values_cell;
    dealii::FEFaceValues<dim,dim>     *fe_values_face_int;
    dealii::FESubfaceValues<dim,dim>  *fe_values_subface_int;
    dealii::FEFaceValues<dim,dim>     *fe_values_face_ext;
    /// Main loop of the DGBase class.
    /** It loops over all the cells, evaluates the volume contributions,
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
    //virtual void allocate_system_implicit () = 0;
    //virtual void assemble_system_implicit () = 0;

}; // end of DG class

/// DG class templated on the number of state variables
/*  Contains the functions that need to be templated on the number of state variables.
 *  
 *  Also defines the main loop of the DG class which is assemble_system
 */
template <int dim, int nstate, typename real>
class DG : public DGBase<dim, real>
{
public:
    /// Constructor
    DG(
        const Parameters::AllParameters *const parameters_input, 
        const unsigned int degree);

    /// Destructor
    ~DG();

private:
    /// Contains the physics of the PDE
    Physics::PhysicsBase<dim, nstate, Sacado::Fad::DFad<real> >  *pde_physics;
    /// Convective numerical flux
    NumericalFlux::NumericalFluxConvective<dim, nstate, Sacado::Fad::DFad<real> > *conv_num_flux;
    /// Dissipative numerical flux
    NumericalFlux::NumericalFluxDissipative<dim, nstate, Sacado::Fad::DFad<real> > *diss_num_flux;



    /// Allocates the system.
    /** Must be done after setting the mesh and before assembling the system.
     */
    void allocate_system ();

    /// Main loop of the DG class.
    /** It loops over all the cells, evaluates the volume contributions,
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
    void assemble_system ();


    /// Evaluate the integral over the cell volume
    void assemble_cell_terms_implicit(
        const dealii::FEValues<dim,dim> *fe_values_cell,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the cell edges that are on domain boundaries
    void assemble_boundary_term_implicit(
        const dealii::FEFaceValues<dim,dim> *fe_values_face_int,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        dealii::Vector<real> &current_cell_rhs);
    /// Evaluate the integral over the internal cell edges
    void assemble_face_term_implicit(
        const dealii::FEValuesBase<dim,dim>     *fe_values_face_int,
        const dealii::FEFaceValues<dim,dim>     *fe_values_face_ext,
        const real penalty,
        const std::vector<dealii::types::global_dof_index> &current_dofs_indices,
        const std::vector<dealii::types::global_dof_index> &neighbor_dofs_indices,
        dealii::Vector<real>          &current_cell_rhs,
        dealii::Vector<real>          &neighbor_cell_rhs);

}; // end of DG class

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
