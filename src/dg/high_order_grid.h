#ifndef __HIGHORDERGRID_H__
#define __HIGHORDERGRID_H__

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/distributed/solution_transfer.h>

#include <deal.II/lac/vector.h>

#include "parameters/all_parameters.h"

namespace PHiLiP {

/** This HighOrderGrid class basically contains all the different part necessary to generate
 *  a dealii::MappingFEField that corresponds to the current Triangulation and attached Manifold.
 *  Once the high order grid is generated, the mesh can be deformed by assigning different values to the
 *  nodes vector. The dof_handler_grid is used to access and loop through those nodes.
 *  This will especially be useful when performing shape optimization, where the surface and volume 
 *  nodes will need to be displaced.
 *  Note that there are a lot of pre-processor statements, and that is because the SolutionTransfer class
 *  and the Vector class act quite differently between serial and parallel implementation. Hopefully,
 *  deal.II will change this one day such that we have one interface for both.
 */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim = PHILIP_DIM, typename real = double, typename VectorType = dealii::Vector<double>, typename DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>>
#else
template <int dim = PHILIP_DIM, typename real = double, typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>, typename DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>>
#endif
class HighOrderGrid
{
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    using Vector = dealii::Vector<double>;
    using SolutionTransfer = dealii::SolutionTransfer<dim, dealii::Vector<double>, dealii::DoFHandler<dim>>;
#else
    using Vector = dealii::LinearAlgebra::distributed::Vector<double>;
    using SolutionTransfer = dealii::parallel::distributed::SolutionTransfer<dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::DoFHandler<dim>>;
#endif
public:
    /// Principal constructor that will call delegated constructor.
    HighOrderGrid(const Parameters::AllParameters *const parameters_input, const unsigned int max_degree, dealii::Triangulation<dim> *const triangulation_input);

    /// Needed to allocate the correct number of nodes when initializing and after the mesh is refined
    void allocate();

    /// Return a MappingFEField that corresponds to the current node locations
    dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> get_MappingFEField();

    /// Return a MappingFEField that corresponds to the current node locations
    void update_MappingFEField();

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Maximum degree of the geometry polynomial representing the grid.
    const unsigned int max_degree;

    dealii::Triangulation<dim> *triangulation; ///< Mesh

    /// Degrees of freedom handler for the high-order grid
    dealii::DoFHandler<dim> dof_handler_grid;

    /// Current nodal coefficients of the high-order grid.
    Vector nodes;

    /// List of surface points
    Vector surface_nodes;

    /// RBF mesh deformation  -  To be done
    void deform_mesh(Vector surface_displacements);

    /// Evaluate cell metric Jacobian
    /** The metric Jacobian is given by the gradient of the physical location
     *  with respect to the reference locations
     *  \f[ J_{ij} = \frac{
     *      (  \mathbf{F}_{conv}( u ) 
     *          + \mathbf{F}_{diss}( u, \boldsymbol{\nabla}(u) )
     *      = s(\mathbf{x})
     *  \f]
     */ 
    //dealii::Tensor<2,dim,real> cell_jacobian (const typename dealii::Triangulation<dim,spacedim>::cell_iterator &cell, const dealii::Point<dim> &point) const override
    //{
    //}
    //


    /** Prepares the solution transfer such that the curved refined grid is on top of the curved coarse grid.
     *  This function needs to be called before Triangulation::execute_coarsening_and_refinement() or Triangulation::refine_global()
     */
    void prepare_for_coarsening_and_refinement();
    /** Transfers the coarse curved curve onto the fine curved grid.
     *  This function needs to be called after Triangulation::execute_coarsening_and_refinement() or Triangulation::refine_global()
     */
    void execute_coarsening_and_refinement();

    /// Use Lagrange polynomial to represent the spatial location.
    const dealii::FE_Q<dim>     fe_q;
    /// Using system of polynomials to represent the x, y, and z directions.
    const dealii::FESystem<dim> fe_system;


    /** MappingFEField that will provide the polynomial-based grid.
     *  It is a shared smart pointer because the constructor requires the dof_handler_grid to be properly initialized.
     *  See discussion in the following thread:
     *  https://stackoverflow.com/questions/7557153/defining-an-object-without-calling-its-constructor-in-c
     */
    std::shared_ptr<dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType>> mapping_fe_field;

    dealii::IndexSet locally_owned_dofs_grid; ///< Locally own degrees of freedom for the grid
    dealii::IndexSet ghost_dofs_grid; ///< Locally relevant ghost degrees of freedom for the grid
    dealii::IndexSet locally_relevant_dofs_grid; ///< Union of locally owned degrees of freedom and relevant ghost degrees of freedom for the grid
protected:

    /// Used for the SolutionTransfer when performing grid adaptation.
    Vector old_nodes;

    /** Transfers the coarse curved curve onto the fine curved grid.
     *  Used in prepare_for_coarsening_and_refinement() and execute_coarsening_and_refinement()
     */
    SolutionTransfer solution_transfer;

    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

} // namespace PHiLiP

#endif
