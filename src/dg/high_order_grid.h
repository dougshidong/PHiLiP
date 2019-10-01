#ifndef __HIGHORDERGRID_H__
#define __HIGHORDERGRID_H__

#include <deal.II/grid/tria.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe_field.h> 

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/lac/vector.h>

#include "parameters/all_parameters.h"

namespace PHiLiP {

/** This class is used to generate and control the high-order nodes given a Triangulation and a Manifold.
 *  This will especially be useful when performing shape optimization, where the surface and volume 
 *  nodes will need to be displaced.
 */
//template <int dim, typename real, typename VectorType , typename DoFHandlerType>
template <int dim = PHILIP_DIM, typename real = double, typename VectorType = dealii::LinearAlgebra::distributed::Vector<double>, typename DoFHandlerType = dealii::DoFHandler<PHILIP_DIM>>
class HighOrderGrid
{
public:
    /// Principal constructor that will call delegated constructor.
    HighOrderGrid(const Parameters::AllParameters *const parameters_input, const unsigned int max_degree, dealii::Triangulation<dim> *const triangulation_input);

    /// Needed after the mesh is refined
    void allocate();

    dealii::MappingFEField<dim,dim,VectorType,DoFHandlerType> get_MappingFEField();

    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters

    /// Maximum degree of the grid.
    const unsigned int max_degree;

    dealii::Triangulation<dim> *triangulation; ///< Mesh

    /// Get evaluate high-order grid from Triangulation and Manifold


    /// Degrees of freedom handler for the high-order grid
    dealii::DoFHandler<dim> dof_handler_grid;

    /// Current nodal coefficients of the high-order grid.
    dealii::LinearAlgebra::distributed::Vector<double> nodes;

    /// List of surface points
    dealii::LinearAlgebra::distributed::Vector<double> surface_nodes;


    /// RBF mesh deformation
    void deform_mesh(dealii::LinearAlgebra::distributed::Vector<double> surface_displacements);

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



    /// Using system of FE_Q to represent the grid
    const dealii::FE_Q<dim>     fe_q;
    const dealii::FESystem<dim> fe_system;

    dealii::IndexSet locally_owned_dofs_grid; ///< Locally own degrees of freedom for the grid
    dealii::IndexSet ghost_dofs_grid; ///< Locally relevant ghost degrees of freedom for the grid
    dealii::IndexSet locally_relevant_dofs_grid; ///< Union of locally owned degrees of freedom and relevant ghost degrees of freedom for the grid
protected:

    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

} // namespace PHiLiP

#endif
