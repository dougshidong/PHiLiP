#ifndef __GRID_REFINEMENT_H__
#define __GRID_REFINEMENT_H__

#include <deal.II/grid/tria.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters_grid_refinement.h"

#include "dg/dg.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "physics/physics.h"

#include "grid_refinement/field.h"

namespace PHiLiP {

namespace GridRefinement {

/// Base Grid Refinement Class
/** This class provides access to the basic refinement control methods implemented for 
  * uniform, fixed-fraction and continuous style methods in the associated *.cpp files.
  * Although this class contains no refinement functionality of its own, the virtual functions
  * implemented from here provide a uniform interface for adapting the grid based on a variety
  * of h-, p- and hp- style refinement techniques with indicators from the exact error (manufactured
  * solution), feature-based (generalization of hessian based methods for high order), the local
  * residual distribution on the fine grid and goal-oriented adjoint-based techniques.
  * 
  * Additionally, this class contains functionality for writing a description of the current 
  * grid refinement object to a .vtk file with additional refinement information passed
  * from the subclass implementations.
  * 
  * See the related parameter object PHiLiP::Parameters::GridRefinementParam for more information about
  * the various options and controls availible.
  * 
  * Note: This class templated on the mesh type as anisotropic fixed-fraction splitting is 
  *       not availible in parralel at this time. 
  */
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinementBase
{
public:
    /// Deleted default constructor
    GridRefinementBase() = delete;

    /// Constructor. Stores the adjoint object, physics and parameters
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    /// Constructor. Storers the dg object, physics, functional and parameters.
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input);

    /// Constructor. Stores the dg object, physics and parameters
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    /// Constructor. Stores the dg object and parameters
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg_input);

protected:
    /// Delegated constructor which handles the various optional inputs and setup.
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >    adj_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input);

public:
    /// Perform call to the grid refinement object of choice
    /** This will automatically select the proper subclass, error indicator
      * and various refinement types based on the grid refinement parameters
      * passed at setup to the grid refinement factor class.
      * 
      * See subclass functions for details of refinement types.
      */
    virtual void refine_grid() = 0;

public:
    /// Write information about the grid refinement step to a .vtk file
    /** Includes various information about the mesh, solution, error indicators,
      * target refinements (both h- and p-), functional solution, physics, etc.
      * 
      * also provides interface for subclasses to output additional visualization fields.
      */
    void output_results_vtk(const unsigned int iref);

protected:
    // helper output classes

    /// Output refinement results related to the DG object
    void output_results_vtk_dg(
        dealii::DataOut<dim, dealii::DoFHandler<dim>> &    data_out,
        std::shared_ptr< dealii::DataPostprocessor<dim> > &post_processor,
        dealii::Vector<float> &                            subdomain,
        std::vector<unsigned int> &                        active_fe_indices,
        dealii::Vector<double> &                           cell_poly_degree,
        std::vector<std::string> &                         residual_names);

    /// Output refinement results related to the functional object
    void output_results_vtk_functional(
        dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out);

    /// Output refinement results related to the problem physics
    void output_results_vtk_physics(
        dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out);

    /// Output refinement results related to the adjoint object
    void output_results_vtk_adjoint(
        dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out,
        std::vector<std::string> &                     dIdw_names_coarse,
        std::vector<std::string> &                     adjoint_names_coarse,
        std::vector<std::string> &                     dIdw_names_fine,
        std::vector<std::string> &                     adjoint_names_fine);
    
    /// Output refinement results related to the solution error
    void output_results_vtk_error(
        dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out,
        dealii::Vector<real> &                         l2_error_vec);  

protected:
    /// Output refinement method dependent results
    /** This class is overridden in the subclasses with any additional visualization fields.
      */ 
    virtual std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() = 0; 

    /// Grid refinement parameters
    PHiLiP::Parameters::GridRefinementParam grid_refinement_param;

    using ErrorIndicatorEnum = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    // Error indicator type
    ErrorIndicatorEnum error_indicator_type;

    /// Adjoint object (if provided to factory)
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> > adjoint;

    /// Functional object (if provided to factory, directly or indirectly)
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional;

    /// Discontinuous Galerkin object
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg;

    // high order grid, not a pointer 
    // so needs to be manipulated through dg->high_order_grid
    // HighOrderGrid<dim,real> high_order_grid
    
    /// Problem physics (if provided to factory, directly or indirectly)
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics;

    // triangulation
    // dealii::Triangulation<dim, dim> &tria;
    // Triangulation &tria;

    /// Triangulation object of templated mesh type
    /** Note: anisotropic, p- type and other refinements may not work in all cases.
      */ 
    std::shared_ptr<MeshType> tria;

    /// Internal refinement steps iteration counter
    unsigned int iteration;

    MPI_Comm mpi_communicator; ///< MPI communicator
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

    /// update flags needed at volume points
    const dealii::UpdateFlags volume_update_flags = 
        dealii::update_values | 
        dealii::update_gradients | 
        dealii::update_quadrature_points | 
        dealii::update_JxW_values | 
        dealii::update_inverse_jacobians;

    /// update flags needed at face points
    const dealii::UpdateFlags face_update_flags = 
        dealii::update_values | 
        dealii::update_gradients | 
        dealii::update_quadrature_points | 
        dealii::update_JxW_values | 
        dealii::update_normal_vectors | 
        dealii::update_jacobians;
    
    /// update flags needed at neighbor's face points
    const dealii::UpdateFlags neighbor_face_update_flags = 
        dealii::update_values | 
        dealii::update_gradients | 
        dealii::update_quadrature_points | 
        dealii::update_JxW_values;
};

/// Grid Refinement Class Factory
/** Contains various factory functions for generating different grid refinement classes 
  * dependant on what input information is provided. For example, certain grid refinement
  * operations will need access to additional infomation such as the adjoint, or problem
  * physics in addition to discontinuous galerkin formulation. If a more stringent initial
  * call is made but only a simple refinement operation is needed, the factory calls will 
  * chain to provide the proper output. 
  * 
  * The choice of refinement object is made based on the selected refinement_method and error_indicator
  * from the PHiLiP::Parameters::GridRefinementParam object needed for each call.
  * 
  * Note: This class templated on the mesh type as anisotropic fixed-fraction splitting is 
  *       not availible in parralel at this time. 
  */ 
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinementFactory
{
public:

    /// Construct grid refinement class based on adjoint and physics
    /** Provides access to all refinement types. Needs to be called for adjoint_based
      * error indicators. Adjoint object also provides access to dg and functional objects.
      */ 
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    /// Construct grid refinement class based on dg, physics and functional
    /** Provides access to non-adjoint based method. However, allows the functional object
      * to still be passed to the grid refinement class for tracking the goal-oriented error
      * convergence when working with feature-based refinement types (if an adjoint-object 
      * itself is not availible).
      */ 
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                            gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional);

    /// Construct grid refinement class based on dg and physics
    /** Provides access to feature-based (hessian_based) error indicators and exact error-based
      * refinement methods using the manufactured solution from the physics object. 
      */
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics);

    /// Construct grid refinement class based on dg only
    /** Provides access to basic uniform refinement methods and residual based refinement
      * methods (not yet implemented).
      */ 
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg);

};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_H__
