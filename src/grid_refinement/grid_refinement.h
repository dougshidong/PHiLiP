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

// central class of the grid_refinement, controls refinements
#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinementBase
{
public:
    // deleting the default constructor
    GridRefinementBase() = delete;

    // constructor stores the parameters
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input);

    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                gr_param_input,
        // PHiLiP::Parameters::AllParameters const *const param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg_input);

protected:
    // delegated constructor
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >    adj_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input);

public:
    // refine_grid is the main function
    virtual void refine_grid() = 0;

public:
    // main output class
    void output_results_vtk(const unsigned int iref);

protected:
    // helper output classes
    void output_results_vtk_dg(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::shared_ptr< dealii::DataPostprocessor<dim> > &post_processor,
        dealii::Vector<float> &                            subdomain,
        std::vector<unsigned int> &                        active_fe_indices,
        dealii::Vector<double> &                           cell_poly_degree,
        std::vector<std::string> &                         residual_names);

    void output_results_vtk_functional(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out);

    void output_results_vtk_physics(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out);

    void output_results_vtk_adjoint(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::vector<std::string> &                         dIdw_names_coarse,
        std::vector<std::string> &                         adjoint_names_coarse,
        std::vector<std::string> &                         dIdw_names_fine,
        std::vector<std::string> &                         adjoint_names_fine);
    
    void output_results_vtk_error(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        dealii::Vector<real> &                             l2_error_vec);  

protected:
    // refinement method dependent outputs (to be overrided in derived classes)
    virtual std::vector< std::pair<dealii::Vector<real>, std::string> > output_results_vtk_method() = 0; 

    // parameters
    PHiLiP::Parameters::GridRefinementParam grid_refinement_param;

    // indicator type
    using ErrorIndicatorEnum = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    ErrorIndicatorEnum error_indicator_type;

    // adj
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> > adjoint;

    // Functional
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional;

    // dg
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg;

    // high order grid, not a pointer 
    // so needs to be manipulated through dg->high_order_grid
    // HighOrderGrid<dim,real> high_order_grid
    
    // physics
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics;

    // triangulation
    // dealii::Triangulation<dim, dim> &tria;
    // Triangulation &tria;
    MeshType *const tria;

    // iteration counter
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
