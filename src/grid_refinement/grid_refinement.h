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
    virtual void refine_grid()    = 0;

protected:
    // refine grid functions to be called
    virtual void refine_grid_h()  = 0;
    virtual void refine_grid_p()  = 0;
    virtual void refine_grid_hp() = 0;

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

public:
    // setting the size of the array used for referencing values in output_results_vtk_method 
    const static unsigned int MAX_METHOD_VEC = 10;

protected:
    // refinement method dependent outputs (to be overrided in derived classes)
    virtual void output_results_vtk_method(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec) = 0; 

    // parameters
    PHiLiP::Parameters::GridRefinementParam grid_refinement_param;

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

#if PHILIP_DIM==1
template <int dim, int nstate, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, int nstate, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class GridRefinementFactory
{
public:
    // different factory calls have access to different Grid refinements
    // adjoint (dg + functional)
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    // dg + physics + Functional
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                            gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional);

    // dg + physics
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics);

    // dg 
    static std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg);

};

} // namespace GridRefinement

} // namespace PHiLiP

#endif // __GRID_REFINEMENT_H__
