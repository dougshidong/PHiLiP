#ifndef __GRID_REFINEMENT_H__
#define __GRID_REFINEMENT_H__

#include <deal.II/grid/tria.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters_grid_refinement.h"

#include "dg/dg.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "physics/physics.h"


namespace PHiLiP {

namespace GridRefinement {

// central class of the grid_refinement, controls refinements
template <int dim, int nstate, typename real>
class GridRefinementBase
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
    // deleting the default constructor
    GridRefinementBase() = delete;

    // constructor stores the parameters
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input);

    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam        gr_param_input,
        // PHiLiP::Parameters::AllParameters const *const param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg_input);

protected:
    // delegated constructor
    GridRefinementBase(
        PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

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
    const static unsigned int MAX_METHOD_VEC = 4;

protected:
    // refinement method dependent outputs (to be overrided in derived classes)
    virtual void output_results_vtk_method(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec) = 0; 

    // parameters
    PHiLiP::Parameters::GridRefinementParam grid_refinement_param;

    // different things needed depending on the choice of refinement
    // these could be held here with nullptr or in the base class
    // if I want these internal then this needs to be templated on
    // template <int dim, int nstate, typename real>

    // adj
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> > adjoint;

    // Functional
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real> > functional;

    // dg
    std::shared_ptr< PHiLiP::DGBase<dim, real> > dg;

    // high order grid, not a pointer 
    // so needs to be manipulated through dg->high_order_grid
    // fix this at some point
    // HighOrderGrid<dim,real> high_order_grid
    
    // physics
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics;

    // triangulation
    // dealii::Triangulation<dim, dim> &tria;
    // Triangulation &tria;
    Triangulation *const tria;

    // iteration counter
    int iteration;

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

template <int dim, int nstate, typename real>
class GridRefinement_Uniform : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    using GridRefinementBase<dim,nstate,real>::MAX_METHOD_VEC;
    void refine_grid()    override;
protected:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
    void output_results_vtk_method(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec) override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    using GridRefinementBase<dim,nstate,real>::MAX_METHOD_VEC;
    void refine_grid()    override;
protected:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;   
    void output_results_vtk_method(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec) override;

    virtual void error_indicator() = 0;
    void smoothness_indicator();
    void anisotropic_h();
protected:
    dealii::Vector<real> indicator;
    dealii::Vector<real> smoothness;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Error : public GridRefinement_FixedFraction<dim,nstate,real>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Hessian : public GridRefinement_FixedFraction<dim,nstate,real>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Residual : public GridRefinement_FixedFraction<dim,nstate,real>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Adjoint : public GridRefinement_FixedFraction<dim,nstate,real>
{
public:
    using GridRefinement_FixedFraction<dim,nstate,real>::GridRefinement_FixedFraction;
    void error_indicator() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    using GridRefinementBase<dim,nstate,real>::MAX_METHOD_VEC;
    void refine_grid()    override;
protected:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;    
    void output_results_vtk_method(
        dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
        std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec) override;

    void field();
    virtual void field_h() = 0;
    virtual void field_p() = 0;
    virtual void field_hp() = 0;
    real current_complexity();
    void get_current_field_h();
    void get_current_field_p();
    dealii::Vector<real> h_field;
    dealii::Vector<real> p_field;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Error : public GridRefinement_Continuous<dim,nstate,real>
{
public:
    using GridRefinement_Continuous<dim,nstate,real>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Hessian : public GridRefinement_Continuous<dim,nstate,real>
{
public:
    using GridRefinement_Continuous<dim,nstate,real>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Residual : public GridRefinement_Continuous<dim,nstate,real>
{
public:
    using GridRefinement_Continuous<dim,nstate,real>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Adjoint : public GridRefinement_Continuous<dim,nstate,real>
{
public:
    using GridRefinement_Continuous<dim,nstate,real>::GridRefinement_Continuous;
    void field_h()  override;
    void field_p()  override;
    void field_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinementFactory
{
public:
    // different factory calls have access to different Grid refinements
    // adjoint (dg + functional)
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    // dg + physics + Functional
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional);

    // dg + physics
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam                          gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics);

    // dg 
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::GridRefinementParam        gr_param,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg);

};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GRID_REFINEMENT_H__
