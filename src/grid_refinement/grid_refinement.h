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
        PHiLiP::Parameters::AllParameters const *const                   param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinementBase(
        PHiLiP::Parameters::AllParameters const *const                   param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input);

    GridRefinementBase(
        PHiLiP::Parameters::AllParameters const *const                   param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    GridRefinementBase(
        PHiLiP::Parameters::AllParameters const *const param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg_input);

    // refine_grid is the main function
    void refine_grid();

    // refine grid functions to be called
    virtual void refine_grid_h()  = 0;
    virtual void refine_grid_p()  = 0;
    virtual void refine_grid_hp() = 0;

protected:
    // delegated constructor
    GridRefinementBase(
        PHiLiP::Parameters::AllParameters const *const                   param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    // parameters
    PHiLiP::Parameters::AllParameters const *const param;
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
};

template <int dim, int nstate, typename real>
class GridRefinement_Uniform : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;    
    virtual void error_indicator() = 0;
    void smoothness_indicator();
protected:
    dealii::Vector<real> indicator;
    dealii::Vector<real> smoothness;
};

// TODO: check if I need to directly inherit GridRefinementBase as well
// TODO: Could these all also be made private (aka just remove the public)
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
class GridRefinement_Continuous_Error : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Hessian : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Residual : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Adjoint : public GridRefinementBase<dim,nstate,real>
{
public:
    using GridRefinementBase<dim,nstate,real>::GridRefinementBase;
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinementFactory
{
public:
    // different factory calls have access to different Grid refinements
    // adjoint (dg + functional)
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::AllParameters const *const                   param,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input);

    // dg + physics + Functional
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::AllParameters const *const                   param,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional);

    // dg + physics
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::AllParameters const *const                   param,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics);

    // dg 
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::AllParameters const *const param,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg);

};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GRID_REFINEMENT_H__
