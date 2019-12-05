#ifndef __GRID_REFINEMENT_H__
#define __GRID_REFINEMENT_H__

namespace PHiLiP {

namespace GridRefinement {

class GridRefinementParam
{
public:
    // main set of parameters for deciding the method

    enum RefinementMethod{
        uniform,        // all cells are refined
        fixed_fraction, // picks fraction with largest indicators
        continuous,     // generates a new mesh based on a size field
        };
    RefinementMethod refinement_method;

    enum RefinementType{
        h,  // element size only
        p,  // polynomial orders
        hp, // mix of both
        };
    RefinementType refinement_type;
    
    bool isotropic;

    enum ErrorIndicator{
        error_based,    // using the exact error for testing
        hessian_based,  // feature_based
        residual_based, // requires a fine grid projection
        adjoint_based,  // adjoint based
        };
    ErrorIndicator error_indicator;    

    // need to add: isotropy indicators AND smoothness indicator

    // double p; // polynomial order when fixed, should take this from the grid
    double q; // for the Lq norm

    double fixed_fraction;

    GridRefinementParam(); ///< Constructor

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters(dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters(dealii::ParameterHandler &prm);
};

template <int dim, int nstate, typename real>
class GridRefinement_Uniform : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Error : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Hessian : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Residual : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_FixedFraction_Adjoint : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Error : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Hessian : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Residual : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

template <int dim, int nstate, typename real>
class GridRefinement_Continuous_Adjoint : GridRefinementBase<dim,nstate,real>
{
public:
    void refine_grid_h()  override;
    void refine_grid_p()  override;
    void refine_grid_hp() override;
};

// central class of the grid_refinement, controls refinements
template <int dim, int nstate, typename real>
class GridRefinementBase
{
public:
    // deleting the default constructor
    GridRefinementBase() = delete;

    // constructor stores the parameters
    GridRefinementBase(
        PHiLiP::Parameters::AllParameters const *const        param_input,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> > adj_input);

    GridRefinementBase(
        PHiLiP::Parameters::AllParameters const *const                   param_input,
        std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
        std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input,
        std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional);

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

    // parameters
    PHiLiP::Parameters::AllParameters const *const param;
    PHiLiP::Parameters::GridRefinementParam grid_refinement_param;

    // different things needed depending on the choice of refinement
    // these could be held here with nullptr or in the base class
    // if I want these internal then this needs to be templated on
    // template <int dim, int nstate, typename real>

    // adj
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> > adj;

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

    // // triangulation
    // dealii::Triangulation<dim, dim> &tria;
};

template <int dim, int nstate, typename real>
class GridRefinementFactory
{
    // different factory calls have access to different Grid refinements
    // adjoint (dg + functional)
    static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    create_GridRefinement(
        PHiLiP::Parameters::AllParameters const *const        param,
        std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> > adj);

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

    // cannot use either of these as they'd need a high_order_grid, object in dg        
    // // physics + triangulation
    // static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    // create_GridRefinement(
    //     PHiLiP::Parameters::AllParameters const *const          param,
    //     std::shared_ptr< PHiLiP::PhysicsBase<dim,nstate,real> > physics,
    //     dealii::Triangulation<dim, dim> &                       tria);

    // // triangulation
    // static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
    // create_GridRefinement(
    //     PHiLiP::Parameters::AllParameters const *const param,
    //     dealii::Triangulation<dim, dim> &              tria);
};

} // namespace GridRefinement

} //namespace PHiLiP

#endif // __GRID_REFINEMENT_H__
