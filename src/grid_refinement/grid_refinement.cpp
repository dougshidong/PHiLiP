
#include "grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

// functions for the refinement calls for each of the classes
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void ridRefinement_FixedFraction_Error<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void ridRefinement_FixedFraction_Error<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void ridRefinement_FixedFraction_Error<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Residual<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Residual<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Residual<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Residual<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Residual<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Residual<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Adjoint<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Adjoint<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Adjoint<dim,nstate,real>::refine_grid_hp(){}

// central refine grid call
template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = grid_refinement_param.RefinementType;

    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }
}

// constructors for GridRefinementBase
template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const        param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> > adj_input) : 
        GridRefinementBase<dim,nstate,real>(param_input, adj_input->dg, adj_input->physics, adj_input->functional),
        adj(adj_input){}

template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const                   param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input) :
        GridRefinementBase<dim,nstate,real>(param_input, dg_input, physics_input),
        functional(functional_input){}

template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const                   param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) : 
        GridRefinementBase<dim,nstate,real>(param_input, dg_input),
        physics(physics_input){}

template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg_input) :
        param(param_input),
        grid_refinement_param(param_input->grid_refinement_param),
        dg(dg_input){}

// factory for different options, ensures that the provided 
// values match with the selected refinement type

// adjoint (dg + functional)
template <int dim, int nstate, typename real>
std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
    PHiLiP::Parameters::AllParameters const *const        param,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> > adj)
{
    // all adjoint based methods should be constructed here
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = param->grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_param.error_indicator;

    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_FixedFraction_Adjoint<dim,nstate,real> >(param, adj);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_Continuous_Adjoint<dim,nstate,real> >(param, adj);
    }

    return create_GridRefinement(param, adj->dg, adj->physics, adj->functional);
}

// dg + physics + Functional
template <int dim, int nstate, typename real>
std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
    PHiLiP::Parameters::AllParameters const *const                   param,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         /*functional*/)
{
    // currently nothing that uses only the functional directly
    return create_GridRefinement(param, dg, physics);
}

// dg + physics
template <int dim, int nstate, typename real>
std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
    PHiLiP::Parameters::AllParameters const *const                   param,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics)
{
    // hessian and error based
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = param->grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_param.error_indicator;

    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_FixedFraction_Hessian<dim,nstate,real> >(param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction_Error<dim,nstate,real> >(param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous_Hessian<dim,nstate,real> >(param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous_Error<dim,nstate,real> >(param, dg, physics);
    }

    return create_GridRefinement(param, dg);
}

// dg
template <int dim, int nstate, typename real>
std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
    PHiLiP::Parameters::AllParameters const *const param,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg)
{
    // residual based or uniform
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = param->grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_param.error_indicator;

    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction_Residual<dim,nstate,real> >(param, dg);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous_Residual<dim,nstate,real> >(param, dg);
    }else if(refinement_method == RefinementMethodEnum::uniform){
        return std::make_shared< GridRefinement_Uniform<dim, nstate, real> >(param, dg);
    }

    std::cout << "Invalid GridRefinement." << std::endl;

    return nullptr;
}

// cannot use either of these as they'd need a high_order_grid, object in dg    
// // physics + triangulation
// template <int dim, int nstate, typename real>
// static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
// GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
//     PHiLiP::Parameters::AllParameters const *const          param,
//     std::shared_ptr< PHiLiP::PhysicsBase<dim,nstate,real> > physics,
//     dealii::Triangulation<dim, dim> &                       tria)
// {
//     return create_GridRefinement(param, tria);
// }

// // triangulation
// template <int dim, int nstate, typename real>
// static std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
// GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
//     PHiLiP::Parameters::AllParameters const *const param,
//     dealii::Triangulation<dim, dim> &              tria)
// {
//     std::cout << "Invalid Grid refinement." << std::endl;

//     return nullptr;
// }

// large amount of templating to be done, move to an .inst file and see if it can be reduced
template class GridRefinement_Uniform<PHILIP_DIM, 1, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double>;

template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 1, double>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 2, double>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 3, double>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 4, double>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 5, double>;

template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 1, double>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 2, double>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 3, double>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 4, double>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 5, double>;

template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 1, double>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 2, double>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 3, double>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 4, double>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 5, double>;

template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 1, double>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 2, double>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 3, double>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 4, double>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 5, double>;

template class GridRefinement_Continuous_Error<PHILIP_DIM, 1, double>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 2, double>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 3, double>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 4, double>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 5, double>;

template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 1, double>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 2, double>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 3, double>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 4, double>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 5, double>;

template class GridRefinement_Continuous_Residual<PHILIP_DIM, 1, double>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 2, double>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 3, double>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 4, double>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 5, double>;

template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 1, double>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 2, double>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 3, double>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 4, double>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 5, double>;

template class GridRefinementBase<PHILIP_DIM, 1, double>;
template class GridRefinementBase<PHILIP_DIM, 2, double>;
template class GridRefinementBase<PHILIP_DIM, 3, double>;
template class GridRefinementBase<PHILIP_DIM, 4, double>;
template class GridRefinementBase<PHILIP_DIM, 5, double>;

template class GridRefinementFactory<PHILIP_DIM, 1, double>;
template class GridRefinementFactory<PHILIP_DIM, 2, double>;
template class GridRefinementFactory<PHILIP_DIM, 3, double>;
template class GridRefinementFactory<PHILIP_DIM, 4, double>;
template class GridRefinementFactory<PHILIP_DIM, 5, double>;

} // namespace GridRefinement

} // namespace PHiLiP
