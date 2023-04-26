#include "mesh_optimizer.hpp"
#include "functional/dual_weighted_residual_obj_func1.h"
#include "functional/dual_weighted_residual_obj_func2.h"


namespace PHiLiP {

template<int dim, int nstate>
MeshOptimizer::MeshOptimizer(
    std::shared_ptr<DGBase<dim,double>> dg_input,
    const Parameters::AllParameters *const parameters_input, 
    const bool _use_full_space_method)
    : dg(dg_input)
    , all_parameters(parameters_input)
    , use_full_space_method(_use_full_space_method)
{
   design_parameterization = std::make_shared<InnerVolParameterization<dim>>(dg->high_order_grid); 
   bool use_coarse_residual = false;
   if(use_full_space_method) {use_coarse_residual = true};
   objective_function = std::make_shared<DualWeightedResidualObjFunc2<dim, nstate, double>>(dg,true,false,use_coarse_residual);
}


} // PHiLiP namespace
