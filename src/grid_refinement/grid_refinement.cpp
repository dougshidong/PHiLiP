#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters_grid_refinement.h"

#include "dg/dg.h"
#include "dg/high_order_grid.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "physics/physics.h"

#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/size_field.h"
#include "grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

// functions for the refinement calls for each of the classes
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_h()
{
    this->tria.refine_global(1);
}
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_p()
{
    // TODO: add check on polynomial dergee
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            cell->set_future_fe_index(cell->active_fe_index()+1);
}
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_hp()
{
    // TODO: check if an execute statement needs to be added between these
    refine_grid_h();
    refine_grid_p();
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Error<dim,nstate,real>::error_indicator()
{
    //TODO: use manufactured solution to measure the cell-wise error (overintegrate)
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::error_indicator()
{
    // TODO: Feature based, should use the reconstructed next mode as an indication
    // make a function to call that does this reconstruction? will be needed for other classes
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Residual<dim,nstate,real>::error_indicator()
{
    // TODO: project to fine grid and evaluate the Lq norm of the residual
    // may conflict with the solution transfer to a fine grid so could do it on another element?
    // see if nested solution transfers cause issues if the execute is never called
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::error_indicator()
{
    // TODO: use the adjoint to obtain the DWR as the indicator
}

// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Error<dim,nstate,real>::refine_grid_h(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Error<dim,nstate,real>::refine_grid_p(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Error<dim,nstate,real>::refine_grid_hp(){}

// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::refine_grid_h(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::refine_grid_p(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Hessian<dim,nstate,real>::refine_grid_hp(){}

// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Residual<dim,nstate,real>::refine_grid_h(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Residual<dim,nstate,real>::refine_grid_p(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Residual<dim,nstate,real>::refine_grid_hp(){}

// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::refine_grid_h(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::refine_grid_p(){}
// template <int dim, int nstate, typename real>
// void GridRefinement_FixedFraction_Adjoint<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::refine_grid_h()
{
    std::cout << "calling the correct function?" << std::endl;

    int igrid = 0;
    int poly_degree = 1;

    // building error based on exact hessian
    double complexity = 4.0*this->tria.n_active_cells()*4;
    dealii::Vector<double> h_field;
    SizeField<dim,double>::isotropic_uniform(
        this->tria,
        *(this->dg->high_order_grid.mapping_fe_field),
        this->dg->fe_collection[poly_degree],
        this->physics->manufactured_solution_function,
        complexity,
        h_field);

    // now outputting this new field
    std::string write_posname = "grid-"+std::to_string(igrid)+".pos";
    std::ofstream outpos(write_posname);
    GmshOut<dim,double>::write_pos(this->tria,h_field,outpos);

    std::string write_geoname = "grid-"+std::to_string(igrid)+".geo";
    std::ofstream outgeo(write_geoname);
    GmshOut<dim,double>::write_geo(write_posname,outgeo);

    std::string output_name = "grid-"+std::to_string(igrid)+".msh";
    std::cout << "Command is: " << ("gmsh " + write_geoname + " -2 -o " + output_name).c_str() << '\n';
    int a = std::system(("gmsh " + write_geoname + " -2 -o " + output_name).c_str());
    std::cout << "a" << a << std::endl;

    this->tria.clear();
    dealii::GridIn<dim> gridin;
    gridin.attach_triangulation(this->tria);
    std::ifstream f(output_name);
    gridin.read_msh(f);
}
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
void GridRefinementBase<dim,nstate,real>::refine_grid()
{
    // using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using RefinementTypeEnum   = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    // RefinementMethodEnum refinement_method = this->grid_refinement_param.refinement_method;
    RefinementTypeEnum   refinement_type   = this->grid_refinement_param.refinement_type;

    // // TODO: add solution transfer flag here
    // // add to constructor
    // // dealii::parallel::distributed::SolutionTransfer< 
    // //     dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim> 
    // //     > solution_transfer(dg->dof_handler);
    // if(true){
    //     // TODO: check if this can be the same vector or most likely needs to be copied first
    //     // solution_transfer.prepare_for_coarsening_and_refinement(old_solution);
    // }

    // // TODO: prepare, only needed in cases where using the default DEALii refinements
    // if(refinement_method == RefinementMethodEnum::uniform || 
    //    refinement_method == RefinementMethodEnum::fixed_fraction){
    //     this->dg->high_order_grid.prepare_for_coarsening_and_refinement();
    //     this->tria.prepare_coarsening_and_refinement();
    // }

    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    // // TODO: exectute
    // if(refinement_method == RefinementMethodEnum::uniform || 
    //    refinement_method == RefinementMethodEnum::fixed_fraction){
    //     this->tria.execute_coarsening_and_refinement(); // check if this one is necessary
    //     this->dg->high_order_grid.execute_coarsening_and_refinement();
    // }
    // // TODO: complete the refinement
    // if(true){
    //     this->dg->allocate_system();
    //     this->dg->solution.zero_out_ghosts();
    //     // solution_transfer.interpolate(dg->solution);
    //     this->dg->solution.update_ghost_values();
    // }

    // // TODO: if reinit
    // if(true){

    // }
}

// constructors for GridRefinementBase
template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const                   param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) : 
        GridRefinementBase<dim,nstate,real>(
            param_input, 
            adj_input, 
            adj_input->functional, 
            adj_input->dg, 
            physics_input){}

template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const                   param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input) :
        GridRefinementBase<dim,nstate,real>(
            param_input, 
            nullptr, 
            functional_input, 
            dg_input, 
            physics_input){}

template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const                   param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) : 
        GridRefinementBase<dim,nstate,real>(
            param_input, 
            nullptr, 
            nullptr, 
            dg_input, 
            physics_input){}

template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >   dg_input) :
        GridRefinementBase<dim,nstate,real>(
            param_input, 
            nullptr, 
            nullptr, 
            dg_input, 
            nullptr){}

// main constructor is private for constructor delegation
template <int dim, int nstate, typename real>
GridRefinementBase<dim,nstate,real>::GridRefinementBase(
    PHiLiP::Parameters::AllParameters const *const                   param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real> >         functional_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) :
        param(param_input),
        grid_refinement_param(param_input->grid_refinement_param),
        adj(adj_input),
        functional(functional_input),
        dg(dg_input),
        physics(physics_input),
        tria(*(dg_input->triangulation)){}

// factory for different options, ensures that the provided 
// values match with the selected refinement type

// adjoint (dg + functional)
template <int dim, int nstate, typename real>
std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
    PHiLiP::Parameters::AllParameters const *const                   param,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real> >            adj,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics)
{
    // all adjoint based methods should be constructed here
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = param->grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_param.error_indicator;

    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_FixedFraction_Adjoint<dim,nstate,real> >(param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_Continuous_Adjoint<dim,nstate,real> >(param, adj, physics);
    }

    return create_GridRefinement(param, adj->dg, physics, adj->functional);
}

// dg + physics + Functional
template <int dim, int nstate, typename real>
std::shared_ptr< GridRefinementBase<dim,nstate,real> > 
GridRefinementFactory<dim,nstate,real>::create_GridRefinement(
    PHiLiP::Parameters::AllParameters const *const                   param,
    std::shared_ptr< PHiLiP::DGBase<dim, real> >                     dg,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics,
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
template class GridRefinementBase<PHILIP_DIM, 1, double>;
template class GridRefinementBase<PHILIP_DIM, 2, double>;
template class GridRefinementBase<PHILIP_DIM, 3, double>;
template class GridRefinementBase<PHILIP_DIM, 4, double>;
template class GridRefinementBase<PHILIP_DIM, 5, double>;

template class GridRefinement_Uniform<PHILIP_DIM, 1, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double>;

template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double>;

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

template class GridRefinementFactory<PHILIP_DIM, 1, double>;
template class GridRefinementFactory<PHILIP_DIM, 2, double>;
template class GridRefinementFactory<PHILIP_DIM, 3, double>;
template class GridRefinementFactory<PHILIP_DIM, 4, double>;
template class GridRefinementFactory<PHILIP_DIM, 5, double>;

} // namespace GridRefinement

} // namespace PHiLiP
