#include <deal.II/grid/tria.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_values.h>

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
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    template <int dim> using Triangulation = dealii::Triangulation<dim>;
#else
    template <int dim> using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

// functions for the refinement calls for each of the classes
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_h()
{
    this->tria->set_all_refine_flags();
}
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_p()
{
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned() && cell->active_fe_index()+1 <= this->dg->max_degree)
            cell->set_future_fe_index(cell->active_fe_index()+1);
    
}
template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid_hp()
{
    refine_grid_h();
    refine_grid_p();
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_h()
{
    // Compute the error indicator to define this->indicator
    error_indicator();

    // Performing the call for refinement
#if PHILIP_DIM==1
    dealii::GridRefinement::refine_and_coarsen_fixed_number(
        *(this->tria),
        this->indicator,
        this->grid_refinement_param.refinement_fraction,
        this->grid_refinement_param.coarsening_fraction);
#else
    dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        *(this->tria),
        this->indicator,
        this->grid_refinement_param.refinement_fraction,
        this->grid_refinement_param.coarsening_fraction);
#endif
}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_p()
{
    // TODO: call refine_grid_h, then loop over and replace any h refinement
    //       flags with a polynomial enrichment
    refine_grid_h();
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            if(cell->refine_flag_set()){
                cell->clear_refine_flag();
                cell->set_active_fe_index(cell->active_fe_index()+1);
            }

}
template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_hp()
{
    // TODO: Same idea as above, except the switch in refine_grid_p
    //       now has to meet some tolerance, e.g. smoothness, jump
    // will need to implement the choice between different methods here
    refine_grid_h();
    smoothness_indicator();
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::smoothness_indicator()
{
    // reads the options and determines the proper smoothness indicator
    
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Error<dim,nstate,real>::error_indicator()
{
    // TODO: update this to work with p-adaptive schemes
    // see dg.cpp
    // const auto mapping = (*(high_order_grid.mapping_fe_field));
    // dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    // dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, fe_collection, this->dg->volume_quadrature_collection, this->dg->volume_update_flags);

    // use manufactured solution to measure the cell-wise error (overintegrate)
    int overintegrate = 10;
    int poly_degree = 1; // need a way of determining the polynomial order more easily
    dealii::QGauss<dim> quadrature(this->dg->max_degree+overintegrate);
    dealii::FEValues<dim,dim> fe_values(*(this->dg->high_order_grid.mapping_fe_field), this->dg->fe_collection[poly_degree], quadrature, 
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    // norm to use 
    double norm_Lq = this->grid_refinement_param.norm_Lq;

    // storing the result in 
    std::vector<dealii::types::global_dof_index> dofs_indices(fe_values.dofs_per_cell);
    this->indicator.reinit(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        fe_values.reinit(cell);
        cell->get_dof_indices(dofs_indices);

        double cell_error = 0;
        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for(unsigned int idof = 0; idof < fe_values.dofs_per_cell; ++idof){
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += this->dg->solution[dofs_indices[idof]] * fe_values.shape_value_component(idof,iquad,istate);
            }
            
            const dealii::Point<dim> qpoint = (fe_values.quadrature_point(iquad));

            for(int istate = 0; istate < nstate; ++istate){
                const double uexact = this->physics->manufactured_solution_function->value(qpoint,istate);
                cell_error += pow(abs(soln_at_q[istate] - uexact), norm_Lq) * fe_values.JxW(iquad);
            }
        }
        this->indicator[cell->active_cell_index()] = cell_error;
    }

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
    // reinitializing the adjoint with current values
    this->adjoint->reinit();

    // evaluating the functional derivatives and adjoint
    this->adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,real>::AdjointStateEnum::fine);
    this->adjoint->fine_grid_adjoint();
    
    // reinitializing the error indicator vector
    this->indicator.reinit(this->adjoint->dg->triangulation->n_active_cells());
    this->indicator = this->adjoint->dual_weighted_residual();

    // return to the coarse grid
    this->adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,real>::AdjointStateEnum::coarse);
}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::refine_grid_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::refine_grid_h()
{
    int igrid = 0;
    int poly_degree = 1;

    // building error based on exact hessian
    double complexity = pow(poly_degree+1, dim)*this->tria->n_active_cells()*4;
    dealii::Vector<double> h_field;
    SizeField<dim,double>::isotropic_uniform(
        *(this->tria),
        *(this->dg->high_order_grid.mapping_fe_field),
        this->dg->fe_collection[poly_degree],
        this->physics->manufactured_solution_function,
        complexity,
        h_field);

    // now outputting this new field
    std::string write_posname = "grid-"+std::to_string(igrid)+".pos";
    std::ofstream outpos(write_posname);
    GmshOut<dim,double>::write_pos(*(this->tria),h_field,outpos);

    std::string write_geoname = "grid-"+std::to_string(igrid)+".geo";
    std::ofstream outgeo(write_geoname);
    GmshOut<dim,double>::write_geo(write_posname,outgeo);

    std::string output_name = "grid-"+std::to_string(igrid)+".msh";
    std::cout << "Command is: " << ("gmsh " + write_geoname + " -2 -o " + output_name).c_str() << '\n';
    int a = std::system(("gmsh " + write_geoname + " -2 -o " + output_name).c_str());
    std::cout << "a" << a << std::endl;

    this->tria->clear();
    dealii::GridIn<dim> gridin;
    gridin.attach_triangulation(*(this->tria));
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
    //     this->tria->prepare_coarsening_and_refinement();
    // }

    // if(refinement_method == RefinementMethodEnum::uniform 
    // || refinement_method == RefinementMethodEnum::fixed_fraction){
    //     this->dg->high_order_grid.prepare_for_coarsening_and_refinement();
    //     dg->triangulation->prepare_coarsening_and_refinement();
    // }

    this->dg->high_order_grid.prepare_for_coarsening_and_refinement();
    this->dg->triangulation->prepare_coarsening_and_refinement();

    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    this->tria->execute_coarsening_and_refinement();
    this->dg->high_order_grid.execute_coarsening_and_refinement();

    // if(refinement_method == RefinementMethodEnum::uniform){
    //     this->tria->execute_coarsening_and_refinement();
    //     this->dg->high_order_grid.execute_coarsening_and_refinement();
    // }

    // if(refinement_method == RefinementMethodEnum::fixed_fraction){
    //     this->tria->execute_coarsening_and_refinement();
    //     this->dg->high_order_grid.execute_coarsening_and_refinement();
    // }

    // reallocating
    // this->dg->allocate_system();

    // transfer the solution if desired
    // this->dg->solution.zero_out_ghosts();
    // solution_transfer.interpolate(dg->solution);
    // this->dg->solution.update_ghost_values();

    // // TODO: exectute
    // if(refinement_method == RefinementMethodEnum::uniform || 
    //    refinement_method == RefinementMethodEnum::fixed_fraction){
    //     this->tria->execute_coarsening_and_refinement(); // check if this one is necessary
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
        grid_refinement_param(param_input->grid_refinement_study_param.grid_refinement_param),
        adjoint(adj_input),
        functional(functional_input),
        dg(dg_input),
        physics(physics_input),
        tria(dg_input->triangulation){}

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
    RefinementMethodEnum refinement_method = param->grid_refinement_study_param.grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_study_param.grid_refinement_param.error_indicator;

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
    RefinementMethodEnum refinement_method = param->grid_refinement_study_param.grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_study_param.grid_refinement_param.error_indicator;

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
    RefinementMethodEnum refinement_method = param->grid_refinement_study_param.grid_refinement_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = param->grid_refinement_study_param.grid_refinement_param.error_indicator;

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
