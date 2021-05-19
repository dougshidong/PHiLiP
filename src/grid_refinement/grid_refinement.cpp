#include <vector>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>

#include "parameters/all_parameters.h"
#include "parameters/parameters_grid_refinement.h"

#include "dg/dg.h"
#include "mesh/high_order_grid.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "physics/physics.h"

#include "post_processor/physics_post_processor.h"

#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/msh_out.h"
#include "grid_refinement/size_field.h"
#include "grid_refinement/reconstruct_poly.h"
#include "grid_refinement/field.h"

#include "grid_refinement/grid_refinement_uniform.h"
#include "grid_refinement/grid_refinement_fixed_fraction.h"
#include "grid_refinement/grid_refinement_continuous.h"

#include "grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

// output results functions
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk(const unsigned int iref)
{
    // creating the data out stream
    dealii::DataOut<dim, dealii::DoFHandler<dim>> data_out;
    data_out.attach_dof_handler(dg->dof_handler);

    // dg should always be valid
    std::shared_ptr< dealii::DataPostprocessor<dim> > post_processor;
    dealii::Vector<float>                             subdomain;
    std::vector<unsigned int>                         active_fe_indices;
    dealii::Vector<double>                            cell_poly_degree;
    std::vector<std::string>                          residual_names;
    if(dg)
        output_results_vtk_dg(data_out, post_processor, subdomain, active_fe_indices, cell_poly_degree, residual_names);

    // checking nullptr for each subsection
    // functional 
    if(functional)
        output_results_vtk_functional(data_out);

    // physics
    if(physics)
        output_results_vtk_physics(data_out);

    // adjoint
    /*
    std::vector<std::string> dIdw_names_coarse;
    std::vector<std::string> adjoint_names_coarse;
    std::vector<std::string> dIdw_names_fine;
    std::vector<std::string> adjoint_names_fine;
    if(adjoint)
        output_results_vtk_adjoint(data_out, dIdw_names_coarse, adjoint_names_coarse, dIdw_names_fine, adjoint_names_fine);
    */

    // plotting the error compared to the manufactured solution
    dealii::Vector<real> l2_error_vec;
    if(physics && physics->manufactured_solution_function)
        output_results_vtk_error(data_out, l2_error_vec);

    // virtual method to call each refinement type 
    // gets a vector of pairs and strings to be returned (needed to be kept in scope for output)
    std::vector< std::pair<dealii::Vector<real>, std::string> > data_out_vector = output_results_vtk_method();

    // looping through the vector list to assign the items
    for(unsigned int index = 0; index < data_out_vector.size(); index++){
        data_out.add_data_vector(
            data_out_vector[index].first,
            data_out_vector[index].second,
            dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    }

    // performing the ouput on each core
    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    // default
    data_out.build_patches();

    // curved
    // typename dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion curved 
    //     = dealii::DataOut<dim,dealii::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells;
    // const dealii::Mapping<dim> &mapping = (*(dg->high_order_grid.mapping_fe_field));
    // const int n_subdivisions = dg->max_degree;
    // data_out.build_patches(mapping, n_subdivisions, curved);
    // const bool write_higher_order_cells = (dim>1) ? true : false; 
    // dealii::DataOutBase::VtkFlags vtkflags(0.0,igrid,true,dealii::DataOutBase::VtkFlags::ZlibCompressionLevel::best_compression,write_higher_order_cells);
    // data_out.set_flags(vtkflags);

    std::string filename = "gridRefinement-"
                        //  + dealii::Utilities::int_to_string(dim, 1) + "D-"
                         + dealii::Utilities::int_to_string(iref, 4) + "."
                         + dealii::Utilities::int_to_string(iteration, 4) + "."
                         + dealii::Utilities::int_to_string(iproc, 4) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);

    // master file
    if(iproc == 0){
        std::vector<std::string> filenames;
        for(unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(mpi_communicator); ++iproc){
            std::string fn = "gridRefinement-"
                             //  + dealii::Utilities::int_to_string(dim, 1) + "D-"
                             + dealii::Utilities::int_to_string(iref, 4) + "."
                             + dealii::Utilities::int_to_string(iteration, 4) + "."
                             + dealii::Utilities::int_to_string(iproc, 4) + ".vtu";
            filenames.push_back(fn);
        }
    
        std::string master_filename = "gridRefinement-"
                                      //  + dealii::Utilities::int_to_string(dim, 1) + "D-"
                                      + dealii::Utilities::int_to_string(iref, 4) + "."
                                      + dealii::Utilities::int_to_string(iteration, 4) + ".pvtu";
        std::ofstream master_output(master_filename);
        data_out.write_pvtu_record(master_output, filenames);
    }

}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_dg(
    dealii::DataOut<dim, dealii::DoFHandler<dim>> &    data_out,
    std::shared_ptr< dealii::DataPostprocessor<dim> > &post_processor,
    dealii::Vector<float> &                            subdomain,
    std::vector<unsigned int> &                        active_fe_indices,
    dealii::Vector<double> &                           cell_poly_degree,
    std::vector<std::string> &                         residual_names)
{
    post_processor = std::make_shared< PHiLiP::Postprocess::PhysicsPostprocessor<dim,nstate> >(dg->all_parameters);
    data_out.add_data_vector(dg->solution, *post_processor);

    subdomain.reinit(dg->triangulation->n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i) {
        subdomain(i) = dg->triangulation->locally_owned_subdomain();
    }
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output the polynomial degree in each cell
    dg->dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector(cell_poly_degree, "PolynomialDegree", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    for(int s=0;s<nstate;++s) {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }

    data_out.add_data_vector(dg->right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_functional(
    dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out)
{
    // nothing here for now, could plot the contributions or weighting function
    (void) data_out;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_physics(
    dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out)
{
    // TODO: plot the function value, gradient, tensor, etc.
    (void) data_out;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_adjoint(
    dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out,
    std::vector<std::string> &                     dIdw_names_coarse,
    std::vector<std::string> &                     adjoint_names_coarse,
    std::vector<std::string> &                     dIdw_names_fine,
    std::vector<std::string> &                     adjoint_names_fine)
{
    // starting with coarse grid results
    adjoint->reinit();
    adjoint->coarse_grid_adjoint();

    for(int s=0;s<nstate;++s) {
        std::string varname = "dIdw" + dealii::Utilities::int_to_string(s,1) + "_coarse";
        dIdw_names_coarse.push_back(varname);
    }
    data_out.add_data_vector(adjoint->dIdw_coarse, dIdw_names_coarse, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    for(int s=0;s<nstate;++s) {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1) + "_coarse";
        adjoint_names_coarse.push_back(varname);
    }
    data_out.add_data_vector(adjoint->adjoint_coarse, adjoint_names_coarse, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    // next for fine grid results
    adjoint->fine_grid_adjoint();
    adjoint->dual_weighted_residual();
    
    for(int s=0;s<nstate;++s) {
        std::string varname = "dIdw" + dealii::Utilities::int_to_string(s,1) + "_fine";
        dIdw_names_fine.push_back(varname);
    }
    // data_out.add_data_vector(adjoint->dIdw_fine, dIdw_names_fine, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    for(int s=0;s<nstate;++s) {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1) + "_fine";
        adjoint_names_fine.push_back(varname);
    }
    // data_out.add_data_vector(adjoint->adjoint_fine, adjoint_names_fine, dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
    
    data_out.add_data_vector(adjoint->dual_weighted_residual_fine, "DWR", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // returning to original state
    adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double,MeshType>::AdjointStateEnum::coarse);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_error(
    dealii::DataOut<dim, dealii::DoFHandler<dim>> &data_out,
    dealii::Vector<real> &                             l2_error_vec)
{
    int overintegrate = 10;
    int poly_degree = dg->get_max_fe_degree();
    dealii::QGauss<dim> quad_extra(dg->max_degree+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid->mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
        dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);
    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<real,nstate> soln_at_q;
    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);

    // L2 error (squared) contribution per cell
    l2_error_vec.reinit(tria->n_active_cells());
    for(auto cell = dg->dof_handler.begin_active(); cell < dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        fe_values_extra.reinit(cell);
        cell->get_dof_indices(dofs_indices);

        real cell_l2_error   = 0.0;

        for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
            for(unsigned int idof = 0; idof < fe_values_extra.dofs_per_cell; ++idof){
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg->solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            for(unsigned int istate = 0; istate < nstate; ++istate){
                const double uexact = physics->manufactured_solution_function->value(qpoint, istate);
                cell_l2_error += pow(soln_at_q[istate] - uexact, 2) * fe_values_extra.JxW(iquad);
            }
        }

        l2_error_vec[cell->active_cell_index()] += cell_l2_error;
    }

    data_out.add_data_vector(l2_error_vec, "l2_error", dealii::DataOut_DoFData<dealii::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
}

// constructors for GridRefinementBase
template <int dim, int nstate, typename real, typename MeshType>
GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase(
    PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) : 
        GridRefinementBase<dim,nstate,real,MeshType>(
            gr_param_input, 
            adj_input, 
            adj_input->functional, 
            adj_input->dg, 
            physics_input){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase(
    PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input) :
        GridRefinementBase<dim,nstate,real,MeshType>(
            gr_param_input, 
            nullptr, 
            functional_input, 
            dg_input, 
            physics_input){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase(
    PHiLiP::Parameters::GridRefinementParam                          gr_param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics_input) : 
        GridRefinementBase<dim,nstate,real,MeshType>(
            gr_param_input, 
            nullptr, 
            nullptr, 
            dg_input, 
            physics_input){}

template <int dim, int nstate, typename real, typename MeshType>
GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase(
    PHiLiP::Parameters::GridRefinementParam                gr_param_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg_input) :
        GridRefinementBase<dim,nstate,real,MeshType>(
            gr_param_input, 
            nullptr, 
            nullptr, 
            dg_input, 
            nullptr){}

// main constructor is private for constructor delegation
template <int dim, int nstate, typename real, typename MeshType>
GridRefinementBase<dim,nstate,real,MeshType>::GridRefinementBase(
    PHiLiP::Parameters::GridRefinementParam                            gr_param_input,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >    adj_input,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional_input,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg_input,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics_input) :
        grid_refinement_param(gr_param_input),
        error_indicator_type(gr_param_input.error_indicator),
        adjoint(adj_input),
        functional(functional_input),
        dg(dg_input),
        physics(physics_input),
        tria(dg_input->triangulation),
        iteration(0),
        mpi_communicator(MPI_COMM_WORLD),
        pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_communicator)==0){}

// factory for different options, ensures that the provided 
// values match with the selected refinement type

// adjoint (dg + functional)
template <int dim, int nstate, typename real, typename MeshType>
std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
GridRefinementFactory<dim,nstate,real,MeshType>::create_GridRefinement(
    PHiLiP::Parameters::GridRefinementParam                          gr_param,
    std::shared_ptr< PHiLiP::Adjoint<dim, nstate, real, MeshType> >  adj,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics)
{
    // all adjoint based methods should be constructed here
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = gr_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = gr_param.error_indicator;

    // adjoint (dg + functional)
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }

    // dg + physics
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }

    // dg
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::uniform){
        return std::make_shared< GridRefinement_Uniform<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }

    return create_GridRefinement(gr_param, adj->dg, physics, adj->functional);
}

// dg + physics + Functional
template <int dim, int nstate, typename real, typename MeshType>
std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
GridRefinementFactory<dim,nstate,real,MeshType>::create_GridRefinement(
    PHiLiP::Parameters::GridRefinementParam                            gr_param,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >             dg,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> >   physics,
    std::shared_ptr< PHiLiP::Functional<dim, nstate, real, MeshType> > functional)
{
    // hessian and error based
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = gr_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = gr_param.error_indicator;

    // dg + physics
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }

    // dg
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::uniform){
        return std::make_shared< GridRefinement_Uniform<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }

    return create_GridRefinement(gr_param, dg, physics);
}

// dg + physics
template <int dim, int nstate, typename real, typename MeshType>
std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
GridRefinementFactory<dim,nstate,real,MeshType>::create_GridRefinement(
    PHiLiP::Parameters::GridRefinementParam                          gr_param,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> >           dg,
    std::shared_ptr< PHiLiP::Physics::PhysicsBase<dim,nstate,real> > physics)
{
    // hessian and error based
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = gr_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = gr_param.error_indicator;

    // dg + physics
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }

    // dg
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::uniform){
        return std::make_shared< GridRefinement_Uniform<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }

    return create_GridRefinement(gr_param, dg);
}

// dg
template <int dim, int nstate, typename real, typename MeshType>
std::shared_ptr< GridRefinementBase<dim,nstate,real,MeshType> > 
GridRefinementFactory<dim,nstate,real,MeshType>::create_GridRefinement(
    PHiLiP::Parameters::GridRefinementParam                gr_param,
    std::shared_ptr< PHiLiP::DGBase<dim, real, MeshType> > dg)
{
    // residual based or uniform
    using RefinementMethodEnum = PHiLiP::Parameters::GridRefinementParam::RefinementMethod;
    using ErrorIndicatorEnum   = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    RefinementMethodEnum refinement_method = gr_param.refinement_method;
    ErrorIndicatorEnum   error_indicator   = gr_param.error_indicator;

    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction<dim,nstate,real,MeshType> >(gr_param, dg);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous<dim,nstate,real,MeshType> >(gr_param, dg);
    }else if(refinement_method == RefinementMethodEnum::uniform){
        return std::make_shared< GridRefinement_Uniform<dim,nstate,real,MeshType> >(gr_param, dg);
    }

    std::cout << "Invalid GridRefinement." << std::endl;

    return nullptr;
}

// large amount of templating to be done, move to an .inst file
// could also try reducing this with BOOST

// dealii::Triangulation<PHILIP_DIM>
template class GridRefinementBase<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinementFactory<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

// dealii::parallel::shared::Triangulation<PHILIP_DIM>
template class GridRefinementBase<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinementFactory<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM != 1
// dealii::parallel::distributed::Triangulation<PHILIP_DIM>
template class GridRefinementBase<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinementFactory<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace GridRefinement

} // namespace PHiLiP
