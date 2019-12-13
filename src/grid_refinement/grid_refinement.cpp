#include <vector>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_tools.h>

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
#include "grid_refinement/reconstruct_poly.h"
#include "grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {
    
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    template <int dim> using Triangulation = dealii::Triangulation<dim>;
#else
    template <int dim> using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

template <int dim, int nstate, typename real>
void GridRefinement_Uniform<dim,nstate,real>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // setting up the solution transfer
    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs = this->dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(this->dg->dof_handler, locally_relevant_dofs);

    dealii::LinearAlgebra::distributed::Vector<real> solution_old(this->dg->solution);
    solution_old.update_ghost_values();

    dealii::parallel::distributed::SolutionTransfer< 
        dim, dealii::LinearAlgebra::distributed::Vector<real>, dealii::hp::DoFHandler<dim> 
        > solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_old);

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

    // transfering the solution from solution_old
    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();
    solution_transfer.interpolate(this->dg->solution);
    this->dg->solution.update_ghost_values();
}

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
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // compute the error indicator, stored in this->indicator
    error_indicator();

    // computing the smoothness_indicator only for the hp case
    if(refinement_type == RefinementTypeEnum::hp){
        smoothness_indicator();
    }

    // setting up the solution transfer
    dealii::IndexSet locally_owned_dofs, locally_relevant_dofs;
    locally_owned_dofs = this->dg->dof_handler.locally_owned_dofs();
    dealii::DoFTools::extract_locally_relevant_dofs(this->dg->dof_handler, locally_relevant_dofs);

    dealii::LinearAlgebra::distributed::Vector<double> solution_old(this->dg->solution);
    solution_old.update_ghost_values();

    dealii::parallel::distributed::SolutionTransfer< 
        dim, dealii::LinearAlgebra::distributed::Vector<double>, dealii::hp::DoFHandler<dim> 
        > solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_old);

    this->dg->high_order_grid.prepare_for_coarsening_and_refinement();
    this->dg->triangulation->prepare_coarsening_and_refinement();

    // performing the refinement
    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    // check for anisotropic h-adaptation
    if(!this->grid_refinement_param.isotropic){
        anisotropic_h();
    }

    this->tria->execute_coarsening_and_refinement();
    this->dg->high_order_grid.execute_coarsening_and_refinement();

    // transfering the solution from solution_old
    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();
    solution_transfer.interpolate(this->dg->solution);
    this->dg->solution.update_ghost_values();
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_h()
{
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
    // flags cells using refine_grid_h, then loop over and replace any h refinement flags with a polynomial enrichment
    refine_grid_h();
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            if(cell->refine_flag_set()){
                cell->clear_refine_flag();
                if(cell->active_fe_index()+1 <= this->dg->max_degree)
                    cell->set_active_fe_index(cell->active_fe_index()+1);
            }
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::refine_grid_hp()
{
    // TODO: Same idea as above, except the switch in refine_grid_p
    //       now has to meet some tolerance, e.g. smoothness, jump
    // will need to implement the choice between different methods here
    // will start with an h_refinement call and then looping over flags
    refine_grid_h();
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned() && cell->active_fe_index()+1 <= this->dg->max_degree)
            if(cell->refine_flag_set()){
                // perform the h/p decision making
                cell->clear_refine_flag();
                cell->set_active_fe_index(cell->active_fe_index()+1);
            }
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::smoothness_indicator()
{
    // reads the options and determines the proper smoothness indicator
    
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction<dim,nstate,real>::anisotropic_h()
{
    // based on dealii step-30
    const dealii::UpdateFlags face_update_flags = 
        dealii::update_values | 
        dealii::update_gradients | 
        dealii::update_quadrature_points | 
        dealii::update_JxW_values | 
        dealii::update_normal_vectors | 
        dealii::update_jacobians;
    const dealii::UpdateFlags neighbor_face_update_flags = 
        dealii::update_values | 
        dealii::update_gradients | 
        dealii::update_quadrature_points | 
        dealii::update_JxW_values;

    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid.mapping_fe_field));
    const dealii::hp::FECollection<dim>      fe_collection(this->dg->fe_collection);
    const dealii::hp::QCollection<dim-1>     face_quadrature_collection(this->dg->face_quadrature_collection);

    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face_int(
        mapping_collection, 
        fe_collection, 
        face_quadrature_collection, 
        face_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face_ext(
        mapping_collection, 
        fe_collection, 
        face_quadrature_collection, 
        neighbor_face_update_flags);
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface(
        mapping_collection,
        fe_collection,
        face_quadrature_collection,
        face_update_flags);

    const dealii::LinearAlgebra::distributed::Vector<real> solution(this->dg->solution);
    solution.update_ghost_values();

    real anisotropic_threshold_ratio = 3.0;

    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned() || !cell->refine_flag_set()) continue;

        dealii::Point<dim> jump;
        dealii::Point<dim> area;

        const unsigned int mapping_index = 0;
        const unsigned int fe_index = cell->active_fe_index();
        const unsigned int quad_index = fe_index; 

        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            if(cell->face(iface)->at_boundary()) continue;

            const auto face = cell->face(iface);
            
            Assert(cell->neighbor(iface).state() == dealii::IteratorState::valid,
                   dealii::ExcInternalError());
            const auto neig = cell->neighbor(iface);

            if(face->has_children()){
                unsigned int neig2 = cell->neighbor_face_no(iface);
                for(unsigned int subface = 0; subface < face->number_of_children(); ++subface){
                    const auto neig_child = cell->neighbor_child_on_subface(iface, subface);
                    Assert(!neig_child->has_children(), dealii::ExcInternalError());

                    fe_values_collection_subface.reinit(cell,iface,subface,quad_index,mapping_index,fe_index);
                    const dealii::FESubfaceValues<dim,dim> &fe_int_subface = fe_values_collection_subface.get_present_fe_values();
                    std::vector<real> u_face(fe_int_subface.n_quadrature_points);
                    fe_int_subface.get_function_values(solution,u_face);

                    fe_values_collection_face_ext.reinit(neig_child,neig2,quad_index,mapping_index,fe_index);
                    const dealii::FEFaceValues<dim,dim> &fe_ext_face = fe_values_collection_face_ext.get_present_fe_values();
                    std::vector<real> u_neig(fe_ext_face.n_quadrature_points);
                    fe_ext_face.get_function_values(solution,u_neig);

                    const std::vector<real> &JxW = fe_int_subface.get_JxW_values();
                    for(unsigned int iquad = 0; iquad < fe_int_subface.n_quadrature_points; ++iquad){
                        jump[iface/2] += abs(u_face[iquad] - u_neig[iquad]) * JxW[iquad];
                        area[iface/2] += JxW[iquad];
                    }
                }
            }else{
                if(!cell->neighbor_is_coarser(iface)){
                    unsigned int neig2 = cell->neighbor_of_neighbor(iface);

                    fe_values_collection_face_int.reinit(cell,iface,quad_index,mapping_index,fe_index);
                    const dealii::FEFaceValues<dim,dim> &fe_int_face = fe_values_collection_face_int.get_present_fe_values();
                    std::vector<real> u_face(fe_int_face.n_quadrature_points);
                    fe_int_face.get_function_values(solution,u_face);

                    fe_values_collection_face_ext.reinit(neig,neig2,quad_index,mapping_index,fe_index);
                    const dealii::FEFaceValues<dim,dim> &fe_ext_face = fe_values_collection_face_ext.get_present_fe_values();
                    std::vector<real> u_neig(fe_ext_face.n_quadrature_points);
                    fe_ext_face.get_function_values(solution,u_neig);

                    const std::vector<real> &JxW = fe_int_face.get_JxW_values();
                    for(unsigned int iquad = 0; iquad < fe_int_face.n_quadrature_points; ++iquad){
                        jump[iface/2] += abs(u_face[iquad] - u_neig[iquad]) * JxW[iquad];
                        area[iface/2] += JxW[iquad];
                    }
                }else{
                    std::pair<unsigned int, unsigned int> neig_face_subface = cell->neighbor_of_coarser_neighbor(iface);
                    Assert(neig_face_subface.first < dealii::GeometryInfo<dim>::faces_per_cell,
                           dealii::ExcInternalError());
                    Assert(neig_face_subface.second < neig->face(neig_face_subface.first)->number_of_children(),
                           dealii::ExcInternalError());
                    Assert(neig->neighbor_child_on_subface(neig_face_subface.first, neig_face_subface.second) == cell,
                           dealii::ExcInternalError());

                    fe_values_collection_face_int.reinit(cell,iface,quad_index,mapping_index,fe_index);
                    const dealii::FEFaceValues<dim,dim> &fe_int_face = fe_values_collection_face_int.get_present_fe_values();
                    std::vector<real> u_face(fe_int_face.n_quadrature_points);
                    fe_int_face.get_function_values(solution,u_face);

                    fe_values_collection_subface.reinit(neig,neig_face_subface.first,neig_face_subface.second,quad_index,mapping_index,fe_index);
                    const dealii::FESubfaceValues<dim,dim> &fe_ext_subface = fe_values_collection_subface.get_present_fe_values();
                    std::vector<real> u_neig(fe_ext_subface.n_quadrature_points);
                    fe_ext_subface.get_function_values(solution,u_neig);

                    const std::vector<real> &JxW = fe_int_face.get_JxW_values();
                    for(unsigned int iquad = 0; iquad < fe_int_face.n_quadrature_points; ++iquad){
                        jump[iface/2] += abs(u_face[iquad] - u_neig[iquad]) * JxW[iquad];
                        area[iface/2] += JxW[iquad];
                    }
                }
            }
        }

        std::array<real,dim> average_jumps;
        real                 sum_of_average_jumps = 0.0;

        for(unsigned int i = 0; i < dim; ++i){
            average_jumps[i] = jump[i]/area[i];
            sum_of_average_jumps += average_jumps[i];
        }

        for(unsigned int i = 0; i < dim; ++i)
            if(average_jumps[i] > anisotropic_threshold_ratio * (sum_of_average_jumps - average_jumps[i]))
                cell->set_refine_flag(dealii::RefinementCase<dim>::cut_axis(i));
    }    
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Error<dim,nstate,real>::error_indicator()
{
    // TODO: update this to work with p-adaptive schemes (will need proper fe_values for each p)
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
    // call to reconstruct poly
    std::vector<dealii::Tensor<1,dim,real>> A(this->tria->n_active_cells());

    // mapping
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid.mapping_fe_field));

    // using p+1 reconstruction
    const unsigned int rel_order = 1;

    // call to the function to reconstruct the derivatives onto A
    PHiLiP::GridRefinement::ReconstructPoly<dim,real>::reconstruct_directional_derivative(
        this->dg->solution,
        this->dg->dof_handler,
        mapping_collection,
        this->dg->fe_collection,
        this->dg->volume_quadrature_collection,
        this->volume_update_flags,
        rel_order,
        A);

    // looping over the vector and taking the product of the eigenvalues as the size measure
    this->indicator.reinit(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell < this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned()){
            // using an averaging scheme
            // error indicator should relate to the cell-size to the power of p+1
            // this->indicator[cell->active_cell_index()] = pow(cell->measure(), (cell->active_fe_index()+rel_order)/dim);
            // for(unsigned int d = 0; d < dim; ++d)
            //     this->indicator[cell->active_cell_index()] *= A[cell->active_cell_index()][d];

            // using max value of the derivative
            this->indicator[cell->active_cell_index()] = 0.0;
            for(unsigned int d = 0; d < dim; ++d)
                if(this->indicator[cell->active_cell_index()] < A[cell->active_cell_index()][d])
                    this->indicator[cell->active_cell_index()] = A[cell->active_cell_index()][d];

            this->indicator[cell->active_cell_index()] *= pow(cell->measure(), (cell->active_fe_index()+rel_order)/dim);
        }
}

template <int dim, int nstate, typename real>
void GridRefinement_FixedFraction_Residual<dim,nstate,real>::error_indicator()
{
    // // projecting the solution to a finer (p) space
    // // this->coarse_to_fine();

    // // compute the residual and take the Lq norm of each cell
    // // TODO: get polynomial orders and corresponding FE_degree
    // int overintegrate = 10;
    // int poly_degree = 1;
    // dealii::QGauss<dim> quadrature(this->dg->max_degree+overintegrate);
    // dealii::FEValues<dim,dim> fe_values(*(this->dg->high_order_grid.mapping_fe_field), this->dg->fe_collection[poly_degree], quadrature, 
    //     dealii::update_values | dealii::update_JxW_values | dealii::update_quadrature_points);

    // const unsigned int n_quad_pts = fe_values.n_quadrature_points;
    // std::array<double,nstate> soln_at_q;

    // // norm to use 
    // double norm_Lq = this->grid_refinement_param.norm_Lq;

    // // storing the result in 
    // std::vector<dealii::types::global_dof_index> dofs_indices(fe_values.dofs_per_cell);
    // this->indicator.reinit(this->tria->n_active_cells());
    // for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
    //     if(!cell->is_locally_owned()) continue;

    //     fe_values.reinit(cell);
    //     cell->get_dof_indices(dofs_indices);

    //     double cell_error = 0;
    //     for(unsigned int iquad = 0; iquad < n_quad_pts; ++iquad){
    //         std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
    //         for(unsigned int idof = 0; idof < fe_values.dofs_per_cell; ++idof){
    //             const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
    //             soln_at_q[istate] += this->dg->right_hand_side[dofs_indices[idof]] * fe_values.shape_value_component(idof,iquad,istate);
    //         }
            
    //         for(int istate = 0; istate < nstate; ++istate)
    //             cell_error += pow(abs(soln_at_q[istate]), norm_Lq) * fe_values.JxW(iquad);
    //     }
    //     this->indicator[cell->active_cell_index()] = cell_error;
    // }

    // // and projecting it back to the original space
    // // this->fine_to_coarse();
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
void GridRefinement_Continuous<dim,nstate,real>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // store the previous solution space

    // compute the necessary size fields
    if(refinement_type == RefinementTypeEnum::h){
        field_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        field_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        field_hp();
    }

    // generate a new grid
    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    // reinitialize the dg object with new coarse triangulation
    this->dg->reinit();

    // interpolate the solution from the previous solution space
    this->dg->allocate_system();
}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous<dim,nstate,real>::refine_grid_h()
{
    int igrid = 0;

    // now outputting this new field
    std::string write_posname = "grid-"+std::to_string(igrid)+".pos";
    std::ofstream outpos(write_posname);
    GmshOut<dim,real>::write_pos(*(this->tria),this->h_field,outpos);

    std::string write_geoname = "grid-"+std::to_string(igrid)+".geo";
    std::ofstream outgeo(write_geoname);
    GmshOut<dim,real>::write_geo(write_posname,outgeo);

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
void GridRefinement_Continuous<dim,nstate,real>::refine_grid_p()
{
    // physical grid stays the same, apply the update to the p_field
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            cell->set_future_fe_index(round(p_field[cell->active_cell_index()]));
}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous<dim,nstate,real>::refine_grid_hp()
{
    // make a copy of the old grid and build a P1 continuous solution averaged at each of the nodes
    // new P will be the weighted average of the integral over the new cell
}

template <int dim, int nstate, typename real>
real GridRefinement_Continuous<dim,nstate,real>::current_complexity()
{
    real complexity_sum;

    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            complexity_sum += pow(cell->active_fe_index()+1, dim);

    return dealii::Utilities::MPI::sum(complexity_sum, MPI_COMM_WORLD);
}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::field_h()
{
    // checking if the polynomial order is uniform
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        int poly_degree = this->dg->get_min_fe_degree();

        // building error based on exact hessian
        real complexity = pow(poly_degree+1, dim)*this->tria->n_active_cells();
        complexity *= this->grid_refinement_param.complexity_scale;
        complexity += this->grid_refinement_param.complexity_add;

        SizeField<dim,real>::isotropic_uniform(
            *(this->tria),
            *(this->dg->high_order_grid.mapping_fe_field),
            this->dg->fe_collection[poly_degree],
            this->physics->manufactured_solution_function,
            complexity,
            this->h_field);
    }else{
        // call the non-uniform hp-version without the p-update
    }


}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::field_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Error<dim,nstate,real>::field_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::field_h()
{
    // call to reconstruct poly
    std::vector<dealii::Tensor<1,dim,real>> A(this->tria->n_active_cells());

    // mapping
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid.mapping_fe_field));

    // call to the function to reconstruct the derivatives onto A
    PHiLiP::GridRefinement::ReconstructPoly<dim,real>::reconstruct_directional_derivative(
        this->dg->solution,
        this->dg->dof_handler,
        mapping_collection,
        this->dg->fe_collection,
        this->dg->volume_quadrature_collection,
        this->volume_update_flags,
        1, // p+1
        A);

    // vector to store the results
    dealii::Vector<real> B(this->tria->n_active_cells());

    // looping over the vector and taking the product of the eigenvalues as the size measure
    for(auto cell = this->dg->dof_handler.begin_active(); cell < this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned()){
            B[cell->active_cell_index()] = 1.0;
            for(unsigned int d = 0; d < dim; ++d)
                B[cell->active_cell_index()] *= A[cell->active_cell_index()][d];
        }

    // TODO: perform the call to calculate the continuous size field

}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::field_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Hessian<dim,nstate,real>::field_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Residual<dim,nstate,real>::field_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Residual<dim,nstate,real>::field_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Residual<dim,nstate,real>::field_hp(){}

template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Adjoint<dim,nstate,real>::field_h(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Adjoint<dim,nstate,real>::field_p(){}
template <int dim, int nstate, typename real>
void GridRefinement_Continuous_Adjoint<dim,nstate,real>::field_hp(){}

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
