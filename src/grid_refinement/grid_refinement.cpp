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
#include "dg/high_order_grid.h"

#include "functional/functional.h"
#include "functional/adjoint.h"

#include "physics/physics.h"

#include "post_processor/physics_post_processor.h"

#include "grid_refinement/gmsh_out.h"
#include "grid_refinement/size_field.h"
#include "grid_refinement/reconstruct_poly.h"
#include "grid_refinement.h"

namespace PHiLiP {

namespace GridRefinement {

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid()
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

    // increase the count
    this->iteration++;
}

// functions for the refinement calls for each of the classes
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid_h()
{
    this->tria->set_all_refine_flags();
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid_p()
{
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned() && cell->active_fe_index()+1 <= this->dg->max_degree)
            cell->set_future_fe_index(cell->active_fe_index()+1);
    
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::refine_grid_hp()
{
    refine_grid_h();
    refine_grid_p();
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::refine_grid()
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

    using VectorType       = typename dealii::LinearAlgebra::distributed::Vector<double>;
    using DoFHandlerType   = typename dealii::hp::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(this->dg->dof_handler);
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

    std::cout << "Checking for aniso option" << std::endl;

    // check for anisotropic h-adaptation
    if(!this->grid_refinement_param.isotropic){
        std::cout << "beginning anistropic flagging" << std::endl;
        anisotropic_h();
    }

    this->tria->execute_coarsening_and_refinement();
    this->dg->high_order_grid.execute_coarsening_and_refinement();

    // transfering the solution from solution_old
    this->dg->allocate_system();
    this->dg->solution.zero_out_ghosts();

    if constexpr (std::is_same_v<typename dealii::SolutionTransfer<dim,VectorType,DoFHandlerType>, 
                                 decltype(solution_transfer)>){
        solution_transfer.interpolate(solution_old, this->dg->solution);
    }else{
        solution_transfer.interpolate(this->dg->solution);
    }

    this->dg->solution.update_ghost_values();

    // increase the count
    this->iteration++;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::refine_grid_h()
{
    // Performing the call for refinement
// #if PHILIP_DIM==1
    dealii::GridRefinement::refine_and_coarsen_fixed_number(
        *(this->tria),
        this->indicator,
        this->grid_refinement_param.refinement_fraction,
        this->grid_refinement_param.coarsening_fraction);
// #else
//     dealii::parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
//         *(this->tria),
//         this->indicator,
//         this->grid_refinement_param.refinement_fraction,
//         this->grid_refinement_param.coarsening_fraction);
// #endif
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::refine_grid_p()
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

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::refine_grid_hp()
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

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::smoothness_indicator()
{
    // reads the options and determines the proper smoothness indicator
    smoothness.reinit(this->tria->n_active_cells());
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::anisotropic_h()
{
    // based on dealii step-30
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid.mapping_fe_field));
    const dealii::hp::FECollection<dim>      fe_collection(this->dg->fe_collection);
    const dealii::hp::QCollection<dim-1>     face_quadrature_collection(this->dg->face_quadrature_collection);

    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face_int(
        mapping_collection, 
        fe_collection, 
        face_quadrature_collection, 
        this->face_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face_ext(
        mapping_collection, 
        fe_collection, 
        face_quadrature_collection, 
        this->neighbor_face_update_flags);
    dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface(
        mapping_collection,
        fe_collection,
        face_quadrature_collection,
        this->face_update_flags);

    const dealii::LinearAlgebra::distributed::Vector<real> solution(this->dg->solution);
    solution.update_ghost_values();

    real anisotropic_threshold_ratio = 1.0;//3.0;
    std::cout << "testing testing 123" << std::endl;

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
            if(average_jumps[i] > anisotropic_threshold_ratio * (sum_of_average_jumps - average_jumps[i])){
                cell->set_refine_flag(dealii::RefinementCase<dim>::cut_axis(i));
                std::cout << "setting the refine flag on axis: " << i << std::endl;
            }
    }    
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction_Error<dim,nstate,real,MeshType>::error_indicator()
{
    // TODO: update this to work with p-adaptive schemes (will need proper fe_values for each p)
    // see dg.cpp
    // const auto mapping = (*(high_order_grid.mapping_fe_field));
    // dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    // dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, fe_collection, this->dg->volume_quadrature_collection, this->dg->volume_update_flags);

    // use manufactured solution to measure the cell-wise error (overintegrate)
    int overintegrate = 10;
    int poly_degree =  this->dg->get_max_fe_degree(); 
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

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction_Hessian<dim,nstate,real,MeshType>::error_indicator()
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

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction_Residual<dim,nstate,real,MeshType>::error_indicator()
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

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction_Adjoint<dim,nstate,real,MeshType>::error_indicator()
{
    // reinitializing the adjoint with current values
    this->adjoint->reinit();

    // evaluating the functional derivatives and adjoint
    this->adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,real,MeshType>::AdjointStateEnum::fine);
    this->adjoint->fine_grid_adjoint();
    
    // reinitializing the error indicator vector
    this->indicator.reinit(this->adjoint->dg->triangulation->n_active_cells());
    this->indicator = this->adjoint->dual_weighted_residual();

    // return to the coarse grid
    this->adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,real,MeshType>::AdjointStateEnum::coarse);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // store the previous solution space

    // compute the necessary size fields
    field();

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

    // increase the count
    this->iteration++;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_h()
{
    const int iproc = dealii::Utilities::MPI::this_mpi_process(this->mpi_communicator);
    
    // now outputting this new field
    std::string write_posname = "grid-" + 
                                dealii::Utilities::int_to_string(this->iteration, 4) + "." + 
                                dealii::Utilities::int_to_string(iproc, 4) + ".pos";
    std::ofstream outpos(write_posname);
    GmshOut<dim,real>::write_pos(*(this->tria),this->h_field,outpos);

    // writing the geo file on the 1st processor and running
    std::string output_name = "grid-" + 
                              dealii::Utilities::int_to_string(this->iteration, 4) + ".msh";
    if(iproc == 0){
        // generating a vector of pos file names
        std::vector<std::string> posname_vec;
        for(unsigned int iproc = 0; iproc < dealii::Utilities::MPI::n_mpi_processes(this->mpi_communicator); ++iproc)
            posname_vec.push_back("grid-" + 
                                  dealii::Utilities::int_to_string(this->iteration, 4) + "." + 
                                  dealii::Utilities::int_to_string(iproc, 4) + ".pos");

        std::string write_geoname = "grid-" + 
                                    dealii::Utilities::int_to_string(this->iteration, 4) + ".geo";
        std::ofstream outgeo(write_geoname);
        GmshOut<dim,real>::write_geo(posname_vec,outgeo);

        std::cout << "Command is: " << ("/usr/local/include/gmsh-master/build/gmsh " + write_geoname + " -2 -o " + output_name).c_str() << '\n';
        int ret = std::system(("/usr/local/include/gmsh-master/build/gmsh " + write_geoname + " -2 -o " + output_name).c_str());
        (void) ret;
    }

    // barrier
    MPI_Barrier(this->mpi_communicator);
    
    // loading the mesh on all processors
    this->tria->clear();
    dealii::GridIn<dim> gridin;
    gridin.attach_triangulation(*(this->tria));
    std::ifstream f(output_name);
    gridin.read_msh(f);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_p()
{
    // physical grid stays the same, apply the update to the p_field
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            cell->set_future_fe_index(round(p_field[cell->active_cell_index()]));
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::refine_grid_hp()
{
    // make a copy of the old grid and build a P1 continuous solution averaged at each of the nodes
    // new P will be the weighted average of the integral over the new cell
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::field()
{
    using RefinementTypeEnum = PHiLiP::Parameters::GridRefinementParam::RefinementType;
    RefinementTypeEnum refinement_type = this->grid_refinement_param.refinement_type;

    // updating the target complexity for this iteration
    target_complexity();

    // compute the necessary size fields
    if(refinement_type == RefinementTypeEnum::h){
        field_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        field_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        field_hp();
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::target_complexity()
{
    // if the complexity vector needs to be expanded
    while(complexity_vector.size() <= this->iteration)
        complexity_vector.push_back(
            complexity_vector.back() 
          * this->grid_refinement_param.complexity_scale 
          + this->grid_refinement_param.complexity_add);

    // copy the current iteration into the complexity target
    complexity_target = complexity_vector[this->iteration];

    std::cout << "Complexity target = " << complexity_target << std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
real GridRefinement_Continuous<dim,nstate,real,MeshType>::current_complexity()
{
    real complexity_sum;

    // two possible cases
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        // case 1: uniform p-order, complexity relates to total dof
        unsigned int poly_degree = this->dg->get_min_fe_degree();
        return pow(poly_degree+1, dim) * this->tria->n_global_active_cells(); //TODO: check how this behaves in MPI
    }else{
        // case 2: non-uniform p-order, complexity related to the local sizes
        for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
            if(cell->is_locally_owned())
                complexity_sum += pow(cell->active_fe_index()+1, dim);
    }

    return dealii::Utilities::MPI::sum(complexity_sum, MPI_COMM_WORLD);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_h()
{
    // gets the current size and copy it into field_h
    // for isotropic, sets the size to be the h = volume ^ (1/dim)
    h_field.reinit(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            this->h_field[cell->active_cell_index()] = pow(cell->measure(), 1.0/dim);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p()
{
    // gets the current polynomiala distribution and copies it into field_p
    p_field.reinit(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell)
        if(cell->is_locally_owned())
            this->p_field[cell->active_cell_index()] = cell->active_fe_index();
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Error<dim,nstate,real,MeshType>::field_h()
{
    real q = 2.0;

    dealii::Vector<real> B(this->tria->n_active_cells());
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;

        // getting the central coordinate as average of vertices
        dealii::Point<dim,real> center_point = cell->center();

        // evaluating the Hessian at this point (using default state)
        dealii::SymmetricTensor<2,dim,real> hessian = 
            this->physics->manufactured_solution_function->hessian(center_point);

        // assuming 2D for now
        // TODO: check generalization of this for different dimensions with eigenvalues
        if(dim == 2)
            B[cell->active_cell_index()] = 
                pow(abs(hessian[0][0]*hessian[1][1] - hessian[0][1]*hessian[1][0]), q/2);
    }

    // checking if the polynomial order is uniform
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        unsigned int poly_degree = this->dg->get_min_fe_degree();

        // building error based on exact hessian
        SizeField<dim,real>::isotropic_uniform(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            this->h_field,
            poly_degree);
    }else{
        // call the non-uniform hp-version without the p-update
        GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p();

        // mapping
        const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid.mapping_fe_field));

        SizeField<dim,real>::isotropic_h(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags,
            this->h_field,
            this->p_field);
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Error<dim,nstate,real,MeshType>::field_p(){}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Error<dim,nstate,real,MeshType>::field_hp(){}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Hessian<dim,nstate,real,MeshType>::field_h()
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

    // setting the current p-field

    // TODO: perform the call to calculate the continuous size field

    // checking if the polynomial order is uniform
    if(this->dg->get_min_fe_degree() == this->dg->get_max_fe_degree()){
        unsigned int poly_degree = this->dg->get_min_fe_degree();

        SizeField<dim,real>::isotropic_uniform(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            this->h_field,
            poly_degree);
    }else{
        // the case of non-uniform p
        GridRefinement_Continuous<dim,nstate,real,MeshType>::get_current_field_p();

        SizeField<dim,real>::isotropic_h(
            this->complexity_target,
            B,
            this->dg->dof_handler,
            mapping_collection,
            this->dg->fe_collection,
            this->dg->volume_quadrature_collection,
            this->volume_update_flags,
            this->h_field,
            this->p_field);
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Hessian<dim,nstate,real,MeshType>::field_p(){}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Hessian<dim,nstate,real,MeshType>::field_hp(){}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Residual<dim,nstate,real,MeshType>::field_h(){}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Residual<dim,nstate,real,MeshType>::field_p(){}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Residual<dim,nstate,real,MeshType>::field_hp(){}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Adjoint<dim,nstate,real,MeshType>::field_h(){}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Adjoint<dim,nstate,real,MeshType>::field_p(){}
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous_Adjoint<dim,nstate,real,MeshType>::field_hp(){}

// output results functions
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk(const unsigned int iref)
{
    // creating the data out stream
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> data_out;
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
    std::vector<std::string> dIdw_names_coarse;
    std::vector<std::string> adjoint_names_coarse;
    std::vector<std::string> dIdw_names_fine;
    std::vector<std::string> adjoint_names_fine;
    if(adjoint)
        output_results_vtk_adjoint(data_out, dIdw_names_coarse, adjoint_names_coarse, dIdw_names_fine, adjoint_names_fine);

    // plotting the error compared to the manufactured solution
    dealii::Vector<real> l2_error_vec;
    if(physics && physics->manufactured_solution_function)
        output_results_vtk_error(data_out, l2_error_vec);

    // virtual method to call each refinement type 
    // passing an std::array to copy and hold references to method specific values
    std::array<dealii::Vector<real>,MAX_METHOD_VEC> dat_vec_vec;
    output_results_vtk_method(data_out, dat_vec_vec);

    // performing the ouput on each core
    const int iproc = dealii::Utilities::MPI::this_mpi_process(mpi_communicator);

    // default
    data_out.build_patches();

    // curved
    // typename dealii::DataOut<dim,dealii::hp::DoFHandler<dim>>::CurvedCellRegion curved 
    //     = dealii::DataOut<dim,dealii::hp::DoFHandler<dim>>::CurvedCellRegion::curved_inner_cells;
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
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
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
    data_out.add_data_vector(subdomain, "subdomain", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // Output the polynomial degree in each cell
    dg->dof_handler.get_active_fe_indices(active_fe_indices);
    dealii::Vector<double> active_fe_indices_dealiivector(active_fe_indices.begin(), active_fe_indices.end());
    cell_poly_degree = active_fe_indices_dealiivector;

    data_out.add_data_vector(cell_poly_degree, "PolynomialDegree", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    for(int s=0;s<nstate;++s) {
        std::string varname = "residual" + dealii::Utilities::int_to_string(s,1);
        residual_names.push_back(varname);
    }

    data_out.add_data_vector(dg->right_hand_side, residual_names, dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_functional(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out)
{
    // nothing here for now, could plot the contributions or weighting function
    (void) data_out;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_physics(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out)
{
    // TODO: plot the function value, gradient, tensor, etc.
    (void) data_out;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_adjoint(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
    std::vector<std::string> &                         dIdw_names_coarse,
    std::vector<std::string> &                         adjoint_names_coarse,
    std::vector<std::string> &                         dIdw_names_fine,
    std::vector<std::string> &                         adjoint_names_fine)
{
    // starting with coarse grid results
    adjoint->reinit();
    adjoint->coarse_grid_adjoint();

    for(int s=0;s<nstate;++s) {
        std::string varname = "dIdw" + dealii::Utilities::int_to_string(s,1) + "_coarse";
        dIdw_names_coarse.push_back(varname);
    }
    data_out.add_data_vector(adjoint->dIdw_coarse, dIdw_names_coarse, dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    for(int s=0;s<nstate;++s) {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1) + "_coarse";
        adjoint_names_coarse.push_back(varname);
    }
    data_out.add_data_vector(adjoint->adjoint_coarse, adjoint_names_coarse, dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    // TODO: add obtaining this on the fine  grids (check if dataout still behaves properly)
    // next for fine grid results
    adjoint->fine_grid_adjoint();
    adjoint->dual_weighted_residual();
    
    for(int s=0;s<nstate;++s) {
        std::string varname = "dIdw" + dealii::Utilities::int_to_string(s,1) + "_fine";
        dIdw_names_fine.push_back(varname);
    }
    // data_out.add_data_vector(adjoint->dIdw_fine, dIdw_names_fine, dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);

    for(int s=0;s<nstate;++s) {
        std::string varname = "psi" + dealii::Utilities::int_to_string(s,1) + "_fine";
        adjoint_names_fine.push_back(varname);
    }
    // data_out.add_data_vector(adjoint->adjoint_fine, adjoint_names_fine, dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_dof_data);
    
    data_out.add_data_vector(adjoint->dual_weighted_residual_fine, "DWR", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // returning to original state
    adjoint->convert_to_state(PHiLiP::Adjoint<dim,nstate,double,MeshType>::AdjointStateEnum::coarse);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinementBase<dim,nstate,real,MeshType>::output_results_vtk_error(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
    dealii::Vector<real> &                             l2_error_vec)
{
    int overintegrate = 10;
    int poly_degree = dg->get_max_fe_degree();
    dealii::QGauss<dim> quad_extra(dg->max_degree+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg->high_order_grid.mapping_fe_field), dg->fe_collection[poly_degree], quad_extra, 
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

    data_out.add_data_vector(l2_error_vec, "l2_error", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    //TODO: could plot the actual error distribution rather than cell-wise
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Uniform<dim,nstate,real,MeshType>::output_results_vtk_method(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
    std::array<dealii::Vector<real>,MAX_METHOD_VEC> & dat_vec_vec)
{
    // nothing special to do here
    (void) data_out;
    (void) dat_vec_vec;
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::output_results_vtk_method(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
    std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec)
{
    // error indicator for adaptation
    error_indicator();
    dat_vec_vec[0] = indicator;
    // dat_vec_vec.push_back(indicator);
    data_out.add_data_vector(dat_vec_vec[0], "error_indicator", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    smoothness_indicator();
    dat_vec_vec[1] = indicator;
    // dat_vec_vec.push_back(smoothness);
    data_out.add_data_vector(dat_vec_vec[1], "smoothness_indicator", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_Continuous<dim,nstate,real,MeshType>::output_results_vtk_method(
    dealii::DataOut<dim, dealii::hp::DoFHandler<dim>> &data_out,
    std::array<dealii::Vector<real>,MAX_METHOD_VEC> &  dat_vec_vec)
{
    // getting the current field sizes
    get_current_field_h();
    dat_vec_vec[0] = h_field;
    // dat_vec_vec.push_back(h_field);
    data_out.add_data_vector(dat_vec_vec[0], "h_field_curr", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    get_current_field_p();
    dat_vec_vec[1] = p_field;
    // dat_vec_vec.push_back(p_field);
    data_out.add_data_vector(dat_vec_vec[1], "p_field_curr", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);

    // computing the (next) update to the fields
    field();
    dat_vec_vec[2] = h_field; 
    // dat_vec_vec.push_back(h_field);
    data_out.add_data_vector(dat_vec_vec[2], "h_field_next", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
    dat_vec_vec[3] = p_field;
    // dat_vec_vec.push_back(p_field);
    data_out.add_data_vector(dat_vec_vec[3], "p_field_next", dealii::DataOut_DoFData<dealii::hp::DoFHandler<dim>,dim>::DataVectorType::type_cell_data);
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
        return std::make_shared< GridRefinement_FixedFraction_Adjoint<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::adjoint_based){
        return std::make_shared< GridRefinement_Continuous_Adjoint<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }

    // dg + physics
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_FixedFraction_Hessian<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction_Error<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous_Hessian<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous_Error<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }

    // dg
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction_Residual<dim,nstate,real,MeshType> >(gr_param, adj, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous_Residual<dim,nstate,real,MeshType> >(gr_param, adj, physics);
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
        return std::make_shared< GridRefinement_FixedFraction_Hessian<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction_Error<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous_Hessian<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous_Error<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }

    // dg
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction_Residual<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous_Residual<dim,nstate,real,MeshType> >(gr_param, dg, physics, functional);
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
        return std::make_shared< GridRefinement_FixedFraction_Hessian<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::fixed_fraction &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_FixedFraction_Error<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::hessian_based){
        return std::make_shared< GridRefinement_Continuous_Hessian<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::error_based){
        return std::make_shared< GridRefinement_Continuous_Error<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }

    // dg
    if(refinement_method == RefinementMethodEnum::fixed_fraction &&
       error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_FixedFraction_Residual<dim,nstate,real,MeshType> >(gr_param, dg, physics);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous_Residual<dim,nstate,real,MeshType> >(gr_param, dg, physics);
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
        return std::make_shared< GridRefinement_FixedFraction_Residual<dim,nstate,real,MeshType> >(gr_param, dg);
    }else if(refinement_method == RefinementMethodEnum::continuous &&
             error_indicator   == ErrorIndicatorEnum::residual_based){
        return std::make_shared< GridRefinement_Continuous_Residual<dim,nstate,real,MeshType> >(gr_param, dg);
    }else if(refinement_method == RefinementMethodEnum::uniform){
        return std::make_shared< GridRefinement_Uniform<dim,nstate,real,MeshType> >(gr_param, dg);
    }

    std::cout << "Invalid GridRefinement." << std::endl;

    return nullptr;
}

// large amount of templating to be done, move to an .inst file
// try reducing this with BOOST

// dealii::Triangulation<PHILIP_DIM>
template class GridRefinementBase<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinementBase<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Uniform<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Error<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Residual<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

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

template class GridRefinement_Uniform<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Error<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Residual<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

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

template class GridRefinement_Uniform<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Uniform<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Error<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Hessian<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Residual<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction_Adjoint<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Error<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Error<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Hessian<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Residual<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Residual<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_Continuous_Adjoint<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class GridRefinementFactory<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinementFactory<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace GridRefinement

} // namespace PHiLiP
