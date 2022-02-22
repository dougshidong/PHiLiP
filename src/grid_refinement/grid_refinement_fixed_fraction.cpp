#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/distributed/grid_refinement.h>

#include "grid_refinement/reconstruct_poly.h"

#include "grid_refinement_fixed_fraction.h"

namespace PHiLiP {

namespace GridRefinement {

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
    using DoFHandlerType   = typename dealii::DoFHandler<dim>;
    using SolutionTransfer = typename MeshTypeHelper<MeshType>::template SolutionTransfer<dim,VectorType,DoFHandlerType>;

    SolutionTransfer solution_transfer(this->dg->dof_handler);
    solution_transfer.prepare_for_coarsening_and_refinement(solution_old);

    this->dg->high_order_grid->prepare_for_coarsening_and_refinement();
    this->dg->triangulation->prepare_coarsening_and_refinement();

    // performing the refinement
    if(refinement_type == RefinementTypeEnum::h){
        refine_grid_h();
    }else if(refinement_type == RefinementTypeEnum::p){
        refine_grid_p();
    }else if(refinement_type == RefinementTypeEnum::hp){
        refine_grid_hp();
    }

    // optionally uncomment to flag all boundaries for refinement
    // refine_boundary_h();

    std::cout << "Checking for aniso option" << std::endl;

    // check for anisotropic h-adaptation
    if(this->grid_refinement_param.anisotropic){
        std::cout << "beginning anistropic flagging" << std::endl;
        anisotropic_h();
    }

    this->tria->execute_coarsening_and_refinement();
    this->dg->high_order_grid->execute_coarsening_and_refinement();

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
                assert(0); // NOT YET IMPLEMENTED
                bool perform_p_refinement_instead = true;

                if(perform_p_refinement_instead)
                {
                    cell->clear_refine_flag();
                    cell->set_active_fe_index(cell->active_fe_index()+1);
                }
            }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::refine_boundary_h()
{
    // setting refinement flag on all boundary cells of the dof_handler
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!cell->is_locally_owned()) continue;
        
        // looping over the faces to check if any are at the boundary
        for(unsigned int face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face){
            if(cell->face(face)->at_boundary()){
                cell->set_refine_flag();
                break;
            }
        }
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::smoothness_indicator()
{
    // reads the options and determines the proper smoothness indicator
    smoothness.reinit(this->tria->n_active_cells());
    
    // NOT IMPLEMENTED 
    // cannot assert(0) as its still callled by the output function

    // placeholder function for future added hp-refinement threshold function
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::anisotropic_h()
{
    using AnisoIndicator = typename PHiLiP::Parameters::GridRefinementParam::AnisoIndicator;
    AnisoIndicator aniso_indicator = this->grid_refinement_param.anisotropic_indicator;

    // selecting the anisotropic method to be used
    if(aniso_indicator == AnisoIndicator::jump_based){
        anisotropic_h_jump_based();
    }else if(aniso_indicator == AnisoIndicator::reconstruction_based){
        anisotropic_h_reconstruction_based();
    }
}

// based on dealii step-30
template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::anisotropic_h_jump_based()
{
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));
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

    real anisotropic_threshold_ratio = this->grid_refinement_param.anisotropic_threshold_ratio;

    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!(cell->is_locally_owned()) || !(cell->refine_flag_set())) continue;

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

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::anisotropic_h_reconstruction_based()
{
    // mapping
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));

    // using p+1 reconstruction
    const unsigned int rel_order = 1;

    // generating object to reconstruct derivatives
    ReconstructPoly<dim,nstate,real> reconstruct_poly(
        this->dg->dof_handler,
        mapping_collection,
        this->dg->fe_collection,
        this->dg->volume_quadrature_collection,
        this->volume_update_flags);
    
    // call to reconstruct the derivatives
    reconstruct_poly.reconstruct_chord_derivative(
        this->dg->solution,
        rel_order);

    // controls degree of anisotropy required to flag cell for cut in x
    real anisotropic_threshold_ratio = this->grid_refinement_param.anisotropic_threshold_ratio;

    // looping over flagged cells to compute anisotropic indicators
    for(auto cell = this->dg->dof_handler.begin_active(); cell != this->dg->dof_handler.end(); ++cell){
        if(!(cell->is_locally_owned()) || !(cell->refine_flag_set())) continue;

        // vector of chord and midface positions
        std::array<std::pair<dealii::Tensor<1,dim,real>, dealii::Tensor<1,dim,real>>,dim> chord_nodes;
        std::array<dealii::Tensor<1,dim,real>,dim> chord_vec;

        // computing the chord associated with each 
        for(unsigned int vertex = 0; vertex < dealii::GeometryInfo<dim>::vertices_per_cell; ++vertex)
            for(unsigned int i = 0; i < dim; ++i)
                if(vertex % (unsigned int)pow(2,i) == 0){
                    chord_nodes[i].first  += cell->vertex(vertex);
                }else{
                    chord_nodes[i].second += cell->vertex(vertex);
                }
        
        // averaging the nodes to get the coord
        for(unsigned int i = 0; i < dim; ++i){
            chord_nodes[i].first  /= pow(2,dim-1);
            chord_nodes[i].second /= pow(2,dim-1);
        }

        // computing the vectors
        for(unsigned int i = 0; i < dim; ++i){
            chord_vec[i] = chord_nodes[i].second - chord_nodes[i].first;
        }

        // computing the indicator scaled to the chord length
        dealii::Tensor<1,dim,real> indicator;
        for(unsigned int i = 0; i < dim; ++i)
            indicator[i] += reconstruct_poly.derivative_value[cell->active_cell_index()][i] * pow(chord_vec[i].norm(), cell->active_fe_index()+rel_order);

        real sum = 0.0;
        for(unsigned int i = 0; i < dim; ++i)
            sum += indicator[i];

        // checking if it meets the criteria for anisotropy
        // equivalent to the form used in jump_based indicator
        for(unsigned int i = 0; i < dim; ++i)
            if(indicator[i] / sum > anisotropic_threshold_ratio/(1.0 + anisotropic_threshold_ratio))
                cell->set_refine_flag(dealii::RefinementCase<dim>::cut_axis(i));
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::error_indicator()
{
    // different cases depending on indicator types
    using ErrorIndicatorEnum = PHiLiP::Parameters::GridRefinementParam::ErrorIndicator;
    if(this->error_indicator_type == ErrorIndicatorEnum::error_based){
        error_indicator_error();
    }else if(this->error_indicator_type == ErrorIndicatorEnum::hessian_based){
        error_indicator_hessian();
    }else if(this->error_indicator_type == ErrorIndicatorEnum::residual_based){
        error_indicator_residual();
    }else if(this->error_indicator_type == ErrorIndicatorEnum::adjoint_based){
        error_indicator_adjoint();
    }else{
        std::cout << "Warning: error_indicator_type not recognized." << std::endl;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::error_indicator_error()
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
    dealii::FEValues<dim,dim> fe_values(*(this->dg->high_order_grid->mapping_fe_field), this->dg->fe_collection[poly_degree], quadrature, 
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
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::error_indicator_hessian()
{
    // TODO: Feature based, should use the reconstructed next mode as an indicator

    // mapping
    const dealii::hp::MappingCollection<dim> mapping_collection(*(this->dg->high_order_grid->mapping_fe_field));

    // using p+1 reconstruction
    const unsigned int rel_order = 1;

    // call to the function to reconstruct the derivatives onto A
    ReconstructPoly<dim,nstate,real> reconstruct_poly(
        this->dg->dof_handler,
        mapping_collection,
        this->dg->fe_collection,
        this->dg->volume_quadrature_collection,
        this->volume_update_flags);

    reconstruct_poly.reconstruct_directional_derivative(
        this->dg->solution,
        rel_order);

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
            for(unsigned int d = 0; d < dim; ++d){

                // getting derivative value
                real derivative_value = reconstruct_poly.derivative_value[cell->active_cell_index()][d];

                // check and update
                if(this->indicator[cell->active_cell_index()] < derivative_value)
                    this->indicator[cell->active_cell_index()] = derivative_value;

            }

            this->indicator[cell->active_cell_index()] *= pow(cell->measure(), (cell->active_fe_index()+rel_order)/dim);
        }
}

template <int dim, int nstate, typename real, typename MeshType>
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::error_indicator_residual()
{
    // NOT IMPLEMENTED
    assert(0);

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
void GridRefinement_FixedFraction<dim,nstate,real,MeshType>::error_indicator_adjoint()
{
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
std::vector< std::pair<dealii::Vector<real>, std::string> > GridRefinement_FixedFraction<dim,nstate,real,MeshType>::output_results_vtk_method()
{
    std::vector< std::pair<dealii::Vector<real>, std::string> > data_out_vector;

    // error indicator for adaptation
    error_indicator();
    data_out_vector.push_back(
        std::make_pair(
            indicator,
            "error_indicator"));

    smoothness_indicator();
    data_out_vector.push_back(
        std::make_pair(
            indicator,
            "smoothness_indicator"));

    return data_out_vector;
}

// dealii::Triangulation<PHILIP_DIM>
template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

// dealii::parallel::shared::Triangulation<PHILIP_DIM>
template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM != 1
// dealii::parallel::distributed::Triangulation<PHILIP_DIM>
template class GridRefinement_FixedFraction<PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class GridRefinement_FixedFraction<PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace GridRefinement

} // namespace PHiLiP
