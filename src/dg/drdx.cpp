#include "dg.h"

namespace PHiLiP {

template <int dim, typename real>
dealii::SparsityPattern DGBase<dim,real>::get_dRdX_sparsity_pattern () {

    const unsigned n_residuals = dof_handler.n_dofs();
    const unsigned n_nodes_coeff = high_order_grid.dof_handler_grid.n_dofs();
    const unsigned n_nodes_per_cell = high_order_grid.dof_handler_grid.get_fe_collection().max_dofs_per_cell();
    dealii::SparsityPattern sparsity_pattern(n_residuals, n_nodes_coeff, n_nodes_per_cell);

    std::vector<dealii::types::global_dof_index> resi_indices;
    std::vector<dealii::types::global_dof_index> node_indices;
    for (auto cell = dof_handler.begin_active(); cell != dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        const unsigned int level = cell->level();
        const unsigned int index = cell->index();

        const unsigned int n_resi_cell = fe_collection[cell->active_fe_index()].n_dofs_per_cell();
        resi_indices.resize(n_resi_cell);
        cell->get_dof_indices (resi_indices);

        for (auto cell_grid = high_order_grid.dof_handler_grid.begin_active(); cell_grid != dof_handler.end(); ++cell_grid) {
            // Brute force search for the same cell.
            // There must be a better way
            if (!cell_grid->is_locally_owned()) continue;
            if (cell_grid->level() != level) continue;
            if (cell_grid->index() != index) continue;

            const unsigned int n_node_cell = high_order_grid.fe_system.n_dofs_per_cell();
            node_indices.resize(n_node_cell);
            cell_grid->get_dof_indices (node_indices);

            for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
                sparsity_pattern.add_entries(*resi_row, node_indices.begin(), node_indices.end());
            }

            break;
        }

    } // end of cell loop
    sparsity_pattern.compress();

    return sparsity_pattern;
}

//template <int dim, typename real>
//dealii::TrilinosWrappers::SparseMatrix DGBase<dim,real>::get_dRdX_finite_differences (dealii::SparsityPattern dRdX_sparsity_pattern) {
//
//    const double pertubation = 1e-8;
//
//    dealii::TrilinosWrappers::SparseMatrix dRdX;
//    dRdX.reinit(locally_owned_dofs, dRdX_sparsity_pattern, mpi_communicator);
//
//    // For now assume same polynomial degree across domain
//    const unsigned int max_dofs_per_cell = dof_handler.get_fe_collection().max_dofs_per_cell();
//    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
//    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
//
//    for (auto current_cell = dof_handler.begin_active(); current_cell != dof_handler.end(); ++current_cell) {
//        if (!current_cell->is_locally_owned()) continue;
//
//        
//        // Current reference element related to this physical cell
//        const unsigned int mapping_index = 0;
//        const unsigned int fe_index_curr_cell = current_cell->active_fe_index();
//        const unsigned int quad_index = fe_index_curr_cell;
//        const dealii::FESystem<dim,dim> &current_fe_ref = fe_collection[fe_index_curr_cell];
//        const unsigned int curr_cell_degree = current_fe_ref.tensor_degree();
//        const unsigned int n_dofs_curr_cell = current_fe_ref.n_dofs_per_cell();
//
//        // Local vector contribution from each cell
//        dealii::Vector<double> current_cell_rhs (n_dofs_curr_cell); // Defaults to 0.0 initialization
//
//        // Obtain the mapping from local dof indices to global dof indices
//        current_dofs_indices.resize(n_dofs_curr_cell);
//        current_cell->get_dof_indices (current_dofs_indices);
//
//
//        // Cell diameter used to scale perturbation
//        double cell_diameter = current_cell.diameter();
//        for (auto resi_row = current_dofs_indices.begin(); resi_row != current_dofs_indices.end(); ++resi_row) {
//            const int irow_glob = *resi_row;
//            const int n_cols = dRdX_sparsity_pattern.row_length(irow_glob);
//            for (int icol=0; icol < n_cols; ++icol) {
//                const int icol_glob = dRdX_sparsity_pattern.column_number(irow_glob, icol);
//
//                double old_node = high_order_grid.nodes[icol_glob];
//                double dx = cell_diameter * perturbation;
//                double high_order_grid.nodes[icol_glob] += dx;
//                high_order_grid.nodes.update_ghost_values();
//
//                double high_order_grid.nodes[icol_glob] = old_node;
//                high_order_grid.nodes.update_ghost_values();
//            }
//        }
//
//        //dealii::hp::MappingCollection<dim> mapping_collection(*(high_order_grid.mapping_fe_field));
//        //const dealii::MappingManifold<dim,dim> mapping;
//        //const dealii::MappingQ<dim,dim> mapping(max_degree+1);
//        const auto mapping = (*(high_order_grid.mapping_fe_field));
//        dealii::hp::MappingCollection<dim> mapping_collection(mapping);
//
//        dealii::hp::FEValues<dim,dim>        fe_values_collection_volume (mapping_collection, fe_collection, volume_quadrature_collection, this->volume_update_flags); ///< FEValues of volume.
//        dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_int (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of interior face.
//        dealii::hp::FEFaceValues<dim,dim>    fe_values_collection_face_ext (mapping_collection, fe_collection, face_quadrature_collection, this->neighbor_face_update_flags); ///< FEValues of exterior face.
//        dealii::hp::FESubfaceValues<dim,dim> fe_values_collection_subface (mapping_collection, fe_collection, face_quadrature_collection, this->face_update_flags); ///< FEValues of subface.
//
//        dealii::hp::FEValues<dim,dim>        fe_values_collection_volume_lagrange (mapping_collection, fe_collection_lagrange, volume_quadrature_collection, this->volume_update_flags);
//
//        // fe_values_collection.reinit(current_cell, quad_collection_index, mapping_collection_index, fe_collection_index)
//        fe_values_collection_volume.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
//        const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
//
//
//        dealii::TriaIterator<dealii::CellAccessor<dim,dim>> cell_iterator = static_cast<dealii::TriaIterator<dealii::CellAccessor<dim,dim>> > (current_cell);
//        //if (!(all_parameters->use_weak_form)) fe_values_collection_volume_lagrange.reinit (current_cell, quad_index, mapping_index, fe_index_curr_cell);
//        fe_values_collection_volume_lagrange.reinit (cell_iterator, quad_index, mapping_index, fe_index_curr_cell);
//        const dealii::FEValues<dim,dim> &fe_values_lagrange = fe_values_collection_volume_lagrange.get_present_fe_values();
//        assemble_volume_terms_explicit (fe_values_volume, current_dofs_indices, current_cell_rhs, fe_values_lagrange);
//
//        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
//
//            auto current_face = current_cell->face(iface);
//            auto neighbor_cell = current_cell->neighbor(iface);
//
//            // See tutorial step-30 for breakdown of 4 face cases
//
//            // Case 1:
//            // Face at boundary
//            if (current_face->at_boundary() && !current_cell->has_periodic_neighbor(iface) ) {
//
//                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
//
//                if(current_face->at_boundary() && all_parameters->use_periodic_bc == true && dim == 1) //using periodic BCs (for 1d)
//                {
//                    int cell_index  = current_cell->index();
//                    //int cell_index = current_cell->index();
//                    if (cell_index == 0 && iface == 0)
//                    {
//                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
//                        neighbor_cell = dof_handler.begin_active();
//                        for (unsigned int i = 0 ; i < triangulation->n_active_cells() - 1; ++i)
//                        {
//                            ++neighbor_cell;
//                        }
//                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
//                         const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
//                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
//                        const unsigned int mapping_index_neigh_cell = 0;
//
//                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1,quad_index_neigh_cell,mapping_index_neigh_cell,fe_index_neigh_cell);
//
//                    }
//                    else if (cell_index == (int) triangulation->n_active_cells() - 1 && iface == 1)
//                    {
//                        fe_values_collection_face_int.reinit(current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
//                        neighbor_cell = dof_handler.begin_active();
//                        neighbor_cell->get_dof_indices(neighbor_dofs_indices);
//                        const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
//                        const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
//                        const unsigned int mapping_index_neigh_cell = 0;
//                        fe_values_collection_face_ext.reinit(neighbor_cell,(iface == 1) ? 0 : 1, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell); //not sure how changing the face number would work in dim!=1-dimensions.
//                    }
//
//                    //std::cout << "cell " << current_cell->index() << "'s " << iface << "th face has neighbour: " << neighbor_cell->index() << std::endl;
//                    const int neighbor_face_no = (iface ==1) ? 0:1;
//                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
//
//                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
//                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                    const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
//                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
//                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
//
//                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization
//
//
//                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
//                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
//                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
//                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);
//
//                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
//                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
//                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);
//
//                    const real penalty1 = deg1sq / vol_div_facearea1;
//                    const real penalty2 = deg2sq / vol_div_facearea2;
//
//                    real penalty = 0.5 * ( penalty1 + penalty2 );
//
//                    assemble_face_term_explicit (
//                                                fe_values_face_int, fe_values_face_ext,
//                                                penalty,
//                                                current_dofs_indices, neighbor_dofs_indices,
//                                                current_cell_rhs, neighbor_cell_rhs);
//
//                } else {
//                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
//                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
//                    const unsigned int normal_direction = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
//                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction);
//
//                    real penalty = deg1sq / vol_div_facearea1;
//
//                    const unsigned int boundary_id = current_face->boundary_id();
//                    // Need to somehow get boundary type from the mesh
//                    assemble_boundary_term_explicit (boundary_id, fe_values_face_int, penalty, current_dofs_indices, current_cell_rhs);
//                }
//
//                //CASE 1.5: periodic boundary conditions
//                //note that periodicity is not adapted for hp adaptivity yet. this needs to be figured out in the future
//            } else if (current_face->at_boundary() && current_cell->has_periodic_neighbor(iface)){
//
//                neighbor_cell = current_cell->periodic_neighbor(iface);
//                //std::cout << "cell " << current_cell->index() << " at boundary" <<std::endl;
//                //std::cout << "periodic neighbour on face " << iface << " is " << neighbor_cell->index() << std::endl;
//
//
//                if (!current_cell->periodic_neighbor_is_coarser(iface) &&
//                    (neighbor_cell->index() > current_cell->index() ||
//                     (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
//                    )
//                   )
//                {
//                    Assert (current_cell->periodic_neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
//
//
//                    // Corresponding face of the neighbor.
//                    // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
//                    const unsigned int neighbor_face_no = current_cell->periodic_neighbor_of_periodic_neighbor(iface);
//
//                    // Get information about neighbor cell
//                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
//                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
//                    const unsigned int mapping_index_neigh_cell = 0;
//                    const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
//                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
//                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
//
//                    // Local rhs contribution from neighbor
//                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization
//
//                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
//                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
//                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);
//
//                    fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
//                    const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
//                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
//                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
//                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
//                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
//                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);
//
//                    //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
//                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
//                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);
//
//                    const real penalty1 = deg1sq / vol_div_facearea1;
//                    const real penalty2 = deg2sq / vol_div_facearea2;
//
//                    real penalty = 0.5 * ( penalty1 + penalty2 );
//                    //penalty = 1;//99;
//
//                    assemble_face_term_explicit (
//                            fe_values_face_int, fe_values_face_ext,
//                            penalty,
//                            current_dofs_indices, neighbor_dofs_indices,
//                            current_cell_rhs, neighbor_cell_rhs);
//
//                    // Add local contribution from neighbor cell to global vector
//                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
//                        right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
//                    }
//                }
//                else
//                {
//                    //do nothing
//                }
//
//
//            // Case 2:
//            // Neighbour is finer occurs if the face has children
//            // In this case, we loop over the current large face's subfaces and visit multiple neighbors
//            } else if (current_face->has_children()) {
//
//                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
//
//                // Obtain cell neighbour
//                const unsigned int neighbor_face_no = current_cell->neighbor_face_no(iface);
//
//                for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {
//
//                    // Get neighbor on ith subface
//                    auto neighbor_cell = current_cell->neighbor_child_on_subface (iface, subface_no);
//                    // Since the neighbor cell is finer than the current cell, it should not have more children
//                    Assert (!neighbor_cell->has_children(), dealii::ExcInternalError());
//
//                    // Get information about neighbor cell
//                    const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
//                    const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
//                    const unsigned int mapping_index_neigh_cell = 0;
//                    const dealii::FESystem<dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
//                    const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
//                    const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
//
//                    dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization
//
//                    // Obtain the mapping from local dof indices to global dof indices for neighbor cell
//                    neighbor_dofs_indices.resize(n_dofs_neigh_cell);
//                    neighbor_cell->get_dof_indices (neighbor_dofs_indices);
//
//                    fe_values_collection_subface.reinit (current_cell, iface, subface_no, quad_index, mapping_index, fe_index_curr_cell);
//                    const dealii::FESubfaceValues<dim,dim> &fe_values_face_int = fe_values_collection_subface.get_present_fe_values();
//
//                    fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
//                    const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                    const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
//                    const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
//                    const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
//                    const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);
//
//                    const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
//                    const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);
//
//                    const real penalty1 = deg1sq / vol_div_facearea1;
//                    const real penalty2 = deg2sq / vol_div_facearea2;
//                    
//                    real penalty = 0.5 * ( penalty1 + penalty2 );
//
//                    assemble_face_term_explicit (
//                        fe_values_face_int, fe_values_face_ext,
//                        penalty,
//                        current_dofs_indices, neighbor_dofs_indices,
//                        current_cell_rhs, neighbor_cell_rhs);
//                    // Add local contribution from neighbor cell to global vector
//                    for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
//                        right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
//                    }
//                }
//
//            // Case 3:
//            // Neighbor cell is NOT coarser
//            // Therefore, they have the same coarseness, and we need to choose one of them to do the work
//            } else if (
//                (   !(current_cell->neighbor_is_coarser(iface))
//                    // In the case the neighbor is a ghost cell, we let the processor with the lower rank do the work on that face
//                    // We cannot use the cell->index() because the index is relative to the distributed triangulation
//                    // Therefore, the cell index of a ghost cell might be different to the physical cell index even if they refer to the same cell
//                 && neighbor_cell->is_ghost()
//                 && current_cell->subdomain_id() < neighbor_cell->subdomain_id()
//                )
//                ||
//                (   !(current_cell->neighbor_is_coarser(iface))
//                    // In the case the neighbor is a local cell, we let the cell with the lower index do the work on that face
//                 && neighbor_cell->is_locally_owned()
//                 &&
//                    (  // Cell with lower index does work
//                       current_cell->index() < neighbor_cell->index()
//                     ||
//                       // If both cells have same index
//                       // See https://www.dealii.org/developer/doxygen/deal.II/classTriaAccessorBase.html#a695efcbe84fefef3e4c93ee7bdb446ad
//                       // then cell at the lower level does the work
//                       (neighbor_cell->index() == current_cell->index() && current_cell->level() < neighbor_cell->level())
//                    )
//                )
//            )
//            {
//                Assert (current_cell->neighbor(iface).state() == dealii::IteratorState::valid, dealii::ExcInternalError());
//
//                auto neighbor_cell = current_cell->neighbor_or_periodic_neighbor(iface);
//                // Corresponding face of the neighbor.
//                // e.g. The 4th face of the current cell might correspond to the 3rd face of the neighbor
//                const unsigned int neighbor_face_no = current_cell->neighbor_of_neighbor(iface);
//
//                // Get information about neighbor cell
//                const unsigned int fe_index_neigh_cell = neighbor_cell->active_fe_index();
//                const unsigned int quad_index_neigh_cell = fe_index_neigh_cell;
//                const unsigned int mapping_index_neigh_cell = 0;
//                const dealii::FESystem<dim,dim> &neigh_fe_ref = fe_collection[fe_index_neigh_cell];
//                const unsigned int neigh_cell_degree = neigh_fe_ref.tensor_degree();
//                const unsigned int n_dofs_neigh_cell = neigh_fe_ref.n_dofs_per_cell();
//
//                // Local rhs contribution from neighbor
//                dealii::Vector<double> neighbor_cell_rhs (n_dofs_neigh_cell); // Defaults to 0.0 initialization
//
//                // Obtain the mapping from local dof indices to global dof indices for neighbor cell
//                neighbor_dofs_indices.resize(n_dofs_neigh_cell);
//                neighbor_cell->get_dof_indices (neighbor_dofs_indices);
//
//                fe_values_collection_face_int.reinit (current_cell, iface, quad_index, mapping_index, fe_index_curr_cell);
//                const dealii::FEFaceValues<dim,dim> &fe_values_face_int = fe_values_collection_face_int.get_present_fe_values();
//                fe_values_collection_face_ext.reinit (neighbor_cell, neighbor_face_no, quad_index_neigh_cell, mapping_index_neigh_cell, fe_index_neigh_cell);
//                const dealii::FEFaceValues<dim,dim> &fe_values_face_ext = fe_values_collection_face_ext.get_present_fe_values();
//
//                const unsigned int normal_direction1 = dealii::GeometryInfo<dim>::unit_normal_direction[iface];
//                const unsigned int normal_direction2 = dealii::GeometryInfo<dim>::unit_normal_direction[neighbor_face_no];
//                const unsigned int deg1sq = (curr_cell_degree == 0) ? 1 : curr_cell_degree * (curr_cell_degree+1);
//                const unsigned int deg2sq = (neigh_cell_degree == 0) ? 1 : neigh_cell_degree * (neigh_cell_degree+1);
//
//                //const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1) / current_face->number_of_children();
//                const real vol_div_facearea1 = current_cell->extent_in_direction(normal_direction1);
//                const real vol_div_facearea2 = neighbor_cell->extent_in_direction(normal_direction2);
//
//                const real penalty1 = deg1sq / vol_div_facearea1;
//                const real penalty2 = deg2sq / vol_div_facearea2;
//                
//                real penalty = 0.5 * ( penalty1 + penalty2 );
//                //penalty = 1;//99;
//
//                assemble_face_term_explicit (
//                        fe_values_face_int, fe_values_face_ext,
//                        penalty,
//                        current_dofs_indices, neighbor_dofs_indices,
//                        current_cell_rhs, neighbor_cell_rhs);
//
//                // Add local contribution from neighbor cell to global vector
//                for (unsigned int i=0; i<n_dofs_neigh_cell; ++i) {
//                    right_hand_side(neighbor_dofs_indices[i]) += neighbor_cell_rhs(i);
//                }
//            } else {
//                // Case 4: Neighbor is coarser
//                // Do nothing.
//                // The face contribution from the current cell will appear then the coarse neighbor checks for subfaces
//            }
//
//        } // end of face loop
//
//        for (unsigned int i=0; i<n_dofs_curr_cell; ++i) {
//            right_hand_side(current_dofs_indices[i]) += current_cell_rhs(i);
//        }
//
//    } // end of cell loop
//
//    return dRdX;
//
//}

} // namespace PHiLiP
