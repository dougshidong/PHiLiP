#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/dofs/dof_tools.h>

#include "dg.h"

namespace PHiLiP {

template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_d2RdWdX_sparsity_pattern ()
{
    return get_dRdX_sparsity_pattern ();
}
template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_dRdW_sparsity_pattern ()
{
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    return sparsity_pattern;
}

template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_d2RdWdW_sparsity_pattern ()
{
    return get_dRdW_sparsity_pattern();
}
template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_d2RdXdX_sparsity_pattern ()
{
    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(high_order_grid->dof_handler_grid, locally_relevant_dofs);
    dealii::DynamicSparsityPattern dsp(locally_relevant_dofs);
    dealii::DoFTools::make_flux_sparsity_pattern(high_order_grid->dof_handler_grid, dsp);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, high_order_grid->dof_handler_grid.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    return sparsity_pattern;
}

template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_dRdX_sparsity_pattern ()
{
    const unsigned n_residuals = dof_handler.n_dofs();
    const unsigned n_nodes_coeff = high_order_grid->dof_handler_grid.n_dofs();
    const unsigned int n_rows = n_residuals;
    const unsigned int n_cols = n_nodes_coeff;

    dealii::DynamicSparsityPattern dsp(n_rows, n_cols);

    const unsigned int n_node_cell = high_order_grid->fe_system.n_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> resi_indices;
    std::vector<dealii::types::global_dof_index> node_indices(n_node_cell);
    auto cell = dof_handler.begin_active();
    auto metric_cell = high_order_grid->dof_handler_grid.begin_active();
    for (; cell != dof_handler.end(); ++cell, ++metric_cell) {
        if (!cell->is_locally_owned()) continue;

        const unsigned int n_resi_cell = fe_collection[cell->active_fe_index()].n_dofs_per_cell();
        resi_indices.resize(n_resi_cell);
        cell->get_dof_indices (resi_indices);

        metric_cell->get_dof_indices (node_indices);
        for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
            dsp.add_entries(*resi_row, node_indices.begin(), node_indices.end());
        }
        for (unsigned int iface=0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface) {
            auto current_face = cell->face(iface);

            if (current_face->at_boundary()) {
            // Do nothing
            } else if (current_face->has_children()) {
            // Finer neighbor
            // Loop over them and add their DoF to dependencies
                for (unsigned int subface_no=0; subface_no < current_face->number_of_children(); ++subface_no) {
                    const auto neighbor_metric_cell = metric_cell->neighbor_child_on_subface (iface, subface_no);
                    neighbor_metric_cell->get_dof_indices (node_indices);
                    for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
                        dsp.add_entries(*resi_row, node_indices.begin(), node_indices.end());
                    }
                }
            } else if (cell->neighbor_is_coarser(iface)) {
            // Coarser neighbor
            // Add DoF of that neighbor.
                const auto neighbor_metric_cell = metric_cell->neighbor (iface);
                neighbor_metric_cell->get_dof_indices (node_indices);
                for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
                    dsp.add_entries(*resi_row, node_indices.begin(), node_indices.end());
                }
            } else {
            //if ( !(cell->neighbor_is_coarser(iface)) ) {A
            // Same level neighbor
            // Add DoF of that neighbor.
                if (dim == 1 && cell->neighbor(iface)->has_children()) {
                    const auto coarse_unactive_neighbor = metric_cell->neighbor (iface);
                    for (unsigned int i_child=0; i_child < coarse_unactive_neighbor->n_children(); ++i_child) {
                        const auto neighbor_metric_cell = coarse_unactive_neighbor->child (i_child);
                        for (unsigned int iface_child=0; iface_child < dealii::GeometryInfo<dim>::faces_per_cell; ++iface_child) {
                            if (neighbor_metric_cell->neighbor(iface_child) == metric_cell) {
                                neighbor_metric_cell->get_dof_indices (node_indices);
                                for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
                                    dsp.add_entries(*resi_row, node_indices.begin(), node_indices.end());
                                }
                            }
                        }
                    }
                } else {
                    const auto neighbor_metric_cell = metric_cell->neighbor (iface);
                    neighbor_metric_cell->get_dof_indices (node_indices);
                    for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
                        dsp.add_entries(*resi_row, node_indices.begin(), node_indices.end());
                    }
                }
            }
        } 
    } // end of cell loop

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), MPI_COMM_WORLD, locally_relevant_dofs);
    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);

    return sparsity_pattern;
}

template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_d2RdWdXs_sparsity_pattern ()
{
    return get_dRdXs_sparsity_pattern ();
}


template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_d2RdXsdXs_sparsity_pattern ()
{
    const auto &partitionner = high_order_grid->surface_to_volume_indices.get_partitioner();
    const dealii::IndexSet owned = partitionner->locally_owned_range();
    dealii::DynamicSparsityPattern dsp(owned);

    const unsigned n_cols = high_order_grid->surface_nodes.size();
    std::vector<dealii::types::global_dof_index> node_indices(n_cols);
    std::iota (std::begin(node_indices), std::end(node_indices), 0);
    for (auto row = owned.begin(); row != owned.end(); ++row) {
        dsp.add_entries(*row, node_indices.begin(), node_indices.end());
    }

    //dealii::SparsityTools::distribute_sparsity_pattern(dsp, high_order_grid->n_locally_owned_surface_nodes_per_mpi, mpi_communicator, locally_relevant_dofs);
    dealii::SparsityTools::distribute_sparsity_pattern(dsp, high_order_grid->n_locally_owned_surface_nodes_per_mpi, mpi_communicator, owned);

    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);
    return sparsity_pattern;
}

template <int dim, typename real, typename MeshType>
dealii::SparsityPattern DGBase<dim,real,MeshType>::get_dRdXs_sparsity_pattern ()
{
    const unsigned n_residuals = dof_handler.n_dofs();
    const unsigned n_nodes_coeff = high_order_grid->surface_nodes.size();
    const unsigned int n_rows = n_residuals;
    const unsigned int n_cols = n_nodes_coeff;

    dealii::DynamicSparsityPattern dsp(n_rows, n_cols);

    std::vector<dealii::types::global_dof_index> resi_indices;
    std::vector<dealii::types::global_dof_index> node_indices(n_cols);
    std::iota (std::begin(node_indices), std::end(node_indices), 0);
    auto cell = dof_handler.begin_active();
    for (; cell != dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;

        const unsigned int n_resi_cell = fe_collection[cell->active_fe_index()].n_dofs_per_cell();
        resi_indices.resize(n_resi_cell);
        cell->get_dof_indices (resi_indices);

        for (auto resi_row = resi_indices.begin(); resi_row!=resi_indices.end(); ++resi_row) {
            dsp.add_entries(*resi_row, node_indices.begin(), node_indices.end());
        }
    } // end of cell loop

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), MPI_COMM_WORLD, locally_owned_dofs);
    dealii::SparsityPattern sparsity_pattern;
    sparsity_pattern.copy_from(dsp);

    return sparsity_pattern;
}

template class DGBase <PHILIP_DIM, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGBase <PHILIP_DIM, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
#if PHILIP_DIM!=1
template class DGBase <PHILIP_DIM, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // namespace PHiLiP

