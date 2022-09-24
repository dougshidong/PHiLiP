#include "inner_vol_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
DesignParameterizationInnerVol<dim> :: DesignParameterizationInnerVol(
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : DesignParameterizationBase<dim>(_high_order_grid)
{
    compute_inner_vol_index_to_vol_index();
}

template<int dim>
void DesignParameterizationInnerVol<dim> :: compute_inner_vol_index_to_vol_index()
{
    unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    unsigned int n_surf_nodes = this->high_order_grid->surface_nodes.size();
    n_inner_nodes = n_vol_nodes - n_surf_nodes;

    dealii::LinearAlgebra::distributed::Vector<int> is_a_surface_node;
    is_a_surface_node.reinit(this->high_order_grid->volume_nodes); // Copies parallel layout, without values.
    is_a_surface_node *= 0; 

    // Get locally owned volume and surface ranges of indices held by current processor.
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &surface_range = this->high_order_grid->surface_nodes.get_partitioner()->locally_owned_range();

    // Set is_a_surface_node. Makes it easier to iterate over inner nodes later.
    // Using surface_range.begin() and surface_range.end() might make it quicker to iterate over local ranges. To be checked later.
    for(unsigned int i_surf = 0; i_surf<n_surf_nodes; ++i_surf) 
    {
        if(!(surface_range.is_element(i_surf))) continue;

        const unsigned int vol_index = this->high_order_grid->surface_to_volume_indices(i_surf);
        Assert(volume_range.is_element(vol_index), 
                dealii::ExcMessage("Surface index is in range, so vol index is expected to be in the range of this processor."));
        is_a_surface_node(vol_index) = 1;
    }
    is_a_surface_node.update_ghost_values();

    //=========== Set inner_vol_range IndexSet of current processor ================================================================

    unsigned int n_elements_this_mpi = volume_range.n_elements() - surface_range.n_elements(); // Size of local indexset
    std::vector<unsigned int> n_elements_per_mpi(this->n_mpi);
    MPI_Allgather(&n_elements_this_mpi, 1, MPI_UNSIGNED, &(n_elements_per_mpi[0]), 1, MPI_UNSIGNED, this->mpi_communicator);
    
    // Set lower index and hgher index of locally owned IndexSet on each processor
    unsigned int lower_index = 0, higher_index = 0;
    for(int i_mpi = 0; i_mpi < this->mpi_rank; ++i_mpi)
    {
        lower_index += n_elements_per_mpi[i_mpi];
    }
    higher_index = lower_index + n_elements_this_mpi;

    inner_vol_range.set_size(n_inner_nodes);
    inner_vol_range.add_range(lower_index, higher_index);
    
    //=========== Set inner_vol_index_to_vol_index ================================================================
    inner_vol_index_to_vol_index.reinit(inner_vol_range, this->mpi_communicator); // No need of ghosts. To be verified later.

    unsigned int count1 = lower_index;
    for(unsigned int i_vol = 0; i_vol < n_vol_nodes; ++i_vol)
    {
        if(!volume_range.is_element(i_vol)) continue;
        
        if(is_a_surface_node(i_vol)) continue;

        inner_vol_index_to_vol_index[count1++] = i_vol;
    }
    AssertDimension(count1, higher_index);
}

template<int dim>
void DesignParameterizationInnerVol<dim> :: initialize_design_variables(VectorType &design_var)
{
    design_var.reinit(inner_vol_range, this->mpi_communicator);

    for(unsigned int i=0; i<n_inner_nodes; ++i)
    {
        if(inner_vol_range.is_element(i))
        {
            const unsigned int vol_index = inner_vol_index_to_vol_index[i];
            design_var[i] = this->high_order_grid->volume_nodes[vol_index];
        }
    }
    current_design_var = design_var;
    design_var.update_ghost_values(); // Not required as there are no ghost values. Might be required later.
    current_design_var.update_ghost_values();
}

template<int dim>
void DesignParameterizationInnerVol<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_inner_nodes, volume_range);
    for(unsigned int i=0; i<n_inner_nodes; ++i)
    {
        if(! inner_vol_range.is_element(i)) continue;

        dsp.add(inner_vol_index_to_vol_index[i],i);
    }


    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, this->mpi_communicator, locally_relevant_dofs);

    dXv_dXp.reinit(volume_range, inner_vol_range, dsp, this->mpi_communicator);

    for(unsigned int i=0; i<n_inner_nodes; ++i)
    {
        if(! inner_vol_range.is_element(i)) continue;

        dXv_dXp.set(inner_vol_index_to_vol_index[i], i, 1.0);
    }

    dXv_dXp.compress(dealii::VectorOperation::insert);
}

template<int dim>
bool DesignParameterizationInnerVol<dim> ::update_mesh_from_design_variables(
    const MatrixType &dXv_dXp,
    const VectorType &design_var)
{
    AssertDimension(dXv_dXp.n(), design_var.size());
    
    // check if design variables have changed.
    bool design_variable_has_changed = this->has_design_variable_been_updated(current_design_var, design_var);
    bool mesh_updated;
    if(!(design_variable_has_changed))
    {
        mesh_updated = false;
        return mesh_updated;
    }
    VectorType change_in_des_var = design_var;
    change_in_des_var -= current_design_var;

    current_design_var = design_var;
    dXv_dXp.vmult_add(this->high_order_grid->volume_nodes, change_in_des_var); // Xv = Xv + dXv_dXp*(Xp,new - Xp); Gives Xv for surface nodes and Xp,new for inner vol nodes. 
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
unsigned int DesignParameterizationInnerVol<dim> :: get_number_of_design_variables() const
{
    return n_inner_nodes;
}

template class DesignParameterizationInnerVol<PHILIP_DIM>;
} // namespace PHiLiP
