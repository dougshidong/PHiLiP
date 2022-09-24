#include "identity_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
DesignParameterizationIdentity<dim> :: DesignParameterizationIdentity(
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : DesignParameterizationBase<dim>(_high_order_grid)
    {}

template<int dim>
void DesignParameterizationIdentity<dim> :: initialize_design_variables(VectorType &design_var)
{
    design_var = this->high_order_grid->volume_nodes; // Copies both the values and parallel distribution layout.
    current_volume_nodes = design_var;
    design_var.update_ghost_values();
    current_volume_nodes.update_ghost_values();
}

template<int dim>
void DesignParameterizationIdentity<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    // This might not be the best way to create parallel partitioned Identity matrix. To be updated if found.
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_vol_nodes, volume_range);

    for(unsigned int i=0; i<n_vol_nodes; ++i)
    {
        if(!volume_range.is_element(i)) continue;
        dsp.add(i,i);
    }

    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, this->mpi_communicator, locally_relevant_dofs);

    dXv_dXp.reinit(volume_range, volume_range, dsp, this->mpi_communicator);

    for(unsigned int i=0; i<n_vol_nodes; ++i)
    {
        if(!volume_range.is_element(i)) continue;
        dXv_dXp.set(i,i,1.0);
    }

    dXv_dXp.compress(dealii::VectorOperation::insert);
}

template<int dim>
bool DesignParameterizationIdentity<dim> ::update_mesh_from_design_variables(
    const MatrixType &dXv_dXp,
    const VectorType &design_var)
{
    AssertDimension(dXv_dXp.n(), design_var.size());
    
    bool design_variable_has_changed = this->has_design_variable_been_updated(current_volume_nodes, design_var);
    bool mesh_updated;
    if(!(design_variable_has_changed))
    {
        mesh_updated = false;
        return mesh_updated;
    }

    current_volume_nodes = design_var;
    dXv_dXp.vmult(this->high_order_grid->volume_nodes, design_var);
    this->high_order_grid->volume_nodes.update_ghost_values();
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
unsigned int DesignParameterizationIdentity<dim> :: get_number_of_design_variables() const
{
    return this->high_order_grid->volume_nodes.size();
}

template class DesignParameterizationIdentity<PHILIP_DIM>;
} // namespace PHiLiP
