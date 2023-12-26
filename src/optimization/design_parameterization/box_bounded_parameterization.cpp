#include "box_bounded_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
BoxBoundedParameterization<dim> :: BoxBoundedParameterization(
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : BaseParameterization<dim>(_high_order_grid)
{
    compute_control_index_to_vol_index();
}

template<int dim>
void BoxBoundedParameterization<dim> :: compute_control_index_to_vol_index()
{
    const double x_low = -0.6;
    const double x_high = 1.2;
    const double y_low = -1.3;
    const double y_high = 1.4;
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    const unsigned int n_surf_nodes = this->high_order_grid->surface_nodes.size();

    dealii::LinearAlgebra::distributed::Vector<int> is_a_control_node;
    is_a_control_node.reinit(this->high_order_grid->volume_nodes); // Copies parallel layout, without values. Initializes to 0 by default.
    is_a_control_node = 0;
    is_a_control_node.update_ghost_values();

    // Get locally owned volume and surface ranges of indices held by current processor.
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &surface_range = this->high_order_grid->surface_nodes.get_partitioner()->locally_owned_range();

    for(unsigned int i_vol = 0; i_vol<n_vol_nodes; ++i_vol) 
    {
        if(!(volume_range.is_element(i_vol))) continue;
        
        if(i_vol % dim == 0.0)
        {
            if(!(volume_range.is_element(i_vol+1)))
            {
                std::cout<<"ivol+1 does not belong to the same processor, as initially expected. Aborting.."<<std::endl<<std::flush;
                std::abort();
            }
        }
        
        if(i_vol % dim == 0.0)
        {
            const double x = this->high_order_grid->volume_nodes(i_vol);
            const double y = this->high_order_grid->volume_nodes(i_vol+1);

            if( (x > x_low) && (x < x_high) && (y > y_low) && (y < y_high) )
            {
                is_a_control_node(i_vol) = 1;
                is_a_control_node(i_vol+1) = 1;
            }
        }
    }
    is_a_control_node.update_ghost_values();

    for(unsigned int i_surf = 0; i_surf < n_surf_nodes; ++i_surf)
    {
        if(!(surface_range.is_element(i_surf))) continue;
        const unsigned int vol_index = this->high_order_grid->surface_to_volume_indices(i_surf);
        is_a_control_node(vol_index) = 0;
    }
    is_a_control_node.update_ghost_values();
    n_control_nodes = is_a_control_node.l1_norm();

    unsigned int n_control_nodes_this_processor = 0;
    for(unsigned int i_vol = 0; i_vol<n_vol_nodes; ++i_vol) 
    {
        if(!(volume_range.is_element(i_vol))) continue;
        if(is_a_control_node(i_vol) == 1) {++n_control_nodes_this_processor;}
    }


    //=========== Set inner_vol_range IndexSet of current processor ================================================================

    unsigned int n_elements_this_mpi = n_control_nodes_this_processor; // Size of local indexset
    std::vector<unsigned int> n_elements_per_mpi(this->n_mpi);
    MPI_Allgather(&n_elements_this_mpi, 1, MPI_UNSIGNED, &(n_elements_per_mpi[0]), 1, MPI_UNSIGNED, this->mpi_communicator);
    
    // Set lower index and hgher index of locally owned IndexSet on each processor
    unsigned int lower_index = 0, higher_index = 0;
    for(int i_mpi = 0; i_mpi < this->mpi_rank; ++i_mpi)
    {
        lower_index += n_elements_per_mpi[i_mpi];
    }
    higher_index = lower_index + n_elements_this_mpi;

    control_index_range.set_size(n_control_nodes);
    control_index_range.add_range(lower_index, higher_index);
    
    //=========== Set control_index_to_vol_index ================================================================
    control_index_to_vol_index.reinit(control_index_range, this->mpi_communicator); // No need of ghosts. To be verified later.

    unsigned int count1 = lower_index;
    for(unsigned int i_vol = 0; i_vol < n_vol_nodes; ++i_vol)
    {
        if(!volume_range.is_element(i_vol)) continue;
        
        if(is_a_control_node(i_vol) == 1)
        {
            control_index_to_vol_index[count1++] = i_vol;
        }
    }
    AssertDimension(count1, higher_index);
}

template<int dim>
void BoxBoundedParameterization<dim> :: initialize_design_variables(VectorType &design_var)
{
    control_ghost_range.set_size(n_control_nodes);
    control_ghost_range.add_range(0, n_control_nodes);
    design_var.reinit(control_index_range, control_ghost_range, this->mpi_communicator);

    for(unsigned int i_control=0; i_control<n_control_nodes; ++i_control)
    {
        if(control_index_range.is_element(i_control))
        {
            const unsigned int vol_index = control_index_to_vol_index[i_control];
            design_var[i_control] = this->high_order_grid->volume_nodes[vol_index];
        }
    }
    design_var.update_ghost_values();
    current_design_var = design_var;
    current_design_var.update_ghost_values();
}

template<int dim>
void BoxBoundedParameterization<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_control_nodes, volume_range);
    for(unsigned int i_control=0; i_control<n_control_nodes; ++i_control)
    {
        if(control_index_range.is_element(i_control))
        {
            dsp.add(control_index_to_vol_index[i_control],i_control);
        }
    }


    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, this->mpi_communicator, locally_relevant_dofs);

    dXv_dXp.reinit(volume_range, control_index_range, dsp, this->mpi_communicator);

    for(unsigned int i_control=0; i_control<n_control_nodes; ++i_control)
    {
        if(control_index_range.is_element(i_control))
        {
            dXv_dXp.set(control_index_to_vol_index[i_control], i_control, 1.0);
        }
    }

    dXv_dXp.compress(dealii::VectorOperation::insert);
}

template<int dim>
bool BoxBoundedParameterization<dim> ::update_mesh_from_design_variables(
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
    change_in_des_var.update_ghost_values();

    current_design_var = design_var;
    current_design_var.update_ghost_values();
    dXv_dXp.vmult_add(this->high_order_grid->volume_nodes, change_in_des_var); // Xv = Xv + dXv_dXp*(Xp,new - Xp); Gives Xv for surface nodes and Xp,new for inner vol nodes. 
    this->high_order_grid->volume_nodes.update_ghost_values();
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
unsigned int BoxBoundedParameterization<dim> :: get_number_of_design_variables() const
{
    return n_control_nodes;
}

template<int dim>
int BoxBoundedParameterization<dim> :: is_design_variable_valid(
    const MatrixType &dXv_dXp, 
    const VectorType &design_var) const
{
    this->pcout<<"Checking if mesh is valid before updating variables..."<<std::endl;
    VectorType vol_nodes_from_design_var = this->high_order_grid->volume_nodes;
    VectorType change_in_des_var = design_var;
    change_in_des_var -= current_design_var;
    change_in_des_var.update_ghost_values();

    dXv_dXp.vmult_add(vol_nodes_from_design_var, change_in_des_var); // Xv = Xv + dXv_dXp*(Xp,new - Xp); Gives Xv for surface nodes and Xp,new for inner vol nodes. 
    vol_nodes_from_design_var.update_ghost_values();
    
    int mesh_error_this_processor = 0;
    const dealii::FESystem<dim,dim> &fe_metric = this->high_order_grid->get_current_fe_system();
    const unsigned int n_dofs_per_cell = fe_metric.n_dofs_per_cell();
    const std::vector< dealii::Point<dim> > &ref_points = fe_metric.get_unit_support_points();
    for (const auto &cell : this->high_order_grid->dof_handler_grid.active_cell_iterators()) 
    {
        if (! cell->is_locally_owned()) {continue;}

        const std::vector<double> jac_det = this->high_order_grid->evaluate_jacobian_at_points(vol_nodes_from_design_var, cell, ref_points);
        for (unsigned int i=0; i<n_dofs_per_cell; ++i) 
        {
            if(jac_det[i] < 1.0e-12)
            {
                std::cout<<"Cell is distorted"<<std::endl;
                ++mesh_error_this_processor;
                break;
            }
        }

        if(mesh_error_this_processor > 0) {break;}
    }

    const int mesh_error_mpi = dealii::Utilities::MPI::sum(mesh_error_this_processor, this->mpi_communicator);
    return mesh_error_mpi;
}

template class BoxBoundedParameterization<PHILIP_DIM>;
} // namespace PHiLiP
