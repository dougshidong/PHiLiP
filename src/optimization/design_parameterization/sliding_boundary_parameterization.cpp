#include "sliding_boundary_parameterization.hpp"
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

namespace PHiLiP {

template<int dim>
SlidingBoundaryParameterization<dim> :: SlidingBoundaryParameterization(
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : BaseParameterization<dim>(_high_order_grid)
{
    compute_innersliding_vol_index_to_vol_index();
}

template<int dim>
void SlidingBoundaryParameterization<dim> :: compute_innersliding_vol_index_to_vol_index()
{
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    const unsigned int n_surf_nodes = this->high_order_grid->surface_nodes.size();
    const unsigned int n_corner_nodes = 2^dim * dim;
    const unsigned int n_surface_without_corner_nodes = n_surf_nodes - n_corner_nodes;
    n_innersliding_nodes = n_vol_nodes - n_surf_nodes + n_surface_without_corner_nodes * (dim-1)/dim;

    dealii::LinearAlgebra::distributed::Vector<int> is_a_fixed_node;
    is_a_fixed_node.reinit(this->high_order_grid->volume_nodes); // Copies parallel layout, without values. Initializes to 0 by default.

    // Get locally owned volume and surface ranges of indices held by current processor.
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const dealii::IndexSet &surface_range = this->high_order_grid->surface_nodes.get_partitioner()->locally_owned_range();
    int n_fixed_nodes_local = 0;
    // Set is_a_fixed_node. Makes it easier to iterate over innersliding nodes later.
    // Using surface_range.begin() and surface_range.end() might make it quicker to iterate over local ranges. To be checked later.
    for(unsigned int i_surf = 0; i_surf<n_surf_nodes; ++i_surf) 
    {
        if(!(surface_range.is_element(i_surf))) continue;

        const unsigned int vol_index = this->high_order_grid->surface_to_volume_indices(i_surf);
        Assert(volume_range.is_element(vol_index), 
                dealii::ExcMessage("Surface index is in range, so vol index is expected to be in the range of this processor."));
        if(vol_index % dim == 0.0)
        {
            const double x = this->high_order_grid->volume_nodes(vol_index);
            const double y = this->high_order_grid->volume_nodes(vol_index + 1);
            std::cout<<"On surface (x,y) = ("<<x<<", "<<y<<"). ";
            const double x_low = -1.0;
            const double x_high = 1.0;
            const double y_low = 0.0;
            const double y_high = 1.0;

            if( (x == x_low) || (x == x_high) )
            {
                // Constrain x
                is_a_fixed_node(vol_index) = 1;
                ++n_fixed_nodes_local;
                std::cout<<" x is constrained."<<std::endl;
            }
            if( (y == y_low) || (y == y_high) )
            {
                // Constrain y
                is_a_fixed_node(vol_index + 1) = 1;
                ++n_fixed_nodes_local;
                std::cout<<"y is constrained."<<std::endl;
            }

            if( (x==0.0) && (y==0.0))
            {
                // Constrain both
                is_a_fixed_node(vol_index) = 1;
                is_a_fixed_node(vol_index + 1) = 1;
                ++n_fixed_nodes_local;
                std::cout<<"both x and y are constrained."<<std::endl;
            }
        }
    }
    is_a_fixed_node.update_ghost_values();

    //=========== Set innersliding_vol_range IndexSet of current processor ================================================================

    unsigned int n_elements_this_mpi = volume_range.n_elements() - n_fixed_nodes_local; // Size of local indexset
    std::vector<unsigned int> n_elements_per_mpi(this->n_mpi);
    MPI_Allgather(&n_elements_this_mpi, 1, MPI_UNSIGNED, &(n_elements_per_mpi[0]), 1, MPI_UNSIGNED, this->mpi_communicator);
    
    // Set lower index and hgher index of locally owned IndexSet on each processor
    unsigned int lower_index = 0, higher_index = 0;
    for(int i_mpi = 0; i_mpi < this->mpi_rank; ++i_mpi)
    {
        lower_index += n_elements_per_mpi[i_mpi];
    }
    higher_index = lower_index + n_elements_this_mpi;

    innersliding_vol_range.set_size(n_innersliding_nodes);
    innersliding_vol_range.add_range(lower_index, higher_index);
    
    //=========== Set inner_vol_index_to_vol_index ================================================================
    innersliding_vol_index_to_vol_index.reinit(innersliding_vol_range, this->mpi_communicator); // No need of ghosts. To be verified later.

    unsigned int count1 = lower_index;
    for(unsigned int i_vol = 0; i_vol < n_vol_nodes; ++i_vol)
    {
        if(!volume_range.is_element(i_vol)) continue;
        
        if(is_a_fixed_node(i_vol)) continue;

        innersliding_vol_index_to_vol_index[count1++] = i_vol;
    }
    AssertDimension(count1, higher_index);
}

template<int dim>
void SlidingBoundaryParameterization<dim> :: initialize_design_variables(VectorType &design_var)
{
    design_var.reinit(innersliding_vol_range, this->mpi_communicator);

    for(unsigned int i=0; i<n_innersliding_nodes; ++i)
    {
        if(innersliding_vol_range.is_element(i))
        {
            const unsigned int vol_index = innersliding_vol_index_to_vol_index[i];
            design_var[i] = this->high_order_grid->volume_nodes[vol_index];
        }
    }
    current_design_var = design_var;
    design_var.update_ghost_values(); // Not required as there are no ghost values. Might be required later.
    current_design_var.update_ghost_values();
}

template<int dim>
void SlidingBoundaryParameterization<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    const dealii::IndexSet &volume_range = this->high_order_grid->volume_nodes.get_partitioner()->locally_owned_range();
    const unsigned int n_vol_nodes = this->high_order_grid->volume_nodes.size();
    
    dealii::DynamicSparsityPattern dsp(n_vol_nodes, n_innersliding_nodes, volume_range);
    for(unsigned int i=0; i<n_innersliding_nodes; ++i)
    {
        if(! innersliding_vol_range.is_element(i)) continue;

        dsp.add(innersliding_vol_index_to_vol_index[i],i);
    }


    dealii::IndexSet locally_relevant_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(this->high_order_grid->dof_handler_grid, locally_relevant_dofs);

    dealii::SparsityTools::distribute_sparsity_pattern(dsp, volume_range, this->mpi_communicator, locally_relevant_dofs);

    dXv_dXp.reinit(volume_range, innersliding_vol_range, dsp, this->mpi_communicator);

    for(unsigned int i=0; i<n_innersliding_nodes; ++i)
    {
        if(! innersliding_vol_range.is_element(i)) continue;

        dXv_dXp.set(innersliding_vol_index_to_vol_index[i], i, 1.0);
    }

    dXv_dXp.compress(dealii::VectorOperation::insert);
}

template<int dim>
bool SlidingBoundaryParameterization<dim> ::update_mesh_from_design_variables(
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
unsigned int SlidingBoundaryParameterization<dim> :: get_number_of_design_variables() const
{
    return n_innersliding_nodes;
}

template<int dim>
int SlidingBoundaryParameterization<dim> :: is_design_variable_valid(
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
    const dealii::FESystem<dim,dim> &fe_metric = this->high_order_grid->fe_system;
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

template class SlidingBoundaryParameterization<PHILIP_DIM>;
} // namespace PHiLiP
