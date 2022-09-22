#include "design_parameterization_ffd.hpp"

namespace PHiLiP {

template<int dim>
DesignParameterizationFreeFormDeformation<dim> :: DesignParameterizationFreeFormDeformation(
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid,
    const FreeFormDeformation<dim> &_ffd,
    std::vector< std::pair< unsigned int, unsigned int > > &_ffd_design_variables_indices_dim)
    : DesignParameterizationBase<dim>(_high_order_grid)
    , ffd(_ffd)
    , ffd_design_variables_indices_dim(_ffd_design_variables_indices_dim)
    {}

template<int dim>
void DesignParameterizationFreeFormDeformation<dim> :: initialize_design_variables(
    VectorType &ffd_des_var) 
{
    const unsigned int n_design_variables = ffd_design_variables_indices_dim.size();
    const dealii::IndexSet row_part = dealii::Utilities::MPI::create_evenly_distributed_partitioning(MPI_COMM_WORLD,n_design_variables);
    dealii::IndexSet ghost_row_part(n_design_variables);
    ghost_row_part.add_range(0,n_design_variables);
    ffd_des_var.reinit(row_part, ghost_row_part, MPI_COMM_WORLD);

    ffd.get_design_variables(ffd_design_variables_indices_dim, ffd_des_var);
    
    initial_ffd_des_var = ffd_des_var;
    initial_ffd_des_var.update_ghost_values();
}

template<int dim>
void DesignParameterizationFreeFormDeformation<dim> :: compute_dXv_dXp(MatrixType &dXv_dXp) const
{
    ffd.get_dXvdXp(*(this->high_order_grid), ffd_design_variables_indices_dim, dXv_dXp);
}

template<int dim>
bool DesignParameterizationFreeFormDeformation<dim> :: update_mesh_from_design_variables(
    const MatrixType &dXv_dXp,
    const VectorType &ffd_des_var)
{
    AssertDimension(ffd_des_var.size(), initial_ffd_des_var.size());
    VectorType current_ffd_des_var = ffd_des_var;
    ffd.get_design_variables( ffd_design_variables_indices_dim, current_ffd_des_var);

    VectorType diff = ffd_des_var;
    diff -= current_ffd_des_var;
    const double l2_norm = diff.l2_norm();
    bool mesh_updated;
    if(l2_norm == 0.0)
    {
        mesh_updated = false;
        return mesh_updated;
    }
    // Above if statement not executed -> design variables have changed
    ffd.set_design_variables( ffd_design_variables_indices_dim, ffd_des_var);

    VectorType dXp = ffd_des_var;
    dXp -= initial_ffd_des_var;
    dXp.update_ghost_values();
    VectorType dXv = this->high_order_grid->volume_nodes;
    dXv_dXp.vmult(dXv, dXp);
    dXv.update_ghost_values();
    this->high_order_grid->volume_nodes = this->high_order_grid->initial_volume_nodes;
    this->high_order_grid->volume_nodes += dXv;
    this->high_order_grid->volume_nodes.update_ghost_values();
    mesh_updated = true;
    return mesh_updated;
}

template<int dim>
void DesignParameterizationFreeFormDeformation<dim> :: output_design_variables(const unsigned int iteration_no) const
{
    ffd.output_ffd_vtu(iteration_no);
}

template<int dim>
unsigned int DesignParameterizationFreeFormDeformation<dim> :: get_number_of_design_variables() const
{
    return ffd_design_variables_indices_dim.size();
}

template class DesignParameterizationFreeFormDeformation<PHILIP_DIM>;
} // PHiLiP namespace
