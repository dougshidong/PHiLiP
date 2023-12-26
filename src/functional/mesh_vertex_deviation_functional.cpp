#include "mesh_vertex_deviation_functional.h"

namespace PHiLiP {

template<int dim, int nstate, typename real>
MeshVertexDeviation<dim, nstate, real> :: MeshVertexDeviation(
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient)
    : Functional<dim,nstate,real>(dg_input,uses_solution_values, uses_solution_gradient)
    , initial_vol_nodes(this->dg->high_order_grid->volume_nodes)
    , mesh_weight(this->dg->all_parameters->optimization_param.mesh_weight_factor)
{
    initial_vol_nodes.update_ghost_values();
}

template<int dim, int nstate, typename real>
real MeshVertexDeviation<dim,nstate,real> :: evaluate_functional(
    const bool compute_dIdw, 
    const bool compute_dIdX, 
    const bool compute_d2I) 
{
    bool actually_compute_value = true;
    bool actually_compute_dIdw = compute_dIdw;
    bool actually_compute_dIdX = compute_dIdX;
    bool actually_compute_d2I  = compute_d2I;

    this->need_compute(actually_compute_value, actually_compute_dIdw, actually_compute_dIdX, actually_compute_d2I);
    
    if (!actually_compute_value && !actually_compute_dIdw && !actually_compute_dIdX && !actually_compute_d2I) 
    {
        return this->current_functional_value;
    }

    this->allocate_derivatives(actually_compute_dIdw, actually_compute_dIdX, actually_compute_d2I);
    
    real local_functional = 0.0;
    const unsigned int poly_degree = this->dg->get_min_fe_degree();

    const dealii::FESystem<dim,dim> &fe_metric = this->dg->high_order_grid->get_current_fe_system();
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    std::vector<FadFadType> coords_coeff(n_metric_dofs_cell);
    std::vector<real>   target_vals(n_metric_dofs_cell);
    std::vector<real>   local_dIdX(n_metric_dofs_cell);

    for(const auto &metric_cell : this->dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if(!metric_cell->is_locally_owned()) continue;

        metric_cell->get_dof_indices (cell_metric_dofs_indices);

        for(unsigned int idof = 0; idof<n_metric_dofs_cell; ++idof)
        {
            coords_coeff[idof] = this->dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
            target_vals[idof] = initial_vol_nodes[cell_metric_dofs_indices[idof]];
        }

        // Setup AD variables
        unsigned int n_total_indep = 0;
        if (actually_compute_dIdX || actually_compute_d2I) n_total_indep += n_metric_dofs_cell;

        unsigned int i_derivative = 0;
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            const real val = this->dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
            coords_coeff[idof] = val;
            if(actually_compute_dIdX || actually_compute_d2I) coords_coeff[idof].diff(i_derivative++, n_total_indep);
        }

        if(actually_compute_d2I)
        {
            i_derivative = 0;
            for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) 
            {
                const real val = this->dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
                coords_coeff[idof].val() = val;
                coords_coeff[idof].val().diff(i_derivative++, n_total_indep);
            }
        }

        // Get quadrature point on reference cell
        const dealii::Quadrature<dim> &volume_quadrature = this->dg->volume_quadrature_collection[poly_degree];

        FadFadType volume_local_sum = evaluate_cell_volume_term(coords_coeff,target_vals,fe_metric,volume_quadrature);

        local_functional += volume_local_sum.val().val();
        
        // Differentiate using AD.
        i_derivative = 0;
        if (actually_compute_dIdX) {
            local_dIdX.resize(n_metric_dofs_cell);
            for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
            {
                local_dIdX[idof] = volume_local_sum.dx(i_derivative++).val();
            }
            this->dIdX.add(cell_metric_dofs_indices, local_dIdX);
        }

        if(actually_compute_d2I)
        {
            std::vector<real> dXidX(n_metric_dofs_cell);
            i_derivative = 0;
            for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) 
            {
                const FadType dXi = volume_local_sum.dx(i_derivative++);
                unsigned int j_derivative = 0;
                for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) 
                {
                    dXidX[jdof] = dXi.dx(j_derivative++);
                }
                this->d2IdXdX->add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices, dXidX);
            }
        }
    } // cell loop ends

    this->current_functional_value = dealii::Utilities::MPI::sum(local_functional, MPI_COMM_WORLD);
    if (actually_compute_dIdX) this->dIdX.compress(dealii::VectorOperation::add);
    if (actually_compute_d2I) this->d2IdXdX->compress(dealii::VectorOperation::add);
 
    return this->current_functional_value;
}

template<int dim, int nstate, typename real>
template<typename real2>
real2 MeshVertexDeviation<dim,nstate,real> :: evaluate_cell_volume_term(
    const std::vector<real2> &coords_coeff,
    const std::vector<real> &target_values,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    (void) volume_quadrature;
    (void) fe_metric;
    
    real2 cell_volume_term = 0.0;
    for(unsigned int idof = 0; idof < target_values.size(); ++idof)
    {
        real2 diff = coords_coeff[idof] - target_values[idof];
        cell_volume_term += mesh_weight*diff*diff;
    }
    return cell_volume_term;
}

template class MeshVertexDeviation<PHILIP_DIM, 1, double>;
template class MeshVertexDeviation<PHILIP_DIM, 2, double>;
template class MeshVertexDeviation<PHILIP_DIM, 3, double>;
template class MeshVertexDeviation<PHILIP_DIM, 4, double>;
template class MeshVertexDeviation<PHILIP_DIM, 5, double>;
} // PHiLiP namespace


