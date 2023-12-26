#include "mesh_jacobian_deviation_functional.h"

namespace PHiLiP {

template<int dim, int nstate, typename real>
MeshJacobianDeviation<dim, nstate, real> :: MeshJacobianDeviation(
    std::shared_ptr<DGBase<dim,real>> dg_input,
    const bool uses_solution_values,
    const bool uses_solution_gradient)
    : Functional<dim,nstate,real>(dg_input,uses_solution_values, uses_solution_gradient)
    , mesh_weight(this->dg->all_parameters->optimization_param.mesh_weight_factor)
{
    if(this->dg->get_min_fe_degree() != this->dg->get_min_fe_degree())
    {
        std::cout<<"This class is currently coded for constant poly degree."<<std::endl<<std::flush;
        std::abort();
    }
    store_initial_rmsh();
}

template<int dim, int nstate, typename real>
void MeshJacobianDeviation<dim,nstate,real> :: store_initial_rmsh()
{
    initial_rmsh.reinit(this->dg->triangulation->n_active_cells());
    const dealii::FESystem<dim,dim> &fe_metric = this->dg->high_order_grid->get_current_fe_system();
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    auto metric_cell = this->dg->high_order_grid->dof_handler_grid.begin_active();
    auto soln_cell = this->dg->dof_handler.begin_active();
    for( ; soln_cell != this->dg->dof_handler.end(); ++soln_cell, ++metric_cell)
    {
        if(! soln_cell->is_locally_owned()) {continue;}
        const unsigned int cell_index = soln_cell->active_cell_index();
        const unsigned int i_fele = soln_cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        const dealii::Quadrature<dim> &volume_quadrature = this->dg->volume_quadrature_collection[i_quad];
        
        metric_cell->get_dof_indices (cell_metric_dofs_indices);

        std::vector<double> coords_coeff(n_metric_dofs_cell);
        for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof)
        {
            coords_coeff[idof] = this->dg->high_order_grid->volume_nodes(cell_metric_dofs_indices[idof]);
        }

        initial_rmsh(cell_index) = compute_cell_rmsh(coords_coeff, fe_metric, volume_quadrature);
    }
}


template<int dim, int nstate, typename real>
template<typename real2>
real2 MeshJacobianDeviation<dim,nstate,real> :: compute_cell_rmsh(
    const std::vector<real2> &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    real2 cell_rmsh = 0.0;

    const unsigned int n_vol_quad_pts = volume_quadrature.size();
    const unsigned int n_metric_dofs_cell = coords_coeff.size();

    for(unsigned int iquad = 0.0; iquad < n_vol_quad_pts; ++iquad)
    {
        const dealii::Point<dim,double> &ref_point = volume_quadrature.point(iquad);
        const double quad_weight = volume_quadrature.weight(iquad);

        std::array<dealii::Tensor<1, dim, real2>, dim> coord_grad;
        dealii::Tensor<2,dim,real2> metric_jacobian;

        for(unsigned int idof=0; idof<n_metric_dofs_cell; ++idof)
        {
            unsigned int idim = fe_metric.system_to_component_index(idof).first;
            coord_grad[idim] += coords_coeff[idof] * fe_metric.shape_grad(idof, ref_point);  
        }

        real2 jacobian_frobenius_norm_squared = 0.0;
        for (int row=0;row<dim;++row) 
        {
            for (int col=0;col<dim;++col) 
            {
                metric_jacobian[row][col] = coord_grad[row][col];
                jacobian_frobenius_norm_squared += pow(coord_grad[row][col], 2);
            }
        }
        real2 jacobian_determinant = dealii::determinant(metric_jacobian);

        real2 integrand_distortion_1 = jacobian_frobenius_norm_squared/pow(jacobian_determinant, 2/dim);
        real2 integrand_distortion = pow(integrand_distortion_1, 2);
        cell_rmsh += integrand_distortion * jacobian_determinant * quad_weight;
    }

    return cell_rmsh;
}


template<int dim, int nstate, typename real>
real MeshJacobianDeviation<dim,nstate,real> :: evaluate_functional(
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
    std::vector<real>   local_dIdX(n_metric_dofs_cell);

    for(const auto &metric_cell : this->dg->high_order_grid->dof_handler_grid.active_cell_iterators())
    {
        if(!metric_cell->is_locally_owned()) continue;

        const unsigned int cell_index = metric_cell->active_cell_index();
        metric_cell->get_dof_indices (cell_metric_dofs_indices);

        for(unsigned int idof = 0; idof<n_metric_dofs_cell; ++idof)
        {
            coords_coeff[idof] = this->dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
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
        const real initial_cell_rmsh = initial_rmsh(cell_index);
        FadFadType volume_local_sum = evaluate_cell_volume_term(coords_coeff,initial_cell_rmsh,fe_metric,volume_quadrature);

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
real2 MeshJacobianDeviation<dim,nstate,real> :: evaluate_cell_volume_term(
    const std::vector<real2> &coords_coeff,
    const real initial_cell_rmsh,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
   real2 cell_rmsh = compute_cell_rmsh(coords_coeff, fe_metric, volume_quadrature);

   real2 cell_volume_term = 0.5 * mesh_weight * pow( (cell_rmsh - initial_cell_rmsh) ,2);

   return cell_volume_term;
}

template class MeshJacobianDeviation<PHILIP_DIM, 1, double>;
template class MeshJacobianDeviation<PHILIP_DIM, 2, double>;
template class MeshJacobianDeviation<PHILIP_DIM, 3, double>;
template class MeshJacobianDeviation<PHILIP_DIM, 4, double>;
template class MeshJacobianDeviation<PHILIP_DIM, 5, double>;
} // PHiLiP namespace


