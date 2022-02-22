// includes
#include <vector>
#include <algorithm>

#include <Sacado.hpp>

#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

#include "physics/physics.h"
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "functional.h"

/// Returns y = Ax.
/** Had to rewrite this instead of 
 *  dealii::contract<1,0>(A,x);
 *  because contract doesn't allow the use of codi variables.
 */
template<int dim, typename real1, typename real2>
dealii::Tensor<1,dim,real1> vmult(const dealii::Tensor<2,dim,real1> A, const dealii::Tensor<1,dim,real2> x)
{
     dealii::Tensor<1,dim,real1> y;
     for (int row=0;row<dim;++row) {
         y[row] = 0.0;
         for (int col=0;col<dim;++col) {
             y[row] += A[row][col] * x[col];
         }
     }
     return y;
}

/// Returns norm of dealii::Tensor<1,dim,real>
/** Had to rewrite this instead of 
 *  x.norm()
 *  because norm() doesn't allow the use of codi variables.
 */
template<int dim, typename real1>
real1 norm(const dealii::Tensor<1,dim,real1> x)
{
     real1 val = 0.0;
     for (int row=0;row<dim;++row) {
         val += x[row] * x[row];
     }
     return sqrt(val);
}

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
FunctionalNormLpVolume<dim,nstate,real,MeshType>::FunctionalNormLpVolume(
    const double                               _normLp,
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
    const bool                                 _uses_solution_values,
    const bool                                 _uses_solution_gradient) : 
        Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient),
        normLp(_normLp) {}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 FunctionalNormLpVolume<dim,nstate,real,MeshType>::evaluate_volume_integrand(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/*physics*/,
    const dealii::Point<dim,real2> &                      /*phys_coord*/,
    const std::array<real2,nstate> &                      soln_at_q,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
{
    real2 lpnorm_value = 0;
    for(unsigned int istate = 0; istate < nstate; ++istate)
        lpnorm_value += pow(abs(soln_at_q[istate]), this->normLp);
    return lpnorm_value;
}

template <int dim, int nstate, typename real, typename MeshType>
FunctionalNormLpBoundary<dim,nstate,real,MeshType>::FunctionalNormLpBoundary(
    const double                               _normLp,
    std::vector<unsigned int>                  _boundary_vector,
    const bool                                 _use_all_boundaries,
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
    const bool                                 _uses_solution_values,
    const bool                                 _uses_solution_gradient) : 
        Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient),
        normLp(_normLp),
        boundary_vector(_boundary_vector),
        use_all_boundaries(_use_all_boundaries) {}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 FunctionalNormLpBoundary<dim,nstate,real,MeshType>::evaluate_boundary_integrand(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/* physics */,
    const unsigned int                                    boundary_id,
    const dealii::Point<dim,real2> &                      /* phys_coord */,
    const dealii::Tensor<1,dim,real2> &                   /* normal */,
    const std::array<real2,nstate> &                      soln_at_q,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &/* soln_grad_at_q */) const
{
    real2 lpnorm_value = 0;

    // condition for whether the current cell should be evaluated
    auto boundary_vector_index = std::find(this->boundary_vector.begin(), this->boundary_vector.end(), boundary_id);
    bool eval_boundary = this->use_all_boundaries || boundary_vector_index != this->boundary_vector.end();

    if(!eval_boundary)
        return lpnorm_value;

    for(unsigned int istate = 0; istate < nstate; ++istate)
        lpnorm_value += pow(abs(soln_at_q[istate]), this->normLp);

    return lpnorm_value;
}

template <int dim, int nstate, typename real, typename MeshType>
FunctionalWeightedIntegralVolume<dim,nstate,real,MeshType>::FunctionalWeightedIntegralVolume(
    std::shared_ptr<ManufacturedSolutionFunction<dim,real>>                    _weight_function_double,
    std::shared_ptr<ManufacturedSolutionFunction<dim,FadFadType>>              _weight_function_adtype,
    const bool                                                                 _use_weight_function_laplacian,
    std::shared_ptr<DGBase<dim,real,MeshType>>                                 _dg,
    const bool                                                                 _uses_solution_values,
    const bool                                                                 _uses_solution_gradient) : 
        Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient),
        weight_function_double(_weight_function_double),
        weight_function_adtype(_weight_function_adtype),
        use_weight_function_laplacian(_use_weight_function_laplacian) {}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 FunctionalWeightedIntegralVolume<dim,nstate,real,MeshType>::evaluate_volume_integrand(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &  /*physics*/,
    const dealii::Point<dim,real2> &                        phys_coord,
    const std::array<real2,nstate> &                        soln_at_q,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &  /*soln_grad_at_q*/,
    std::shared_ptr<ManufacturedSolutionFunction<dim,real2>> weight_function) const
{
    real2 val = 0;

    if(this->use_weight_function_laplacian){
        for(unsigned int istate = 0; istate < nstate; ++istate)
            val += soln_at_q[istate] * dealii::trace(weight_function->hessian(phys_coord, istate));
    }else{
        for(unsigned int istate = 0; istate < nstate; ++istate)
            val += soln_at_q[istate] * weight_function->value(phys_coord, istate);
    }

    return val;
}

template <int dim, int nstate, typename real, typename MeshType>
FunctionalWeightedIntegralBoundary<dim,nstate,real,MeshType>::FunctionalWeightedIntegralBoundary(
    std::shared_ptr<ManufacturedSolutionFunction<dim,real>>                    _weight_function_double,
    std::shared_ptr<ManufacturedSolutionFunction<dim,FadFadType>>              _weight_function_adtype,
    const bool                                                                 _use_weight_function_laplacian,
    std::vector<unsigned int>                                                  _boundary_vector,
    const bool                                                                 _use_all_boundaries,
    std::shared_ptr<DGBase<dim,real,MeshType>>                                 _dg,
    const bool                                                                 _uses_solution_values,
    const bool                                                                 _uses_solution_gradient) : 
        Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient),
        weight_function_double(_weight_function_double),
        weight_function_adtype(_weight_function_adtype),
        use_weight_function_laplacian(_use_weight_function_laplacian),
        boundary_vector(_boundary_vector),
        use_all_boundaries(_use_all_boundaries) {}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 FunctionalWeightedIntegralBoundary<dim,nstate,real,MeshType>::evaluate_boundary_integrand(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &/* physics */,
    const unsigned int                                    boundary_id,
    const dealii::Point<dim,real2> &                      phys_coord,
    const dealii::Tensor<1,dim,real2> &                   /* normal */,
    const std::array<real2,nstate> &                      soln_at_q,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &/* soln_grad_at_q */,
    std::shared_ptr<ManufacturedSolutionFunction<dim,real2>> weight_function) const
{
    real2 val = 0;

    // condition for whether the current cell should be evaluated
    auto boundary_vector_index = std::find(this->boundary_vector.begin(), this->boundary_vector.end(), boundary_id);
    bool eval_boundary = this->use_all_boundaries || boundary_vector_index != this->boundary_vector.end();

    if(!eval_boundary)
        return val;

    if(this->use_weight_function_laplacian){
        for(unsigned int istate = 0; istate < nstate; ++istate)
            val += soln_at_q[istate] * dealii::trace(weight_function->hessian(phys_coord, istate));
    }else{
        for(unsigned int istate = 0; istate < nstate; ++istate)
            val += soln_at_q[istate] * weight_function->value(phys_coord, istate);
    }

    return val;
}

template <int dim, int nstate, typename real, typename MeshType>
FunctionalErrorNormLpVolume<dim,nstate,real,MeshType>::FunctionalErrorNormLpVolume(
    const double                               _normLp,
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
    const bool                                 _uses_solution_values,
    const bool                                 _uses_solution_gradient) : 
        Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient),
        normLp(_normLp) {}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 FunctionalErrorNormLpVolume<dim,nstate,real,MeshType>::evaluate_volume_integrand(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
    const dealii::Point<dim,real2> &                      phys_coord,
    const std::array<real2,nstate> &                      soln_at_q,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &/*soln_grad_at_q*/) const
{
    real2 lpnorm_value = 0;
    for(unsigned int istate = 0; istate < nstate; ++istate){
        const real2 uexact = physics.manufactured_solution_function->value(phys_coord, istate);
        lpnorm_value += pow(abs(soln_at_q[istate] - uexact), this->normLp);
    }
    return lpnorm_value;
}

template <int dim, int nstate, typename real, typename MeshType>
FunctionalErrorNormLpBoundary<dim,nstate,real,MeshType>::FunctionalErrorNormLpBoundary(
    const double                               _normLp,
    std::vector<unsigned int>                  _boundary_vector,
    const bool                                 _use_all_boundaries,
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
    const bool                                 _uses_solution_values,
    const bool                                 _uses_solution_gradient) : 
        Functional<dim,nstate,real,MeshType>::Functional(_dg, _uses_solution_values, _uses_solution_gradient),
        normLp(_normLp),
        boundary_vector(_boundary_vector),
        use_all_boundaries(_use_all_boundaries) {}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 FunctionalErrorNormLpBoundary<dim,nstate,real,MeshType>::evaluate_boundary_integrand(
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real2> &physics,
    const unsigned int                                    boundary_id,
    const dealii::Point<dim,real2> &                      phys_coord,
    const dealii::Tensor<1,dim,real2> &                   /* normal */,
    const std::array<real2,nstate> &                      soln_at_q,
    const std::array<dealii::Tensor<1,dim,real2>,nstate> &/* soln_grad_at_q */) const
{
     real2 lpnorm_value = 0;

    // condition for whether the current cell should be evaluated
    auto boundary_vector_index = std::find(this->boundary_vector.begin(), this->boundary_vector.end(), boundary_id);
    bool eval_boundary = this->use_all_boundaries || boundary_vector_index != this->boundary_vector.end();

    if(!eval_boundary)
        return lpnorm_value;

    for(int istate = 0; istate < nstate; ++istate){
        const real2 uexact = physics.manufactured_solution_function->value(phys_coord, istate);
        lpnorm_value += pow(abs(soln_at_q[istate] - uexact), this->normLp);
    }

    return lpnorm_value;
}

template <int dim, int nstate, typename real, typename MeshType>
Functional<dim,nstate,real,MeshType>::Functional(
    std::shared_ptr<DGBase<dim,real,MeshType>> _dg,
    const bool                                 _uses_solution_values,
    const bool                                 _uses_solution_gradient)
    : dg(_dg)
    , uses_solution_values(_uses_solution_values)
    , uses_solution_gradient(_uses_solution_gradient)
    , pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
{ 
    using FadType = Sacado::Fad::DFad<real>;
    using FadFadType = Sacado::Fad::DFad<FadType>;
    physics_fad_fad = Physics::PhysicsFactory<dim,nstate,FadFadType>::create_Physics(dg->all_parameters);

    init_vectors();
}
template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::init_vectors()
{
    solution_value.reinit(dg->solution);
    solution_value *= 0.0;
    volume_nodes_value.reinit(dg->high_order_grid->volume_nodes);
    volume_nodes_value *= 0.0;

    solution_dIdW.reinit(dg->solution);
    solution_dIdW *= 0.0;
    volume_nodes_dIdW.reinit(dg->high_order_grid->volume_nodes);
    volume_nodes_dIdW *= 0.0;

    solution_dIdX.reinit(dg->solution);
    solution_dIdX *= 0.0;
    volume_nodes_dIdX.reinit(dg->high_order_grid->volume_nodes);
    volume_nodes_dIdX *= 0.0;

    solution_d2I.reinit(dg->solution);
    solution_d2I *= 0.0;
    volume_nodes_d2I.reinit(dg->high_order_grid->volume_nodes);
    volume_nodes_d2I *= 0.0;
}

template <int dim, int nstate, typename real, typename MeshType>
Functional<dim,nstate,real,MeshType>::Functional(
    std::shared_ptr<PHiLiP::DGBase<dim,real,MeshType>> _dg,
    std::shared_ptr<PHiLiP::Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>> >> _physics_fad_fad,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional(_dg, _uses_solution_values, _uses_solution_gradient)
{
    physics_fad_fad = _physics_fad_fad;
}

template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::set_state(const dealii::LinearAlgebra::distributed::Vector<real> &solution_set)
{
    dg->solution = solution_set;
}

template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::set_geom(const dealii::LinearAlgebra::distributed::Vector<real> &volume_nodes_set)
{
    dg->high_order_grid->volume_nodes = volume_nodes_set;
}

template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::allocate_dIdX(dealii::LinearAlgebra::distributed::Vector<real> &dIdX) const
{
    // allocating the vector
    dealii::IndexSet locally_owned_dofs = dg->high_order_grid->dof_handler_grid.locally_owned_dofs();
    dealii::IndexSet locally_relevant_dofs, ghost_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid->dof_handler_grid, locally_relevant_dofs);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);
    dIdX.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
}

template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::allocate_derivatives(const bool compute_dIdW, const bool compute_dIdX, const bool compute_d2I)
{
    if (compute_dIdW) {
        // allocating the vector
        dealii::IndexSet locally_owned_dofs = dg->dof_handler.locally_owned_dofs();
        dIdw.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    }
    if (compute_dIdX) {
        allocate_dIdX(dIdX);
    }
    if (compute_d2I) {
        {
            dealii::SparsityPattern sparsity_pattern_d2IdWdX = dg->get_d2RdWdX_sparsity_pattern ();
            const dealii::IndexSet &row_parallel_partitioning_d2IdWdX = dg->locally_owned_dofs;
            const dealii::IndexSet &col_parallel_partitioning_d2IdWdX = dg->high_order_grid->locally_owned_dofs_grid;
            d2IdWdX.reinit(row_parallel_partitioning_d2IdWdX, col_parallel_partitioning_d2IdWdX, sparsity_pattern_d2IdWdX, MPI_COMM_WORLD);
        }

        {
            dealii::SparsityPattern sparsity_pattern_d2IdWdW = dg->get_d2RdWdW_sparsity_pattern ();
            const dealii::IndexSet &row_parallel_partitioning_d2IdWdW = dg->locally_owned_dofs;
            const dealii::IndexSet &col_parallel_partitioning_d2IdWdW = dg->locally_owned_dofs;
            d2IdWdW.reinit(row_parallel_partitioning_d2IdWdW, col_parallel_partitioning_d2IdWdW, sparsity_pattern_d2IdWdW, MPI_COMM_WORLD);
        }

        {
            dealii::SparsityPattern sparsity_pattern_d2IdXdX = dg->get_d2RdXdX_sparsity_pattern ();
            const dealii::IndexSet &row_parallel_partitioning_d2IdXdX = dg->high_order_grid->locally_owned_dofs_grid;
            const dealii::IndexSet &col_parallel_partitioning_d2IdXdX = dg->high_order_grid->locally_owned_dofs_grid;
            d2IdXdX.reinit(row_parallel_partitioning_d2IdXdX, col_parallel_partitioning_d2IdXdX, sparsity_pattern_d2IdXdX, MPI_COMM_WORLD);
        }
    }
}


template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::set_derivatives(
    const bool compute_dIdW, const bool compute_dIdX, const bool compute_d2I,
    const Sacado::Fad::DFad<Sacado::Fad::DFad<real>> volume_local_sum,
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices,
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices)
{
    using FadType = Sacado::Fad::DFad<real>;

    const unsigned int n_total_indep = volume_local_sum.size();
    (void) n_total_indep; // Not used apart from assert.
    const unsigned int n_soln_dofs_cell = cell_soln_dofs_indices.size();
    const unsigned int n_metric_dofs_cell = cell_metric_dofs_indices.size();
    unsigned int i_derivative = 0;

    if (compute_dIdW) {
        std::vector<real> local_dIdw(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
            local_dIdw[idof] = volume_local_sum.dx(i_derivative++).val();
        }
        dIdw.add(cell_soln_dofs_indices, local_dIdw);
    }
    if (compute_dIdX) {
        std::vector<real> local_dIdX(n_metric_dofs_cell);
        for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof){
            local_dIdX[idof] = volume_local_sum.dx(i_derivative++).val();
        }
        dIdX.add(cell_metric_dofs_indices, local_dIdX);
    }
    if (compute_dIdW || compute_dIdX) AssertDimension(i_derivative, n_total_indep);
    if (compute_d2I) {
        std::vector<real> dWidW(n_soln_dofs_cell);
        std::vector<real> dWidX(n_metric_dofs_cell);
        std::vector<real> dXidX(n_metric_dofs_cell);


        i_derivative = 0;
        for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {

            unsigned int j_derivative = 0;
            const FadType dWi = volume_local_sum.dx(i_derivative++);

            for (unsigned int jdof=0; jdof<n_soln_dofs_cell; ++jdof) {
                dWidW[jdof] = dWi.dx(j_derivative++);
            }
            d2IdWdW.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) {
                dWidX[jdof] = dWi.dx(j_derivative++);
            }
            d2IdWdX.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {

            const FadType dXi = volume_local_sum.dx(i_derivative++);

            unsigned int j_derivative = n_soln_dofs_cell;
            for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) {
                dXidX[jdof] = dXi.dx(j_derivative++);
            }
            d2IdXdX.add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices, dXidX);
        }
    }
    AssertDimension(i_derivative, n_total_indep);
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 Functional<dim, nstate, real, MeshType>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const std::vector< real2 > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    const unsigned int n_vol_quad_pts = volume_quadrature.size();
    const unsigned int n_soln_dofs_cell = soln_coeff.size();
    const unsigned int n_metric_dofs_cell = coords_coeff.size();

    real2 volume_local_sum = 0.0;
    for (unsigned int iquad=0; iquad<n_vol_quad_pts; ++iquad) {

        const dealii::Point<dim,double> &ref_point = volume_quadrature.point(iquad);
        const double quad_weight = volume_quadrature.weight(iquad);

        // Obtain physical quadrature coordinates (Might be used if there is a source term or a wall distance)
        // and evaluate metric terms such as the metric Jacobian, its inverse transpose, and its determinant
        dealii::Point<dim,real2> phys_coord;
        for (int d=0;d<dim;++d) { phys_coord[d] = 0.0;}
        std::array< dealii::Tensor<1,dim,real2>, dim > coord_grad; // Tensor initialize with zeros
        dealii::Tensor<2,dim,real2> metric_jacobian;
        for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {
            const unsigned int axis = fe_metric.system_to_component_index(idof).first;
            phys_coord[axis] += coords_coeff[idof] * fe_metric.shape_value(idof, ref_point);
            coord_grad[axis] += coords_coeff[idof] * fe_metric.shape_grad (idof, ref_point);
        }
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[row][col] = coord_grad[row][col];
            }
        }
        const real2 jacobian_determinant = dealii::determinant(metric_jacobian);
        dealii::Tensor<2,dim,real2> jacobian_transpose_inverse;
        jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian));

        // Evaluate the solution and gradient at the quadrature points
        std::array<real2, nstate> soln_at_q;
        soln_at_q.fill(0.0);
        std::array< dealii::Tensor<1,dim,real2>, nstate > soln_grad_at_q;
        for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {
            const unsigned int istate = fe_solution.system_to_component_index(idof).first;
            if (uses_solution_values) {
                soln_at_q[istate]  += soln_coeff[idof] * fe_solution.shape_value(idof,ref_point);
            }
            if (uses_solution_gradient) {
                const dealii::Tensor<1,dim,real2> phys_shape_grad = dealii::contract<1,0>(jacobian_transpose_inverse, fe_solution.shape_grad(idof,ref_point));
                soln_grad_at_q[istate] += soln_coeff[idof] * phys_shape_grad;
            }
        }
        real2 volume_integrand = this->evaluate_volume_integrand(physics, phys_coord, soln_at_q, soln_grad_at_q);

        volume_local_sum += volume_integrand * jacobian_determinant * quad_weight;
    }
    return volume_local_sum;
}

template <int dim, int nstate, typename real, typename MeshType>
template <typename real2>
real2 Functional<dim,nstate,real,MeshType>::evaluate_boundary_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const unsigned int boundary_id,
    const std::vector< real2 > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const unsigned int face_number,
    const dealii::Quadrature<dim-1> &fquadrature) const
{
    const unsigned int n_face_quad_pts = fquadrature.size();
    const unsigned int n_soln_dofs_cell = soln_coeff.size();
    const unsigned int n_metric_dofs_cell = coords_coeff.size();

    const dealii::Quadrature<dim> face_quadrature = dealii::QProjector<dim>::project_to_face( dealii::ReferenceCell::get_hypercube(dim),
                                                                                              fquadrature,
                                                                                              face_number);
    const dealii::Tensor<1,dim,real> surface_unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];

    real2 boundary_local_sum = 0.0;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Point<dim,double> &ref_point = face_quadrature.point(iquad);
        const double quad_weight = face_quadrature.weight(iquad);

        // Obtain physical quadrature coordinates (Might be used if there is a source term or a wall distance)
        // and evaluate metric terms such as the metric Jacobian, its inverse transpose, and its determinant
        dealii::Point<dim,real2> phys_coord;
        for (int d=0;d<dim;++d) { phys_coord[d] = 0.0;}
        std::array< dealii::Tensor<1,dim,real2>, dim > coord_grad; // Tensor initialize with zeros
        dealii::Tensor<2,dim,real2> metric_jacobian;
        for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {
            const unsigned int axis = fe_metric.system_to_component_index(idof).first;
            phys_coord[axis] += coords_coeff[idof] * fe_metric.shape_value(idof, ref_point);
            coord_grad[axis] += coords_coeff[idof] * fe_metric.shape_grad (idof, ref_point);
        }
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[row][col] = coord_grad[row][col];
            }
        }
        const real2 jacobian_determinant = dealii::determinant(metric_jacobian);
        dealii::Tensor<2,dim,real2> jacobian_transpose_inverse;
        jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian));

        const dealii::Tensor<1,dim,real2> phys_normal = vmult(jacobian_transpose_inverse, surface_unit_normal);
        const real2 area = norm(phys_normal);
        const dealii::Tensor<1,dim,real2> phys_unit_normal = phys_normal/area;

        real2 surface_jacobian_determinant = area*jacobian_determinant;

        // Evaluate the solution and gradient at the quadrature points
        std::array<real2, nstate> soln_at_q;
        soln_at_q.fill(0.0);
        std::array< dealii::Tensor<1,dim,real2>, nstate > soln_grad_at_q;
        for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {
            const unsigned int istate = fe_solution.system_to_component_index(idof).first;
            if (uses_solution_values) {
                soln_at_q[istate]  += soln_coeff[idof] * fe_solution.shape_value(idof,ref_point);
            }
            if (uses_solution_gradient) {
                const dealii::Tensor<1,dim,real2> phys_shape_grad = dealii::contract<1,0>(jacobian_transpose_inverse, fe_solution.shape_grad(idof,ref_point));
                soln_grad_at_q[istate] += soln_coeff[idof] * phys_shape_grad;
            }
        }
        real2 boundary_integrand = this->evaluate_boundary_integrand(physics, boundary_id, phys_coord, phys_unit_normal, soln_at_q, soln_grad_at_q);

        boundary_local_sum += boundary_integrand * surface_jacobian_determinant * quad_weight;
    }
    return boundary_local_sum;
}

template <int dim, int nstate, typename real, typename MeshType>
real Functional<dim,nstate,real,MeshType>::evaluate_boundary_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real> &physics,
    const unsigned int boundary_id,
    const std::vector< real > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const unsigned int face_number,
    const dealii::Quadrature<dim-1> &fquadrature) const
{
    return evaluate_boundary_cell_functional<real>(physics, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, face_number, fquadrature);
}

template <int dim, int nstate, typename real, typename MeshType>
Sacado::Fad::DFad<Sacado::Fad::DFad<real>> Functional<dim,nstate,real,MeshType>::evaluate_boundary_cell_functional(
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> &physics,
    const unsigned int boundary_id,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const unsigned int face_number,
    const dealii::Quadrature<dim-1> &fquadrature) const
{
    return evaluate_boundary_cell_functional<Sacado::Fad::DFad<Sacado::Fad::DFad<real>>>(physics, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, face_number, fquadrature);
}

template <int dim, int nstate, typename real, typename MeshType>
real Functional<dim,nstate,real,MeshType>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real> &physics,
    const std::vector< real > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    return evaluate_volume_cell_functional<real>(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
}

template <int dim, int nstate, typename real, typename MeshType>
Sacado::Fad::DFad<Sacado::Fad::DFad<real>> Functional<dim,nstate,real,MeshType>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> &physics_fad_fad,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    return evaluate_volume_cell_functional<Sacado::Fad::DFad<Sacado::Fad::DFad<real>>>(physics_fad_fad, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
}

template <int dim, int nstate, typename real, typename MeshType>
void Functional<dim,nstate,real,MeshType>::need_compute(bool &compute_value, bool &compute_dIdW, bool &compute_dIdX, bool &compute_d2I)
{
    if (compute_value) {
        pcout << " with value...";

        if (dg->solution.size() == solution_value.size() 
            && dg->high_order_grid->volume_nodes.size() == volume_nodes_value.size()) {

            auto diff_sol = dg->solution;
            diff_sol -= solution_value;
            const double l2_norm_sol = diff_sol.l2_norm();

            if (l2_norm_sol == 0.0) {

                auto diff_node = dg->high_order_grid->volume_nodes;
                diff_node -= volume_nodes_value;
                const double l2_norm_node = diff_node.l2_norm();

                if (l2_norm_node == 0.0) {
                    pcout << " which is already assembled...";
                    compute_value = false;
                }
            }
        }
        solution_value = dg->solution;
        volume_nodes_value = dg->high_order_grid->volume_nodes;
    }
    if (compute_dIdW) {
        pcout << " with dIdW...";

        if (dg->solution.size() == solution_dIdW.size() 
            && dg->high_order_grid->volume_nodes.size() == volume_nodes_dIdW.size()) {

            auto diff_sol = dg->solution;
            diff_sol -= solution_dIdW;
            const double l2_norm_sol = diff_sol.l2_norm();

            if (l2_norm_sol == 0.0) {

                auto diff_node = dg->high_order_grid->volume_nodes;
                diff_node -= volume_nodes_dIdW;
                const double l2_norm_node = diff_node.l2_norm();

                if (l2_norm_node == 0.0) {
                    pcout << " which is already assembled...";
                    compute_dIdW = false;
                }
            }
        }
        solution_dIdW = dg->solution;
        volume_nodes_dIdW = dg->high_order_grid->volume_nodes;
    }
    if (compute_dIdX) {
        pcout << " with dIdX...";

        if (dg->solution.size() == solution_dIdX.size() 
            && dg->high_order_grid->volume_nodes.size() == volume_nodes_dIdX.size()) {
            auto diff_sol = dg->solution;
            diff_sol -= solution_dIdX;
            const double l2_norm_sol = diff_sol.l2_norm();

            if (l2_norm_sol == 0.0) {

                auto diff_node = dg->high_order_grid->volume_nodes;
                diff_node -= volume_nodes_dIdX;
                const double l2_norm_node = diff_node.l2_norm();

                if (l2_norm_node == 0.0) {
                    pcout << " which is already assembled...";
                    compute_dIdX = false;
                }
            }
        }
        solution_dIdX = dg->solution;
        volume_nodes_dIdX = dg->high_order_grid->volume_nodes;
    }
    if (compute_d2I) {
        pcout << " with d2IdWdW, d2IdWdX, d2IdXdX...";

        if (dg->solution.size() == solution_d2I.size() 
            && dg->high_order_grid->volume_nodes.size() == volume_nodes_d2I.size()) {

            auto diff_sol = dg->solution;
            diff_sol -= solution_d2I;
            const double l2_norm_sol = diff_sol.l2_norm();

            if (l2_norm_sol == 0.0) {

                auto diff_node = dg->high_order_grid->volume_nodes;
                diff_node -= volume_nodes_d2I;
                const double l2_norm_node = diff_node.l2_norm();

                if (l2_norm_node == 0.0) {

                    pcout << " which is already assembled...";
                    compute_d2I = false;
                }
            }
        }
        solution_d2I = dg->solution;
        volume_nodes_d2I = dg->high_order_grid->volume_nodes;
    }
}

template <int dim, int nstate, typename real, typename MeshType>
real Functional<dim, nstate, real, MeshType>::evaluate_functional(
    const bool compute_dIdW,
    const bool compute_dIdX,
    const bool compute_d2I)
{
    using FadType = Sacado::Fad::DFad<real>;
    using FadFadType = Sacado::Fad::DFad<FadType>;

    bool actually_compute_value = true;
    bool actually_compute_dIdW = compute_dIdW;
    bool actually_compute_dIdX = compute_dIdX;
    bool actually_compute_d2I  = compute_d2I;

    pcout << "Evaluating functional... ";
    need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    pcout << std::endl;

    if (!actually_compute_value && !actually_compute_dIdW && !actually_compute_dIdX && !actually_compute_d2I) {
        return current_functional_value;
    }

    // Returned value
    real local_functional = 0.0;

    // for taking the local derivatives
    const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid->fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<FadFadType> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real>   local_dIdw(max_dofs_per_cell);

    std::vector<real>   local_dIdX(n_metric_dofs_cell);

    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg->fe_collection, dg->face_quadrature_collection,   this->face_update_flags);

    allocate_derivatives(actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);

    dg->solution.update_ghost_values();
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    auto soln_cell = dg->dof_handler.begin_active();
    for( ; soln_cell != dg->dof_handler.end(); ++soln_cell, ++metric_cell) {
        if(!soln_cell->is_locally_owned()) continue;

        // setting up the volume integration
        // const unsigned int i_mapp = 0;
        const unsigned int i_fele = soln_cell->active_fe_index();
        const unsigned int i_quad = i_fele;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg->fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        soln_cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);

        // Get metric coefficients
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector< FadFadType > coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            coords_coeff[idof] = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
        }

        // Setup automatic differentiation
        unsigned int n_total_indep = 0;
        if (actually_compute_dIdW || actually_compute_d2I) n_total_indep += n_soln_dofs_cell;
        if (actually_compute_dIdX || actually_compute_d2I) n_total_indep += n_metric_dofs_cell;
        unsigned int i_derivative = 0;
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            const real val = dg->solution[cell_soln_dofs_indices[idof]];
            soln_coeff[idof] = val;
            if (actually_compute_dIdW || actually_compute_d2I) soln_coeff[idof].diff(i_derivative++, n_total_indep);
        }
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            const real val = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
            coords_coeff[idof] = val;
            if (actually_compute_dIdX || actually_compute_d2I) coords_coeff[idof].diff(i_derivative++, n_total_indep);
        }
        AssertDimension(i_derivative, n_total_indep);
        if (actually_compute_d2I) {
            unsigned int i_derivative = 0;
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
                const real val = dg->solution[cell_soln_dofs_indices[idof]];
                soln_coeff[idof].val() = val;
                soln_coeff[idof].val().diff(i_derivative++, n_total_indep);
            }
            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                const real val = dg->high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
                coords_coeff[idof].val() = val;
                coords_coeff[idof].val().diff(i_derivative++, n_total_indep);
            }
        }
        AssertDimension(i_derivative, n_total_indep);

        // Get quadrature point on reference cell
        const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[i_quad];

        // Evaluate integral on the cell volume
        FadFadType volume_local_sum = evaluate_volume_cell_functional(*physics_fad_fad, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = soln_cell->face(iface);
            
            if(face->at_boundary()){

                const unsigned int boundary_id = face->boundary_id();

                //fe_values_collection_face.reinit(soln_cell, iface, i_quad, i_mapp, i_fele);
                //const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();
                //volume_local_sum += this->evaluate_cell_boundary(*physics_fad_fad, boundary_id, fe_values_face, soln_coeff);
                volume_local_sum += this->evaluate_boundary_cell_functional(*physics_fad_fad, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, iface, dg->face_quadrature_collection[i_quad]);
            }

        }

        local_functional += volume_local_sum.val().val();
        // now getting the values and adding them to the derivaitve vector

        i_derivative = 0;
        if (actually_compute_dIdW) {
            local_dIdw.resize(n_soln_dofs_cell);
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
                local_dIdw[idof] = volume_local_sum.dx(i_derivative++).val();
            }
            dIdw.add(cell_soln_dofs_indices, local_dIdw);
        }
        if (actually_compute_dIdX) {
            local_dIdX.resize(n_metric_dofs_cell);
            for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof){
                local_dIdX[idof] = volume_local_sum.dx(i_derivative++).val();
            }
            dIdX.add(cell_metric_dofs_indices, local_dIdX);
        }
        if (actually_compute_dIdW || actually_compute_dIdX) AssertDimension(i_derivative, n_total_indep);
        if (actually_compute_d2I) {
            std::vector<real> dWidW(n_soln_dofs_cell);
            std::vector<real> dWidX(n_metric_dofs_cell);
            std::vector<real> dXidX(n_metric_dofs_cell);

            i_derivative = 0;
            for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {
                unsigned int j_derivative = 0;
                const FadType dWi = volume_local_sum.dx(i_derivative++);
                for (unsigned int jdof=0; jdof<n_soln_dofs_cell; ++jdof) {
                    dWidW[jdof] = dWi.dx(j_derivative++);
                }
                d2IdWdW.add(cell_soln_dofs_indices[idof], cell_soln_dofs_indices, dWidW);

                for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) {
                    dWidX[jdof] = dWi.dx(j_derivative++);
                }
                d2IdWdX.add(cell_soln_dofs_indices[idof], cell_metric_dofs_indices, dWidX);
            }

            for (unsigned int idof=0; idof<n_metric_dofs_cell; ++idof) {

                const FadType dXi = volume_local_sum.dx(i_derivative++);

                unsigned int j_derivative = n_soln_dofs_cell;
                for (unsigned int jdof=0; jdof<n_metric_dofs_cell; ++jdof) {
                    dXidX[jdof] = dXi.dx(j_derivative++);
                }
                d2IdXdX.add(cell_metric_dofs_indices[idof], cell_metric_dofs_indices, dXidX);
            }
        }
        AssertDimension(i_derivative, n_total_indep);
    }

    current_functional_value = dealii::Utilities::MPI::sum(local_functional, MPI_COMM_WORLD);
    // compress before the return
    if (actually_compute_dIdW) dIdw.compress(dealii::VectorOperation::add);
    if (actually_compute_dIdX) dIdX.compress(dealii::VectorOperation::add);
    if (actually_compute_d2I) {
        d2IdWdW.compress(dealii::VectorOperation::add);
        d2IdWdX.compress(dealii::VectorOperation::add);
        d2IdXdX.compress(dealii::VectorOperation::add);
    }

    return current_functional_value;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> Functional<dim,nstate,real,MeshType>::evaluate_dIdw_finiteDifferences(
    PHiLiP::DGBase<dim,real,MeshType> &dg, 
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
    const double stepsize)
{
    // for taking the local derivatives
    double local_sum_old;
    double local_sum_new;

    // vector for storing the derivatives with respect to each DOF
    dealii::LinearAlgebra::distributed::Vector<real> dIdw;

    // allocating the vector
    dealii::IndexSet locally_owned_dofs = dg.dof_handler.locally_owned_dofs();
    dIdw.reinit(locally_owned_dofs, MPI_COMM_WORLD);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
    std::vector<real> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real> local_dIdw(max_dofs_per_cell);

    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

    dg.solution.update_ghost_values();
    auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
    auto cell = dg.dof_handler.begin_active();
    for( ; cell != dg.dof_handler.end(); ++cell, ++metric_cell) {
        if(!cell->is_locally_owned()) continue;

        // setting up the volume integration
        const unsigned int i_mapp = 0;
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg.fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            soln_coeff[idof] = dg.solution[cell_soln_dofs_indices[idof]];
        }

        // Get metric coefficients
        const dealii::FESystem<dim,dim> &fe_metric = dg.high_order_grid->fe_system;
        const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector<real> coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            coords_coeff[idof] = dg.high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
        }

        const dealii::Quadrature<dim> &volume_quadrature = dg.volume_quadrature_collection[i_quad];

        // adding the contribution from the current volume, also need to pass the solution vector on these points
        //local_sum_old = this->evaluate_volume_integrand(physics, fe_values_volume, soln_coeff);
        local_sum_old = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // Next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){
                fe_values_collection_face.reinit(cell, iface, i_quad, i_mapp, i_fele);

                const unsigned int boundary_id = face->boundary_id();

                //const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();
                //local_sum_old += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
                local_sum_old += this->evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, iface, dg.face_quadrature_collection[i_quad]);
            }

        }

        // now looping over all the DOFs in this cell and taking the FD
        local_dIdw.resize(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
            // for each dof copying the solution
            for(unsigned int idof2 = 0; idof2 < n_soln_dofs_cell; ++idof2){
                soln_coeff[idof2] = dg.solution[cell_soln_dofs_indices[idof2]];
            }
            soln_coeff[idof] += stepsize;

            // then peturb the idof'th value
            // local_sum_new = this->evaluate_volume_integrand(physics, fe_values_volume, soln_coeff);
            local_sum_new = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

            // Next looping over the faces of the cell checking for boundary elements
            for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
                auto face = cell->face(iface);
                
                if(face->at_boundary()){
                    fe_values_collection_face.reinit(cell, iface, i_quad, i_mapp, i_fele);

                    const unsigned int boundary_id = face->boundary_id();

                    //const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();
                    //local_sum_new += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
                    local_sum_new += this->evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, iface, dg.face_quadrature_collection[i_quad]);
                }

            }

            local_dIdw[idof] = (local_sum_new-local_sum_old)/stepsize;
        }

        dIdw.add(cell_soln_dofs_indices, local_dIdw);
    }
    // compress before the return
    dIdw.compress(dealii::VectorOperation::add);
    
    return dIdw;
}

template <int dim, int nstate, typename real, typename MeshType>
dealii::LinearAlgebra::distributed::Vector<real> Functional<dim,nstate,real,MeshType>::evaluate_dIdX_finiteDifferences(
    PHiLiP::DGBase<dim,real,MeshType> &dg, 
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
    const double stepsize)
{
    // for taking the local derivatives
    double local_sum_old;
    double local_sum_new;

    // vector for storing the derivatives with respect to each DOF
    dealii::LinearAlgebra::distributed::Vector<real> dIdX_FD;
    allocate_dIdX(dIdX_FD);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
    std::vector<real> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real> local_dIdX(max_dofs_per_cell);

    const auto mapping = (*(dg.high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

    dg.solution.update_ghost_values();
    auto metric_cell = dg.high_order_grid->dof_handler_grid.begin_active();
    auto cell = dg.dof_handler.begin_active();
    for( ; cell != dg.dof_handler.end(); ++cell, ++metric_cell) {
        if(!cell->is_locally_owned()) continue;

        // setting up the volume integration
        //const unsigned int i_mapp = 0;
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg.fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            soln_coeff[idof] = dg.solution[cell_soln_dofs_indices[idof]];
        }

        // Get metric coefficients
        const dealii::FESystem<dim,dim> &fe_metric = dg.high_order_grid->fe_system;
        const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector<real> coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            coords_coeff[idof] = dg.high_order_grid->volume_nodes[cell_metric_dofs_indices[idof]];
        }

        const dealii::Quadrature<dim> &volume_quadrature = dg.volume_quadrature_collection[i_quad];

        // adding the contribution from the current volume, also need to pass the solution vector on these points
        //local_sum_old = this->evaluate_volume_integrand(physics, fe_values_volume, soln_coeff);
        local_sum_old = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // Next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){

                const unsigned int boundary_id = face->boundary_id();

                //fe_values_collection_face.reinit(cell, iface, i_quad, i_mapp, i_fele);
                //const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();
                //local_sum_old += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
                local_sum_old += this->evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, iface, dg.face_quadrature_collection[i_quad]);
            }

        }

        // now looping over all the DOFs in this cell and taking the FD
        local_dIdX.resize(n_metric_dofs_cell);
        for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof){
            // for each dof copying the solution
            for(unsigned int idof2 = 0; idof2 < n_metric_dofs_cell; ++idof2){
                coords_coeff[idof2] = dg.high_order_grid->volume_nodes[cell_metric_dofs_indices[idof2]];
            }
            coords_coeff[idof] += stepsize;

            // then peturb the idof'th value
            local_sum_new = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

            // Next looping over the faces of the cell checking for boundary elements
            for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
                auto face = cell->face(iface);
                
                if(face->at_boundary()){

                    const unsigned int boundary_id = face->boundary_id();

                    //fe_values_collection_face.reinit(cell, iface, i_quad, i_mapp, i_fele);
                    //const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();
                    //local_sum_new += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
                    local_sum_new += this->evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, fe_solution, coords_coeff, fe_metric, iface, dg.face_quadrature_collection[i_quad]);
                }

            }

            local_dIdX[idof] = (local_sum_new-local_sum_old)/stepsize;
        }

        dIdX_FD.add(cell_metric_dofs_indices, local_dIdX);
    }
    // compress before the return
    dIdX_FD.compress(dealii::VectorOperation::add);
    
    return dIdX_FD;
}

template <int dim, int nstate, typename real, typename MeshType>
std::shared_ptr< Functional<dim,nstate,real,MeshType> >
FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(
    PHiLiP::Parameters::AllParameters const *const param,
    std::shared_ptr< PHiLiP::DGBase<dim,real,MeshType> > dg)
{
    return FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(param->grid_refinement_study_param.functional_param, dg);
}

template <int dim, int nstate, typename real, typename MeshType>
std::shared_ptr< Functional<dim,nstate,real,MeshType> >
FunctionalFactory<dim,nstate,real,MeshType>::create_Functional(
    PHiLiP::Parameters::FunctionalParam param,
    std::shared_ptr< PHiLiP::DGBase<dim,real,MeshType> > dg)
{
    using FadFadType = Sacado::Fad::DFad<FadType>;

    using FunctionalTypeEnum       = Parameters::FunctionalParam::FunctionalType;
    using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
    FunctionalTypeEnum functional_type = param.functional_type;

    const double normLp = param.normLp;

    ManufacturedSolutionEnum  weight_function_type = param.weight_function_type;
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > weight_function_double 
        = ManufacturedSolutionFactory<dim,real>::create_ManufacturedSolution(weight_function_type, nstate);
    std::shared_ptr< ManufacturedSolutionFunction<dim,FadFadType> > weight_function_adtype 
        = ManufacturedSolutionFactory<dim,FadFadType>::create_ManufacturedSolution(weight_function_type, nstate);
    
    const bool use_weight_function_laplacian       = param.use_weight_function_laplacian;

    std::vector<unsigned int> boundary_vector    = param.boundary_vector;
    const bool                use_all_boundaries = param.use_all_boundaries;

    if(functional_type == FunctionalTypeEnum::normLp_volume){
        return std::make_shared<FunctionalNormLpVolume<dim,nstate,real,MeshType>>(
            normLp,
            dg,
            true,
            false);
    }else if(functional_type == FunctionalTypeEnum::normLp_boundary){
        return std::make_shared<FunctionalNormLpBoundary<dim,nstate,real,MeshType>>(
            normLp,
            boundary_vector,
            use_all_boundaries,
            dg,
            true,
            false);
    }else if(functional_type == FunctionalTypeEnum::weighted_integral_volume){
        return std::make_shared<FunctionalWeightedIntegralVolume<dim,nstate,real,MeshType>>(
            weight_function_double,
            weight_function_adtype,
            use_weight_function_laplacian,
            dg,
            true,
            false);
    }else if(functional_type == FunctionalTypeEnum::weighted_integral_boundary){
        return std::make_shared<FunctionalWeightedIntegralBoundary<dim,nstate,real,MeshType>>(
            weight_function_double,
            weight_function_adtype,
            use_weight_function_laplacian,
            boundary_vector,
            use_all_boundaries,
            dg,
            true,
            false);
    }else if(functional_type == FunctionalTypeEnum::error_normLp_volume){
        return std::make_shared<FunctionalErrorNormLpVolume<dim,nstate,real,MeshType>>(
            normLp,
            dg,
            true,
            false);
    }else if(functional_type == FunctionalTypeEnum::error_normLp_boundary){
        return std::make_shared<FunctionalErrorNormLpBoundary<dim,nstate,real,MeshType>>(
            normLp,
            boundary_vector,
            use_all_boundaries,
            dg,
            true,
            false);
    }else{
        std::cout << "Invalid Functional." << std::endl;
    }

    return nullptr;
}

// dealii::Triangulation<PHILIP_DIM>
template class FunctionalNormLpVolume <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class FunctionalNormLpBoundary <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class FunctionalErrorNormLpVolume <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class Functional <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class FunctionalFactory <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

// dealii::parallel::shared::Triangulation<PHILIP_DIM>
template class FunctionalNormLpVolume <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class FunctionalNormLpBoundary <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class FunctionalErrorNormLpVolume <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class Functional <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

template class FunctionalFactory <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM != 1
// dealii::parallel::distributed::Triangulation<PHILIP_DIM>
template class FunctionalNormLpVolume <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpVolume <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class FunctionalNormLpBoundary <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalNormLpBoundary <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralVolume <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalWeightedIntegralBoundary <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class FunctionalErrorNormLpVolume <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpVolume <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalErrorNormLpBoundary <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class Functional <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class Functional <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;

template class FunctionalFactory <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class FunctionalFactory <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace
