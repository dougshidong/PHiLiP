// includes
#include <vector>

#include <Sacado.hpp>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/base/qprojector.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

#include "physics/physics.h"
#include "physics/physics_factory.h"
#include "dg/dg.h"
#include "target_functional.h"

namespace PHiLiP {
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

template <int dim, int nstate, typename real>
TargetFunctional<dim,nstate,real>::TargetFunctional(
    std::shared_ptr<DGBase<dim,real>> _dg,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional<dim,nstate,real>::Functional(_dg, _uses_solution_values, _uses_solution_gradient)
    , target_solution(dg->solution)
{ 
    using FadType = Sacado::Fad::DFad<real>;
    using FadFadType = Sacado::Fad::DFad<FadType>;
    physics_fad_fad = Physics::PhysicsFactory<dim,nstate,FadFadType>::create_Physics(dg->all_parameters);
}

template <int dim, int nstate, typename real>
TargetFunctional<dim,nstate,real>::TargetFunctional(
    std::shared_ptr<DGBase<dim,real>> _dg,
    const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional<dim,nstate,real>::Functional(_dg, _uses_solution_values, _uses_solution_gradient)
    , target_solution(target_solution)
{ 
    using FadType = Sacado::Fad::DFad<real>;
    using FadFadType = Sacado::Fad::DFad<FadType>;
    physics_fad_fad = Physics::PhysicsFactory<dim,nstate,FadFadType>::create_Physics(dg->all_parameters);
}

template <int dim, int nstate, typename real>
TargetFunctional<dim,nstate,real>::TargetFunctional(
    std::shared_ptr<DGBase<dim,real>> _dg,
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> > _physics_fad_fad,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional<dim,nstate,real>::Functional(_dg, _physics_fad_fad, _uses_solution_values, _uses_solution_gradient)
    , target_solution(dg->solution)
{ }

template <int dim, int nstate, typename real>
TargetFunctional<dim,nstate,real>::TargetFunctional(
    std::shared_ptr<DGBase<dim,real>> _dg,
    const dealii::LinearAlgebra::distributed::Vector<real> &target_solution,
    std::shared_ptr< Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> > _physics_fad_fad,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : Functional<dim,nstate,real>::Functional(_dg, _physics_fad_fad, _uses_solution_values, _uses_solution_gradient)
    , target_solution(target_solution)
{ }


template <int dim, int nstate, typename real>
template <typename real2>
real2 TargetFunctional<dim, nstate, real>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const std::vector< real2 > &soln_coeff,
    const std::vector< real > &target_soln_coeff,
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
        std::array<real, nstate> target_soln_at_q;
        soln_at_q.fill(0.0);
        target_soln_at_q.fill(0.0);
        std::array< dealii::Tensor<1,dim,real2>, nstate > soln_grad_at_q;
        std::array< dealii::Tensor<1,dim,real2>, nstate > target_soln_grad_at_q; // Target solution grad needs to be FadType since metric term involve mesh DoFs
        for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {
            const unsigned int istate = fe_solution.system_to_component_index(idof).first;
            if (uses_solution_values) {
                soln_at_q[istate]  += soln_coeff[idof] * fe_solution.shape_value(idof,ref_point);
                target_soln_at_q[istate]  += target_soln_coeff[idof] * fe_solution.shape_value(idof,ref_point);
            }
            if (uses_solution_gradient) {
                const dealii::Tensor<1,dim,real2> phys_shape_grad = dealii::contract<1,0>(jacobian_transpose_inverse, fe_solution.shape_grad(idof,ref_point));
                soln_grad_at_q[istate] += soln_coeff[idof] * phys_shape_grad;
                target_soln_grad_at_q[istate] += target_soln_coeff[idof] * phys_shape_grad;
            }
        }
        real2 volume_integrand = evaluate_volume_integrand(physics, phys_coord, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);

        (void) jacobian_determinant;
        (void) quad_weight;
        volume_local_sum += volume_integrand;// * jacobian_determinant * quad_weight;
        if (volume_local_sum != 0.0 && jacobian_determinant < 0) {
            std::cout << "Bad jacobian... setting volume_local_sum *= 1e200" << std::endl;
            volume_local_sum += 1e200;
        }
    }
    return volume_local_sum;
}

template <int dim, int nstate, typename real>
real TargetFunctional<dim, nstate, real>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real> &physics,
    const std::vector< real > &soln_coeff,
    const std::vector< real > &target_soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    return evaluate_volume_cell_functional<real>(physics, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
}

template <int dim, int nstate, typename real>
Sacado::Fad::DFad<Sacado::Fad::DFad<real>> TargetFunctional<dim, nstate, real>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> &physics_fad_fad,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &soln_coeff,
    const std::vector< real > &target_soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature) const
{
    return evaluate_volume_cell_functional<Sacado::Fad::DFad<Sacado::Fad::DFad<real>>>(physics_fad_fad, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
}

template <int dim, int nstate, typename real>
template <typename real2>
real2 TargetFunctional<dim, nstate, real>::evaluate_boundary_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const unsigned int boundary_id,
    const std::vector< real2 > &soln_coeff,
    const std::vector< real > &target_soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim-1> &fquadrature,
    const unsigned int face_number) const
{
    const dealii::Quadrature<dim> face_quadrature = dealii::QProjector<dim>::project_to_face( dealii::ReferenceCell::get_hypercube(dim), fquadrature, face_number);
    const dealii::Tensor<1,dim,real> ref_unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];

    const unsigned int n_face_quad_pts = face_quadrature.size();
    const unsigned int n_soln_dofs_cell = soln_coeff.size();
    const unsigned int n_metric_dofs_cell = coords_coeff.size();

    real2 face_local_sum = 0.0;
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

        // Evaluate the solution and gradient at the quadrature points
        std::array<real2, nstate> soln_at_q;
        std::array<real, nstate> target_soln_at_q;
        soln_at_q.fill(0.0);
        target_soln_at_q.fill(0.0);
        std::array< dealii::Tensor<1,dim,real2>, nstate > soln_grad_at_q;
        std::array< dealii::Tensor<1,dim,real2>, nstate > target_soln_grad_at_q; // Target solution grad needs to be FadType since metric term involve mesh DoFs
        for (unsigned int idof=0; idof<n_soln_dofs_cell; ++idof) {
            const unsigned int istate = fe_solution.system_to_component_index(idof).first;
            if (uses_solution_values) {
                soln_at_q[istate]  += soln_coeff[idof] * fe_solution.shape_value(idof,ref_point);
                target_soln_at_q[istate]  += target_soln_coeff[idof] * fe_solution.shape_value(idof,ref_point);
            }
            if (uses_solution_gradient) {
                const dealii::Tensor<1,dim,real2> phys_shape_grad = dealii::contract<1,0>(jacobian_transpose_inverse, fe_solution.shape_grad(idof,ref_point));
                soln_grad_at_q[istate] += soln_coeff[idof] * phys_shape_grad;
                target_soln_grad_at_q[istate] += target_soln_coeff[idof] * phys_shape_grad;
            }
        }

        const dealii::Tensor<1,dim,real2> phys_normal = vmult(jacobian_transpose_inverse, ref_unit_normal);
        const real2 area = norm(phys_normal);
        const dealii::Tensor<1,dim,real2> phys_unit_normal = phys_normal/area;
        real2 boundary_integrand = evaluate_boundary_integrand(physics, boundary_id, phys_coord, phys_unit_normal, soln_at_q, target_soln_at_q, soln_grad_at_q, target_soln_grad_at_q);

        real2 surface_jacobian_determinant = area*jacobian_determinant;

        face_local_sum += boundary_integrand * surface_jacobian_determinant * quad_weight;
        if (face_local_sum != 0.0 && jacobian_determinant < 0) {
            std::cout << "Bad jacobian... setting face_local_sum *= 1e40" << std::endl;
            face_local_sum += 1e40;
        }
        //face_local_sum += boundary_integrand;// * jacobian_determinant * quad_weight;
    }
    return face_local_sum;
}

template <int dim, int nstate, typename real>
real TargetFunctional<dim, nstate, real>::evaluate_boundary_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real> &physics,
    const unsigned int boundary_id,
    const std::vector< real > &soln_coeff,
    const std::vector< real > &target_soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim-1> &face_quadrature,
    const unsigned int face_number) const
{
    return evaluate_boundary_cell_functional<real>(physics, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, face_quadrature, face_number);
}

template <int dim, int nstate, typename real>
Sacado::Fad::DFad<Sacado::Fad::DFad<real>> TargetFunctional<dim, nstate, real>::evaluate_boundary_cell_functional(
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<Sacado::Fad::DFad<real>>> &physics_fad_fad,
    const unsigned int boundary_id,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &soln_coeff,
    const std::vector< real > &target_soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim-1> &face_quadrature,
    const unsigned int face_number) const
{
    return evaluate_boundary_cell_functional<Sacado::Fad::DFad<Sacado::Fad::DFad<real>>>(physics_fad_fad, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, face_quadrature, face_number);
}

template <int dim, int nstate, typename real>
real TargetFunctional<dim, nstate, real>::evaluate_functional(
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

    Functional<dim,nstate,real>::pcout << "Evaluating functional... ";
    Functional<dim,nstate,real>::need_compute(actually_compute_value, actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);
    Functional<dim,nstate,real>::pcout << std::endl;
    if (!actually_compute_value && !actually_compute_dIdW && !actually_compute_dIdX && !actually_compute_d2I) {
        return current_functional_value;
    }

    // for taking the local derivatives
    const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid->fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<FadFadType> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real> target_soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real>   local_dIdw(max_dofs_per_cell);

    std::vector<real>   local_dIdX(n_metric_dofs_cell);

    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg->fe_collection, dg->face_quadrature_collection,   this->face_update_flags);

    this->allocate_derivatives(actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I);

    dg->solution.update_ghost_values();
    auto metric_cell = dg->high_order_grid->dof_handler_grid.begin_active();
    auto soln_cell = dg->dof_handler.begin_active();

    real local_functional = 0.0;

    for( ; soln_cell != dg->dof_handler.end(); ++soln_cell, ++metric_cell) {
        if(!soln_cell->is_locally_owned()) continue;

        // setting up the volume integration
        const unsigned int i_mapp = 0; // *** ask doug if this will ever be 
        const unsigned int i_fele = soln_cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        (void) i_mapp;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg->fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        soln_cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);
        target_soln_coeff.resize(n_soln_dofs_cell);

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
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            const real val = target_solution[cell_soln_dofs_indices[idof]];
            target_soln_coeff[idof] = val;
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
        FadFadType volume_local_sum;
        volume_local_sum.resizeAndZero(n_total_indep);
        volume_local_sum += evaluate_volume_cell_functional(*physics_fad_fad, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // std::cout << "volume_local_sum.val().val() : " <<  volume_local_sum.val().val() << std::endl;

        // next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = soln_cell->face(iface);
            
            if(face->at_boundary()){

                const unsigned int boundary_id = face->boundary_id();

                volume_local_sum += evaluate_boundary_cell_functional(*physics_fad_fad, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, dg->face_quadrature_collection[i_quad], iface);
            }

        }

        local_functional += volume_local_sum.val().val();
        // now getting the values and adding them to the derivaitve vector

        this->set_derivatives(actually_compute_dIdW, actually_compute_dIdX, actually_compute_d2I, volume_local_sum, cell_soln_dofs_indices, cell_metric_dofs_indices);
    }
    //std::cout << local_functional << std::endl;
    current_functional_value = dealii::Utilities::MPI::sum(local_functional, MPI_COMM_WORLD);
    //std::cout << current_functional_value << std::endl;
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

template <int dim, int nstate, typename real>
dealii::LinearAlgebra::distributed::Vector<real> TargetFunctional<dim,nstate,real>::evaluate_dIdw_finiteDifferences(
    PHiLiP::DGBase<dim,real> &dg, 
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
    std::vector<real> target_soln_coeff(max_dofs_per_cell);
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
        (void) i_mapp;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg.fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);
        target_soln_coeff.resize(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            soln_coeff[idof] = dg.solution[cell_soln_dofs_indices[idof]];
            target_soln_coeff[idof] = target_solution[cell_soln_dofs_indices[idof]];
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
        local_sum_old = this->evaluate_volume_cell_functional(physics, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // Next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){
                const unsigned int boundary_id = face->boundary_id();
                local_sum_old += evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, dg.face_quadrature_collection[i_quad], iface);
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
            local_sum_new = this->evaluate_volume_cell_functional(physics, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

            // Next looping over the faces of the cell checking for boundary elements
            for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
                auto face = cell->face(iface);
                
                if(face->at_boundary()){
                    const unsigned int boundary_id = face->boundary_id();
                    local_sum_new += evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, dg.face_quadrature_collection[i_quad], iface);
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

template <int dim, int nstate, typename real>
dealii::LinearAlgebra::distributed::Vector<real> TargetFunctional<dim,nstate,real>::evaluate_dIdX_finiteDifferences(
    PHiLiP::DGBase<dim,real> &dg, 
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
    const double stepsize)
{
    // for taking the local derivatives
    double local_sum_old;
    double local_sum_new;

    // vector for storing the derivatives with respect to each DOF
    dealii::LinearAlgebra::distributed::Vector<real> dIdX_FD;
    this->allocate_dIdX(dIdX_FD);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
    std::vector<real> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real> target_soln_coeff(max_dofs_per_cell);
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
        const unsigned int i_mapp = 0;
        const unsigned int i_fele = cell->active_fe_index();
        const unsigned int i_quad = i_fele;
        (void) i_mapp;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg.fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);
        target_soln_coeff.resize(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            soln_coeff[idof] = dg.solution[cell_soln_dofs_indices[idof]];
            target_soln_coeff[idof] = target_solution[cell_soln_dofs_indices[idof]];
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
        local_sum_old = this->evaluate_volume_cell_functional(physics, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // Next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = cell->face(iface);
            
            if(face->at_boundary()){
                const unsigned int boundary_id = face->boundary_id();
                local_sum_old += evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, dg.face_quadrature_collection[i_quad], iface);
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
            local_sum_new = this->evaluate_volume_cell_functional(physics, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

            // Next looping over the faces of the cell checking for boundary elements
            for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
                auto face = cell->face(iface);
                
                if(face->at_boundary()){
                    const unsigned int boundary_id = face->boundary_id();
                    local_sum_new += evaluate_boundary_cell_functional(physics, boundary_id, soln_coeff, target_soln_coeff, fe_solution, coords_coeff, fe_metric, dg.face_quadrature_collection[i_quad], iface);
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

template class TargetFunctional <PHILIP_DIM, 1, double>;
template class TargetFunctional <PHILIP_DIM, 2, double>;
template class TargetFunctional <PHILIP_DIM, 3, double>;
template class TargetFunctional <PHILIP_DIM, 4, double>;
template class TargetFunctional <PHILIP_DIM, 5, double>;

} // PHiLiP namespace

