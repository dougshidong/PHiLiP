// includes
#include <vector>

#include <Sacado.hpp>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_tools.h>

//#include "dg/high_order_grid.h"
#include "physics/physics.h"
#include "dg/dg.h"
#include "dg/high_order_grid.h"
#include "functional.h"

namespace PHiLiP {

template <int dim, int nstate, typename real>
Functional<dim,nstate,real>::Functional(
    std::shared_ptr<DGBase<dim,real>> _dg,
    const bool _uses_solution_values,
    const bool _uses_solution_gradient)
    : dg(_dg)
    , uses_solution_values(_uses_solution_values)
    , uses_solution_gradient(_uses_solution_gradient)
{ }

// template <int dim, int nstate, typename real>
// void Functional<dim, nstate, real>::evaluate_function(const Physics::PhysicsBase<dim,nstate,real> &physics)
// {
//     real local_sum = 0;
// 
//     // allocating vectors for local calculations
//     // could these also be indexSets?
//     const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
//     std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
//     std::vector<real> soln_coeff(max_dofs_per_cell);
// 
//     const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
//     dealii::hp::MappingCollection<dim> mapping_collection(mapping);
// 
//     dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
//     dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);
// 
//     dg.solution.update_ghost_values();
//     for(auto cell = dg.dof_handler.begin_active(); cell != dg.dof_handler.end(); ++cell){
//         if(!cell->is_locally_owned()) continue;
// 
//         // setting up the volume integration
//         const unsigned int i_mapp = 0; // *** ask doug if this will ever be 
//         const unsigned int i_fele = cell->active_fe_index();
//         const unsigned int i_quad = i_fele;
//         const dealii::FESystem<dim,dim> &current_fe_ref = dg.fe_collection[i_fele];
//         const unsigned int n_soln_dofs_cell = current_fe_ref.n_dofs_per_cell();
//         
//         // reinitialize the volume integration
//         fe_values_collection_volume.reinit(cell, i_quad, i_mapp, i_fele);
//         const dealii::FEValues<dim,dim> &fe_values_volume = fe_values_collection_volume.get_present_fe_values();
// 
//         // // number of quadrature points
//         // const unsigned int n_quad_points = fe_values_volume.n_quadrature_points;
// 
//         // getting the indices
//         cell_soln_dofs_indices.resize(n_soln_dofs_cell);
//         cell->get_dof_indices(cell_soln_dofs_indices);
// 
//         // getting solution values
//         soln_coeff.resize(n_soln_dofs_cell);
// 
//         // adding the contribution from the current volume, also need to pass the solution vector on these points
//         for(unsigned int iquad=0; iquad<n_quad_pts; ++iquad){
//             for(unsigned int idof=0; idof<n_soln_dofs_cell; ++idof){
//                 std::array<real,nstate> soln_at_q;
//                 std::array< dealii::Tensor<1,dim,real>, nstate > soln_grad_at_q;
//                 std::fill(soln_at_q.begin(), soln_at_q.end(), 0);
//                 for (unsigned int idof=0; idof<fe_values_volume.dofs_per_cell; ++idof) {
//                     const unsigned int istate = fe_values_volume.get_fe().system_to_component_index(idof).first;
//                     soln_at_q[istate] += soln_coeff[idof] * fe_values_volume.shape_value_component(idof, iquad, istate);
//                     soln_grad_at_q[istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
//                 }
//                 soln_coeff[idof] = dg.solution[cell_soln_dofs_indices[idof]];
//             }
// 
//             real volume_integrand = evaluate_volume_integrand(physics, fe_values_volume.point(iquad), soln_at_q, soln_grad_at_q);
//             local_sum += volume_integrand * fe_values_volume.JxW(iquad);
//         }
// 
//         // next looping over the faces of the cell checking for boundary elements
//         for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
//             auto face = cell->face(iface);
//             
//             if(face->at_boundary()){
//                 fe_values_collection_face.reinit(cell, iface, i_quad, i_mapp, i_fele);
//                 const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();
// 
//                 const unsigned int boundary_id = face->boundary_id();
// 
//                 local_sum += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
//             }
// 
//         }
//     }
// 
//     return dealii::Utilities::MPI::sum(local_sum, MPI_COMM_WORLD);
// }

template <int dim, int nstate, typename real>
template <typename real2>
real2 Functional<dim, nstate, real>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real2> &physics,
    const std::vector< real2 > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real2 > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature)
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
        if (uses_solution_gradient) jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian));

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

template <int dim, int nstate, typename real>
real Functional<dim, nstate, real>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,real> &physics,
    const std::vector< real > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< real > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature)
{
    return evaluate_volume_cell_functional<real>(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
}
template <int dim, int nstate, typename real>
Sacado::Fad::DFad<real> Functional<dim, nstate, real>::evaluate_volume_cell_functional(
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics,
    const std::vector< Sacado::Fad::DFad<real> > &soln_coeff,
    const dealii::FESystem<dim> &fe_solution,
    const std::vector< Sacado::Fad::DFad<real> > &coords_coeff,
    const dealii::FESystem<dim> &fe_metric,
    const dealii::Quadrature<dim> &volume_quadrature)
{
    return evaluate_volume_cell_functional<Sacado::Fad::DFad<real>>(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);
}

template <int dim, int nstate, typename real>
real Functional<dim, nstate, real>::evaluate_functional(
    const Physics::PhysicsBase<dim,nstate,Sacado::Fad::DFad<real>> &physics,
    const bool compute_dIdW,
    const bool compute_dIdX)
{
    // for the AD'd return variable
    using ADType = Sacado::Fad::DFad<real>;

    // Returned value
    real local_functional = 0.0;

    // for taking the local derivatives
    const dealii::FESystem<dim,dim> &fe_metric = dg->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<ADType> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real>   local_dIdw(max_dofs_per_cell);

    std::vector<real>   local_dIdX(n_metric_dofs_cell);

    const auto mapping = (*(dg->high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg->fe_collection, dg->face_quadrature_collection,   this->face_update_flags);

    if (compute_dIdW) {
        // allocating the vector
        dealii::IndexSet locally_owned_dofs = dg->dof_handler.locally_owned_dofs();
        dIdw.reinit(locally_owned_dofs, MPI_COMM_WORLD);
    }
    if (compute_dIdX) {
        // allocating the vector
        dealii::IndexSet locally_owned_dofs = dg->high_order_grid.dof_handler_grid.locally_owned_dofs();
        dealii::IndexSet locally_relevant_dofs, ghost_dofs;
        dealii::DoFTools::extract_locally_relevant_dofs(dg->high_order_grid.dof_handler_grid, locally_relevant_dofs);
        ghost_dofs = locally_relevant_dofs;
        ghost_dofs.subtract_set(locally_owned_dofs);
        dIdX.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);
    }

    dg->solution.update_ghost_values();
    auto metric_cell = dg->high_order_grid.dof_handler_grid.begin_active();
    auto soln_cell = dg->dof_handler.begin_active();
    for( ; soln_cell != dg->dof_handler.end(); ++soln_cell, ++metric_cell) {
        if(!soln_cell->is_locally_owned()) continue;

        // setting up the volume integration
        const unsigned int i_mapp = 0; // *** ask doug if this will ever be 
        const unsigned int i_fele = soln_cell->active_fe_index();
        const unsigned int i_quad = i_fele;

        // Get solution coefficients
        const dealii::FESystem<dim,dim> &fe_solution = dg->fe_collection[i_fele];
        const unsigned int n_soln_dofs_cell = fe_solution.n_dofs_per_cell();
        cell_soln_dofs_indices.resize(n_soln_dofs_cell);
        soln_cell->get_dof_indices(cell_soln_dofs_indices);
        soln_coeff.resize(n_soln_dofs_cell);
        for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
            soln_coeff[idof] = dg->solution[cell_soln_dofs_indices[idof]];
        }

        // Get metric coefficients
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector< ADType > coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            coords_coeff[idof] = dg->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
        }

        // Setup automatic differentiation
        unsigned int n_total_indep = 0;
        if (compute_dIdW) n_total_indep += n_soln_dofs_cell;
        if (compute_dIdX) n_total_indep += n_metric_dofs_cell;
        unsigned int iderivative = 0;
        if (compute_dIdW) {
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof) {
                soln_coeff[idof].diff(iderivative++, n_total_indep);
            }
        }
        if (compute_dIdX) {
            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                coords_coeff[idof].diff(iderivative++, n_total_indep);
            }
        }
        AssertDimension(iderivative, n_total_indep);

        // Get quadrature point on reference cell
        const dealii::Quadrature<dim> &volume_quadrature = dg->volume_quadrature_collection[i_quad];

        // Evaluate integral on the cell volume
        ADType volume_local_sum = evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

        // next looping over the faces of the cell checking for boundary elements
        for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
            auto face = soln_cell->face(iface);
            
            if(face->at_boundary()){
                fe_values_collection_face.reinit(soln_cell, iface, i_quad, i_mapp, i_fele);
                const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                const unsigned int boundary_id = face->boundary_id();

                volume_local_sum += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
            }

        }

        local_functional += volume_local_sum.val();
        // now getting the values and adding them to the derivaitve vector

        iderivative = 0;
        if (compute_dIdW) {
            local_dIdw.resize(n_soln_dofs_cell);
            for(unsigned int idof = 0; idof < n_soln_dofs_cell; ++idof){
                local_dIdw[idof] = volume_local_sum.dx(iderivative++);
            }
            dIdw.add(cell_soln_dofs_indices, local_dIdw);
        }
        if (compute_dIdX) {
            local_dIdX.resize(n_metric_dofs_cell);
            for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof){
                local_dIdX[idof] = volume_local_sum.dx(iderivative++);
            }
            dIdX.add(cell_metric_dofs_indices, local_dIdX);
        }
        AssertDimension(iderivative, n_total_indep);
    }
    dealii::Utilities::MPI::sum(local_functional, MPI_COMM_WORLD);
    // compress before the return
    if (compute_dIdW) dIdw.compress(dealii::VectorOperation::add);
    if (compute_dIdX) dIdX.compress(dealii::VectorOperation::add);

    return local_functional;
}

template <int dim, int nstate, typename real>
dealii::LinearAlgebra::distributed::Vector<real> Functional<dim,nstate,real>::evaluate_dIdw_finiteDifferences(
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
    std::vector<real> local_dIdw(max_dofs_per_cell);

    const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

    dg.solution.update_ghost_values();
    auto metric_cell = dg.high_order_grid.dof_handler_grid.begin_active();
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
        const dealii::FESystem<dim,dim> &fe_metric = dg.high_order_grid.fe_system;
        const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector<real> coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            coords_coeff[idof] = dg.high_order_grid.nodes[cell_metric_dofs_indices[idof]];
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
                const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                const unsigned int boundary_id = face->boundary_id();

                local_sum_old += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
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
                    const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                    const unsigned int boundary_id = face->boundary_id();

                    local_sum_new += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
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
dealii::LinearAlgebra::distributed::Vector<real> Functional<dim,nstate,real>::evaluate_dIdX_finiteDifferences(
    PHiLiP::DGBase<dim,real> &dg, 
    const PHiLiP::Physics::PhysicsBase<dim,nstate,real> &physics,
    const double stepsize)
{
    // for taking the local derivatives
    double local_sum_old;
    double local_sum_new;

    // vector for storing the derivatives with respect to each DOF
    dealii::LinearAlgebra::distributed::Vector<real> dIdX_FD;

    // allocating the vector
    dealii::IndexSet locally_owned_dofs = dg.high_order_grid.dof_handler_grid.locally_owned_dofs();
    dealii::IndexSet locally_relevant_dofs, ghost_dofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dg.high_order_grid.dof_handler_grid, locally_relevant_dofs);
    ghost_dofs = locally_relevant_dofs;
    ghost_dofs.subtract_set(locally_owned_dofs);
    dIdX_FD.reinit(locally_owned_dofs, ghost_dofs, MPI_COMM_WORLD);

    // setup it mostly the same as evaluating the value (with exception that local solution is also AD)
    const unsigned int max_dofs_per_cell = dg.dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> cell_soln_dofs_indices(max_dofs_per_cell);
    std::vector<dealii::types::global_dof_index> neighbor_dofs_indices(max_dofs_per_cell);
    std::vector<real> soln_coeff(max_dofs_per_cell); // for obtaining the local derivatives (to be copied back afterwards)
    std::vector<real> local_dIdX(max_dofs_per_cell);

    const auto mapping = (*(dg.high_order_grid.mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);

    dealii::hp::FEValues<dim,dim>     fe_values_collection_volume(mapping_collection, dg.fe_collection, dg.volume_quadrature_collection, this->volume_update_flags);
    dealii::hp::FEFaceValues<dim,dim> fe_values_collection_face  (mapping_collection, dg.fe_collection, dg.face_quadrature_collection,   this->face_update_flags);

    dg.solution.update_ghost_values();
    auto metric_cell = dg.high_order_grid.dof_handler_grid.begin_active();
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
        const dealii::FESystem<dim,dim> &fe_metric = dg.high_order_grid.fe_system;
        const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> cell_metric_dofs_indices(n_metric_dofs_cell);
        metric_cell->get_dof_indices (cell_metric_dofs_indices);
        std::vector<real> coords_coeff(n_metric_dofs_cell);
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            coords_coeff[idof] = dg.high_order_grid.nodes[cell_metric_dofs_indices[idof]];
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
                const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                const unsigned int boundary_id = face->boundary_id();

                local_sum_old += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
            }

        }

        // now looping over all the DOFs in this cell and taking the FD
        local_dIdX.resize(n_metric_dofs_cell);
        for(unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof){
            // for each dof copying the solution
            for(unsigned int idof2 = 0; idof2 < n_metric_dofs_cell; ++idof2){
                coords_coeff[idof2] = dg.high_order_grid.nodes[cell_metric_dofs_indices[idof2]];
            }
            coords_coeff[idof] += stepsize;

            // then peturb the idof'th value
            local_sum_new = this->evaluate_volume_cell_functional(physics, soln_coeff, fe_solution, coords_coeff, fe_metric, volume_quadrature);

            // Next looping over the faces of the cell checking for boundary elements
            for(unsigned int iface = 0; iface < dealii::GeometryInfo<dim>::faces_per_cell; ++iface){
                auto face = cell->face(iface);
                
                if(face->at_boundary()){
                    fe_values_collection_face.reinit(cell, iface, i_quad, i_mapp, i_fele);
                    const dealii::FEFaceValues<dim,dim> &fe_values_face = fe_values_collection_face.get_present_fe_values();

                    const unsigned int boundary_id = face->boundary_id();

                    local_sum_new += this->evaluate_cell_boundary(physics, boundary_id, fe_values_face, soln_coeff);
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



template class Functional <PHILIP_DIM, 1, double>;
template class Functional <PHILIP_DIM, 2, double>;
template class Functional <PHILIP_DIM, 3, double>;
template class Functional <PHILIP_DIM, 4, double>;
template class Functional <PHILIP_DIM, 5, double>;

} // PHiLiP namespace
