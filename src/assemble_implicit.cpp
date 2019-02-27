#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include <Sacado.hpp>

#include "dg.h"
#include "advection_boundary.h"

namespace PHiLiP
{
    using namespace dealii;

    // For now hard-code advection speed
    template <int dim>
    Tensor<1,dim> velocity_field ()
    {
        Tensor<1,dim> v_field;
        v_field[0] = 1.0;
        if(dim >= 2) v_field[1] = 1.0;
        if(dim >= 3) v_field[2] = 1.0;
        return v_field;
    }

    template <int dim>
    Tensor<1,dim> velocity_field (const Point<dim> &p)
    {
        Tensor<1,dim> v_field;
        //Assert (dim >= 2, ExcNotImplemented());
        //v_field(0) = p(1);
        //v_field(1) = p(0);
        //v_field /= v_field.norm();
        v_field = p;
        return v_field;
    }

    // For now hard-code source term
    template <int dim>
    double evaluate_source_term (const Point<dim> &p)
    {
        double source;
        source = 1.0;
        const double x = p(0), y = p(1), z = p(2);
        if (dim==1) source = cos(x);
        if (dim==2) source = cos(x)*sin(y) + sin(x)*cos(y);
        if (dim==3) source =   cos(x)*sin(y)*sin(z)
                             + sin(x)*cos(y)*sin(z)
                             + sin(x)*sin(y)*cos(z);
        if (dim==1) source = 3.19/dim*cos(3.19/dim*x);
        if (dim==2) source = 3.19/dim*cos(3.19/dim*x)*sin(3.19/dim*y) + 3.19/dim*sin(3.19/dim*x)*cos(3.19/dim*y);
        if (dim==3) source =   3.19/dim*cos(3.19/dim*x)*sin(3.19/dim*y)*sin(3.19/dim*z)
                             + 3.19/dim*sin(3.19/dim*x)*cos(3.19/dim*y)*sin(3.19/dim*z)
                             + 3.19/dim*sin(3.19/dim*x)*sin(3.19/dim*y)*cos(3.19/dim*z);
        return source;
    }

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_cell_terms_implicit(
        const FEValues<dim,dim> *fe_values,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<real> &current_cell_rhs)
    {

        const unsigned int n_quad_pts      = fe_values->n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values->dofs_per_cell;

        AssertDimension (n_dofs_cell, current_dofs_indices.size());

        const std::vector<real> &JxW = fe_values->get_JxW_values ();

        // AD variable
        std::vector<Sacado::Fad::DFad<real>> solution_ad(n_dofs_cell);
        for (unsigned int i = 0; i < n_dofs_cell; ++i) {
            solution_ad[i] = solution(current_dofs_indices[i]);
            solution_ad[i].diff(i, n_dofs_cell);
        }
        std::vector<real> residual_derivatives(n_dofs_cell);

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            Sacado::Fad::DFad<real> rhs = 0;

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                const Tensor<1,dim> vel_at_point = velocity_field<dim>();
                const double source_at_point = evaluate_source_term (fe_values->quadrature_point(iquad));

                const double adv_dot_grad_test = vel_at_point*fe_values->shape_grad(itest, iquad);

                for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                    // Stiffness matrix contibution
                    rhs +=
                        adv_dot_grad_test *
                        //solution(current_dofs_indices[itrial]) *
                        solution_ad[itrial] *
                        fe_values->shape_value(itrial,iquad) *
                        JxW[iquad];
                }
                // Source term contribution
                rhs += 
                    source_at_point *
                    fe_values->shape_value(itest,iquad) *
                    JxW[iquad];
            }

            current_cell_rhs(itest) += rhs.val();

            for (unsigned int itrial = 0; itrial < n_dofs_cell; ++itrial) {
                //residual_derivatives[itrial] = rhs.fastAccessDx(itrial);
                residual_derivatives[itrial] = rhs.dx(itrial);
                //std::cout << current_dofs_indices[itest] << " " << current_dofs_indices[itrial] << " " << residual_derivatives[itrial] << std::endl;
            }
            system_matrix.add(current_dofs_indices[itest], current_dofs_indices, residual_derivatives);
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, double>::assemble_cell_terms_implicit(
        const FEValues<PHILIP_DIM,PHILIP_DIM> *fe_values,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim,real>::assemble_boundary_term_implicit(
        const FEFaceValues<dim,dim> *fe_values_face,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<real> &current_cell_rhs)
    {
        const unsigned int n_quad_pts = fe_values_face->n_quadrature_points;
        const unsigned int n_dofs_cell = fe_values_face->dofs_per_cell;

        AssertDimension (n_dofs_cell, current_dofs_indices.size());

        const std::vector<real> &JxW = fe_values_face->get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals = fe_values_face->get_normal_vectors ();

        // Recover boundary values at quadrature points
        std::vector<real> boundary_values(n_quad_pts);
        static AdvectionBoundary<dim> boundary_function;
        const unsigned int dummy = 0; // Virtual function that requires 3 arguments
        boundary_function.value_list (fe_values_face->get_quadrature_points(), boundary_values, dummy);

        // AD variable
        std::vector<Sacado::Fad::DFad<real>> solution_ad(n_dofs_cell);
        for (unsigned int i = 0; i < n_dofs_cell; ++i) {
            solution_ad[i] = solution(current_dofs_indices[i]);
            solution_ad[i].diff(i, n_dofs_cell);
        }
        std::vector<real> residual_derivatives(n_dofs_cell);

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

            Sacado::Fad::DFad<real> rhs = 0;

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                const real vel_dot_normal = velocity_field<dim> () * normals[iquad];
                const bool inflow = (vel_dot_normal < 0.);

                if (inflow) {
                // Setting the boundary condition when inflow
                    rhs += 
                        -vel_dot_normal *
                        boundary_values[iquad] *
                        fe_values_face->shape_value(itest,iquad) *
                        JxW[iquad];
                } else {
                    // "Numerical flux" at the boundary is the same as the analytical flux
                    for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                        rhs += 
                            -vel_dot_normal *
                            fe_values_face->shape_value(itest,iquad) *
                            fe_values_face->shape_value(itrial,iquad) *
                            solution_ad[itrial] *
                            JxW[iquad];
                    }
                }
            }

            current_cell_rhs(itest) += rhs.val();

            for (unsigned int itrial = 0; itrial < n_dofs_cell; ++itrial) {
                //residual_derivatives[itrial] = rhs.fastAccessDx(itrial);
                residual_derivatives[itrial] = rhs.dx(itrial);
            }
            system_matrix.add(current_dofs_indices[itest], current_dofs_indices, residual_derivatives);

        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM,double>::assemble_boundary_term_implicit(
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM> *fe_values_face,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_face_term_implicit(
        const FEValuesBase<dim,dim>     *fe_values_face_current,
        const FEFaceValues<dim,dim>     *fe_values_face_neighbor,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        const std::vector<types::global_dof_index> &neighbor_dofs_indices,
        Vector<real>          &current_cell_rhs,
        Vector<real>          &neighbor_cell_rhs)
    {
        // Use quadrature points of neighbor cell
        // Might want to use the maximum n_quad_pts1 and n_quad_pts2
        const unsigned int n_quad_pts = fe_values_face_neighbor->n_quadrature_points;

        const unsigned int n_dofs_current_cell = fe_values_face_current->dofs_per_cell;
        const unsigned int n_dofs_neighbor_cell = fe_values_face_neighbor->dofs_per_cell;

        AssertDimension (n_dofs_current_cell, current_dofs_indices.size());
        AssertDimension (n_dofs_neighbor_cell, neighbor_dofs_indices.size());

        // Jacobian and normal should always be consistent between two elements
        // even for non-conforming meshes?
        const std::vector<real> &JxW1 = fe_values_face_current->get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals1 = fe_values_face_current->get_normal_vectors ();

        // AD variable
        std::vector<Sacado::Fad::DFad<real>> current_solution_ad(n_dofs_current_cell);
        std::vector<Sacado::Fad::DFad<real>> neighbor_solution_ad(n_dofs_current_cell);

        const unsigned int total_indep = n_dofs_neighbor_cell + n_dofs_neighbor_cell;

        for (unsigned int i = 0; i < n_dofs_current_cell; ++i) {
            current_solution_ad[i] = solution(current_dofs_indices[i]);
            current_solution_ad[i].diff(i, total_indep);
        }
        for (unsigned int i = 0; i < n_dofs_neighbor_cell; ++i) {
            neighbor_solution_ad[i] = solution(neighbor_dofs_indices[i]);
            neighbor_solution_ad[i].diff(i+n_dofs_current_cell, total_indep);
        }
        std::vector<real> dR1_dW1(n_dofs_current_cell);
        std::vector<real> dR1_dW2(n_dofs_neighbor_cell);
        std::vector<real> dR2_dW1(n_dofs_current_cell);
        std::vector<real> dR2_dW2(n_dofs_neighbor_cell);

        std::vector<Sacado::Fad::DFad<real>> normal1_numerical_flux(n_quad_pts);

        std::vector<Sacado::Fad::DFad<real>> analytical_flux1(dim);
        std::vector<Sacado::Fad::DFad<real>> analytical_flux2(dim);
        std::vector<Sacado::Fad::DFad<real>> numerical_flux(dim);
        std::vector<Sacado::Fad::DFad<real>> n1(dim);


        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            normal1_numerical_flux[iquad] = 0;
            Sacado::Fad::DFad<real> w1 = 0;
            Sacado::Fad::DFad<real> w2 = 0;
            for (unsigned int d=0; d<dim; ++d) {
                analytical_flux1[d] = 0;
                analytical_flux2[d] = 0;
            }
            Sacado::Fad::DFad<real> lambda = 0;

            const Tensor<1,dim,real> velocity_at_q = velocity_field<dim>();

            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                w1 += current_solution_ad[itrial] * fe_values_face_current->shape_value(itrial, iquad);
            }
            for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                w2 += current_solution_ad[itrial] * fe_values_face_neighbor->shape_value(itrial, iquad);
                w2 += solution(neighbor_dofs_indices[itrial]) * fe_values_face_neighbor->shape_value(itrial, iquad);
            }

            for (unsigned int d=0; d<dim; ++d) {
                const real vel = velocity_at_q[d];
                const real n1 = normals1[iquad][d];
                analytical_flux1[d] += vel*w1;
                analytical_flux2[d] += vel*w2;
                lambda += vel * n1;
            }

            for (unsigned int d=0; d<dim; ++d) {
                const real n1 = normals1[iquad][d];
                Sacado::Fad::DFad<real> numerical_flux = (analytical_flux1[d] + analytical_flux2[d] + (n1*w1 - n1*w2) * lambda) * 0.5;
                normal1_numerical_flux[iquad] += numerical_flux*n1;
            }

        }

        for (unsigned int itest=0; itest<n_dofs_current_cell; ++itest) {
            Sacado::Fad::DFad<real> rhs = 0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs -=
                    fe_values_face_current->shape_value(itest,iquad) *
                    normal1_numerical_flux[iquad] *
                    JxW1[iquad];
            }
            current_cell_rhs(itest) += rhs.val();
            for (unsigned int itrial = 0; itrial < n_dofs_current_cell; ++itrial) {
                dR1_dW1[itrial] = rhs.dx(itrial);
            }
            for (unsigned int itrial = 0; itrial < n_dofs_neighbor_cell; ++itrial) {
                dR1_dW2[itrial] = rhs.dx(n_dofs_current_cell+itrial);
            }
            system_matrix.add(current_dofs_indices[itest], current_dofs_indices, dR1_dW1);
            system_matrix.add(current_dofs_indices[itest], neighbor_dofs_indices, dR1_dW2);
        }
        for (unsigned int itest=0; itest<n_dofs_neighbor_cell; ++itest) {
            Sacado::Fad::DFad<real> rhs = 0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs -=
                    fe_values_face_neighbor->shape_value(itest,iquad) *
                    (-normal1_numerical_flux[iquad]) *
                    JxW1[iquad];
            }
            neighbor_cell_rhs(itest) += rhs.val();
            for (unsigned int itrial = 0; itrial < n_dofs_current_cell; ++itrial) {
                dR2_dW1[itrial] = rhs.dx(itrial);
            }
            for (unsigned int itrial = 0; itrial < n_dofs_neighbor_cell; ++itrial) {
                dR2_dW2[itrial] = rhs.dx(n_dofs_current_cell+itrial);
            }
            system_matrix.add(neighbor_dofs_indices[itest], current_dofs_indices, dR2_dW1);
            system_matrix.add(neighbor_dofs_indices[itest], neighbor_dofs_indices, dR2_dW2);
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, double>::assemble_face_term_implicit(
        const FEValuesBase<PHILIP_DIM,PHILIP_DIM>     *fe_values_face_current,
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM>     *fe_values_face_neighbor,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        const std::vector<types::global_dof_index> &neighbor_dofs_indices,
        Vector<double>          &current_cell_rhs,
        Vector<double>          &neighbor_cell_rhs);
} // end of PHiLiP namespace

