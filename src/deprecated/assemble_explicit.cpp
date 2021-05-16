#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include "dg.h"
#include "boundary.h"

#include "manufactured_solution.h"
namespace PHiLiP {
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
    //template <int dim>
    //double evaluate_source_term (const Point<dim> &p)
    //{
    //    double source;
    //    source = 1.0;
    //    if (dim==1) {
    //        const double x = p(0);
    //        source = 3.19/dim*cos(3.19/dim*x);
    //    } else if (dim==2) {
    //        const double x = p(0), y = p(1);
    //        source = 3.19/dim*cos(3.19/dim*x)*sin(3.19/dim*y) + 3.19/dim*sin(3.19/dim*x)*cos(3.19/dim*y);
    //    } else if (dim==3) {
    //        const double x = p(0), y = p(1), z = p(2);
    //        source =   3.19/dim*cos(3.19/dim*x)*sin(3.19/dim*y)*sin(3.19/dim*z)
    //                         + 3.19/dim*sin(3.19/dim*x)*cos(3.19/dim*y)*sin(3.19/dim*z)
    //                         + 3.19/dim*sin(3.19/dim*x)*sin(3.19/dim*y)*cos(3.19/dim*z);
    //    }
    //    return source;
    //}

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_cell_terms_explicit(
        const FEValues<dim,dim> *fe_values,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<real> &current_cell_rhs)
    {
        const unsigned int n_quad_pts      = fe_values->n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values->dofs_per_cell;

        AssertDimension (n_dofs_cell, current_dofs_indices.size());

        const std::vector<real> &JxW = fe_values->get_JxW_values ();

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const Tensor<1,dim> vel_at_point = velocity_field<dim>();
            //const double source_at_point = evaluate_source_term (fe_values->quadrature_point(iquad));
            const double source_at_point = manufactured_convection_diffusion_source (fe_values->quadrature_point(iquad));

            for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

                const double adv_dot_grad_test = vel_at_point*fe_values->shape_grad(itest, iquad);

                for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                    // Stiffness matrix contibution
                    current_cell_rhs(itest) += 
                        adv_dot_grad_test *
                        solution(current_dofs_indices[itrial]) *
                        fe_values->shape_value(itrial,iquad) *
                        JxW[iquad];
                }
                // Source term contribution
                current_cell_rhs(itest) += 
                    source_at_point *
                    fe_values->shape_value(itest,iquad) *
                    JxW[iquad];
            }
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, double>::assemble_cell_terms_explicit(
        const FEValues<PHILIP_DIM,PHILIP_DIM> *fe_values,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim,real>::assemble_boundary_term_explicit(
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
        static Boundary<dim> boundary_function;
        const unsigned int dummy = 0; // Virtual function that requires 3 arguments
        boundary_function.value_list (fe_values_face->get_quadrature_points(), boundary_values, dummy);


        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const real vel_dot_normal = velocity_field<dim> () * normals[iquad];
            const bool inflow = (vel_dot_normal < 0.);

            if (inflow) {
                // Setting the boundary condition when inflow
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    current_cell_rhs(itest) += 
                        -vel_dot_normal *
                        boundary_values[iquad] *
                        fe_values_face->shape_value(itest,iquad) *
                        JxW[iquad];
                }
            } else {
                // "Numerical flux" at the boundary is the same as the analytical flux
                for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
                    for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                        current_cell_rhs(itest) += 
                            -vel_dot_normal *
                            fe_values_face->shape_value(itest,iquad) *
                            fe_values_face->shape_value(itrial,iquad) *
                            solution(current_dofs_indices[itrial]) *
                            JxW[iquad];
                    }
                }
            }
         }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM,double>::assemble_boundary_term_explicit(
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM> *fe_values_face,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_face_term_explicit(
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

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const Tensor<1,dim,real> velocity_at_q = velocity_field<dim>();

            real w1 = 0;
            real w2 = 0;
            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                w1 += solution(current_dofs_indices[itrial]) * fe_values_face_current->shape_value(itrial, iquad);
            }
            for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                w2 += solution(neighbor_dofs_indices[itrial]) * fe_values_face_neighbor->shape_value(itrial, iquad);
            }
            const Tensor<1,dim,real> analytical_flux1 = velocity_at_q*w1;
            const Tensor<1,dim,real> analytical_flux2 = velocity_at_q*w2;
            const Tensor<1,dim,real> n1 = normals1[iquad];
                
            const real lambda = velocity_at_q * n1;

            const Tensor<1,dim> numerical_flux =
                (analytical_flux1 + analytical_flux2 + (n1*w1 - n1*w2) * lambda) * 0.5;

            const real normal1_numerical_flux = numerical_flux*n1;

            for (unsigned int itest=0; itest<n_dofs_current_cell; ++itest) {
                current_cell_rhs(itest) -=
                    fe_values_face_current->shape_value(itest,iquad) *
                    normal1_numerical_flux *
                    JxW1[iquad];
            }
            for (unsigned int itest=0; itest<n_dofs_neighbor_cell; ++itest) {
                neighbor_cell_rhs(itest) -=
                    fe_values_face_neighbor->shape_value(itest,iquad) *
                    (-normal1_numerical_flux) *
                    JxW1[iquad];
            }
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, double>::assemble_face_term_explicit(
        const FEValuesBase<PHILIP_DIM,PHILIP_DIM>     *fe_values_face_current,
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM>     *fe_values_face_neighbor,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        const std::vector<types::global_dof_index> &neighbor_dofs_indices,
        Vector<double>          &current_cell_rhs,
        Vector<double>          &neighbor_cell_rhs);
} // end of PHiLiP namespace
