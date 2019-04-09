#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include <Sacado.hpp>

#include "dg.h"
#include "advection_boundary.h"

#include "manufactured_solution.h"
namespace PHiLiP
{
    using namespace dealii;

    // For now hard-code advection speed
    template <int dim>
    Tensor<1,dim> velocity_field ()
    {
        Tensor<1,dim> v_field;
        v_field[0] = 1.0;//0.5;//1.0;
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

        const unsigned int n_components = fe_values->get_fe().n_components();

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            Sacado::Fad::DFad<real> rhs = 0;

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                const Tensor<1,dim> vel_at_point = velocity_field<dim>();
                //const double source_at_point = manufactured_advection_source (fe_values->quadrature_point(iquad));
                const double source_at_point = manufactured_convection_diffusion_source (fe_values->quadrature_point(iquad));

                const double adv_dot_grad_test = vel_at_point*fe_values->shape_grad(itest, iquad);

                const double dx = JxW[iquad];

                for (unsigned int itrial=0; itrial<n_dofs_cell; ++itrial) {
                    // Convection
                    rhs +=
                        adv_dot_grad_test *
                        solution_ad[itrial] *
                        fe_values->shape_value(itrial,iquad) *
                        dx;


                    // Diffusion term
                    for (unsigned int idim=0; idim<dim; ++idim)
                        rhs -= 
                            solution_ad[itrial] *
                            (fe_values->shape_grad(itrial,iquad)[idim] *
                            fe_values->shape_grad(itest,iquad)[idim]) *
                            dx;
                }
                // Source term contribution
                rhs += 
                    source_at_point *
                    fe_values->shape_value(itest,iquad) *
                    dx;
            }

            current_cell_rhs(itest) += rhs.val();

            for (unsigned int itrial = 0; itrial < n_dofs_cell; ++itrial) {
                //residual_derivatives[itrial] = rhs.fastAccessDx(itrial);
                residual_derivatives[itrial] = rhs.dx(itrial);
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
        const real penalty,
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

            // Convection
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

            // Diffusion
            // *******************
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                Sacado::Fad::DFad<real> soln_int = 0;
                Sacado::Fad::DFad<real> soln_ext = 0;

                std::vector<Sacado::Fad::DFad<real>> soln_grad_int(dim);
                std::vector<Sacado::Fad::DFad<real>> soln_grad_ext(dim);
                // Diffusion
                // Set u1, soln_ext, u1_grad, soln_grad_ext
                for (unsigned int itrial=0; itrial<n_dofs_cell; itrial++) {
                    soln_int += solution_ad[itrial] * fe_values_face->shape_value(itrial, iquad);
                }

                soln_ext = -soln_int+2*boundary_values[iquad];
                soln_ext = boundary_values[iquad];

                for (unsigned int idim=0; idim<dim; ++idim) {
                    soln_grad_int[idim] = 0;
                    soln_grad_ext[idim] = 0;
                    for (unsigned int itrial=0; itrial<n_dofs_cell; itrial++) {
                        soln_grad_int[idim] += solution_ad[itrial] * fe_values_face->shape_grad(itrial,iquad)[idim];
                    }
                    soln_grad_ext[idim] = soln_grad_int[idim];
                }

                // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //       and zero everywhere else.
                const real test_int = fe_values_face->shape_value(itest, iquad);
                const real test_ext = 0;

                for (unsigned int idim=0; idim<dim; ++idim) {

                    const real normal1 = normals[iquad][idim];
                    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                    //       and zero everywhere else.
                    const real test_grad_int = fe_values_face->shape_grad(itest, iquad)[idim];
                    const real test_grad_ext = 0;


                    const Sacado::Fad::DFad<real> soln_jump      = (soln_int - soln_ext) * normal1;
                    const Sacado::Fad::DFad<real> soln_grad_avg  = 0.5*(soln_grad_int[idim] + soln_grad_ext[idim]);
                    const real test_jump                         = (test_int - test_ext) * normal1;
                    const real test_grad_avg                     = 0.5*(test_grad_int + test_grad_ext);

                    //const Sacado::Fad::DFad<real> soln_jump      = (boundary_values[iquad]) * normal1;
                    //const Sacado::Fad::DFad<real> soln_grad_avg  = soln_grad_int[idim];

                    //const real test_jump  = test_int * normal1;
                    //const real test_grad_avg  = test_grad_int;

                    //rhs += (test_grad_int*normal1*boundary_values[iquad]-penalty*test_int*boundary_values[iquad]) * JxW[iquad];

                    rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW[iquad];
                }
            }
            // *******************

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
        const double penalty,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);

    template <int dim, typename real>
    void DiscontinuousGalerkin<dim, real>::assemble_face_term_implicit(
        const FEValuesBase<dim,dim>     *fe_values_face_current,
        const FEFaceValues<dim,dim>     *fe_values_face_neighbor,
        const real penalty,
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
        const std::vector<real> &JxW_int = fe_values_face_current->get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals_int = fe_values_face_current->get_normal_vectors ();

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

        std::vector<Sacado::Fad::DFad<real>> normal_int_numerical_flux(n_quad_pts);

        std::vector<Sacado::Fad::DFad<real>> analytical_flux_int(dim);
        std::vector<Sacado::Fad::DFad<real>> analytical_flux_ext(dim);
        std::vector<Sacado::Fad::DFad<real>> numerical_flux(dim);
        std::vector<Sacado::Fad::DFad<real>> normal1(dim);



        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            normal_int_numerical_flux[iquad] = 0;
            Sacado::Fad::DFad<real> soln_int = 0;
            Sacado::Fad::DFad<real> soln_ext = 0;

            std::vector<Sacado::Fad::DFad<real>> soln_grad_int(dim);
            std::vector<Sacado::Fad::DFad<real>> soln_grad_ext(dim);

            for (unsigned int idim=0; idim<dim; ++idim) {
                analytical_flux_int[idim] = 0;
                analytical_flux_ext[idim] = 0;

                soln_grad_int[idim] = 0;
                soln_grad_ext[idim] = 0;
            }
            Sacado::Fad::DFad<real> lambda = 0;

            const Tensor<1,dim,real> velocity_at_q = velocity_field<dim>();

            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                soln_int += current_solution_ad[itrial] * fe_values_face_current->shape_value(itrial, iquad);
                for (unsigned int idim=0; idim<dim; ++idim) {
                    soln_grad_int[idim] += current_solution_ad[itrial] * fe_values_face_current->shape_grad(itrial,iquad)[idim];
                }
            }
            for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                soln_ext += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_value(itrial, iquad);
                for (unsigned int idim=0; idim<dim; ++idim) {
                    soln_grad_ext[idim] += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_grad(itrial,iquad)[idim];
                }
            }

            // Evaluate analytical flux and scalar parameter for scalar dissipation
            for (unsigned int idim=0; idim<dim; ++idim) {
                const real vel = velocity_at_q[idim];
                const real normal1 = normals_int[iquad][idim];
                analytical_flux_int[idim] += vel*soln_int;
                analytical_flux_ext[idim] += vel*soln_ext;
                lambda += vel * normal1;
            }

            for (unsigned int idim=0; idim<dim; ++idim) {
                const real normal1 = normals_int[iquad][idim];
                Sacado::Fad::DFad<real> numerical_flux = (analytical_flux_int[idim] + analytical_flux_ext[idim] + (normal1*soln_int - normal1*soln_ext) * lambda) * 0.5;
                normal_int_numerical_flux[iquad] += numerical_flux*normal1;
            }


        }

        for (unsigned int itest_current=0; itest_current<n_dofs_current_cell; ++itest_current) {
            // From test functions associated with current cell point of view
            // *******************
            Sacado::Fad::DFad<real> rhs = 0;

            // Convection
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs -=
                    fe_values_face_current->shape_value(itest_current,iquad) *
                    normal_int_numerical_flux[iquad] *
                    JxW_int[iquad];
            }
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                Sacado::Fad::DFad<real> soln_int = 0;
                Sacado::Fad::DFad<real> soln_ext = 0;

                std::vector<Sacado::Fad::DFad<real>> soln_grad_int(dim);
                std::vector<Sacado::Fad::DFad<real>> soln_grad_ext(dim);
                // Diffusion
                for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                    soln_int += current_solution_ad[itrial] * fe_values_face_current->shape_value(itrial, iquad);
                }
                for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                    soln_ext += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_value(itrial, iquad);
                }

                // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //       and zero everywhere else.
                const real test_int = fe_values_face_current->shape_value(itest_current, iquad);
                const real test_ext = 0;

                for (unsigned int idim=0; idim<dim; ++idim) {
                    soln_grad_int[idim] = 0;
                    soln_grad_ext[idim] = 0;
                    for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                        soln_grad_int[idim] += current_solution_ad[itrial] * fe_values_face_current->shape_grad(itrial,iquad)[idim];
                    }
                    for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                        soln_grad_ext[idim] += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_grad(itrial,iquad)[idim];
                    }

                    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                    //       and zero everywhere else.
                    const real test_grad_int = fe_values_face_neighbor->shape_grad(itest_current, iquad)[idim];
                    const real test_grad_ext = 0;

                    const real normal1 = normals_int[iquad][idim];
                    const Sacado::Fad::DFad<real> soln_jump      = (soln_int - soln_ext) * normal1;
                    const Sacado::Fad::DFad<real> soln_grad_avg  = 0.5*(soln_grad_int[idim] + soln_grad_ext[idim]);
                    const real test_jump                         = (test_int - test_ext) * normal1;
                    const real test_grad_avg                     = 0.5*(test_grad_int + test_grad_ext);

                    rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];

                }
            }
            // *******************

            current_cell_rhs(itest_current) += rhs.val();
            for (unsigned int itrial = 0; itrial < n_dofs_current_cell; ++itrial) {
                dR1_dW1[itrial] = rhs.dx(itrial);
            }
            for (unsigned int itrial = 0; itrial < n_dofs_neighbor_cell; ++itrial) {
                dR1_dW2[itrial] = rhs.dx(n_dofs_current_cell+itrial);
            }
            system_matrix.add(current_dofs_indices[itest_current], current_dofs_indices, dR1_dW1);
            system_matrix.add(current_dofs_indices[itest_current], neighbor_dofs_indices, dR1_dW2);
        }

        for (unsigned int itest_neighbor=0; itest_neighbor<n_dofs_neighbor_cell; ++itest_neighbor) {
            // From test functions associated with neighbour cell point of view
            // *******************
            Sacado::Fad::DFad<real> rhs = 0;
            // Convection
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                rhs -=
                    fe_values_face_neighbor->shape_value(itest_neighbor,iquad) *
                    (-normal_int_numerical_flux[iquad]) *
                    JxW_int[iquad];
            }

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                Sacado::Fad::DFad<real> soln_int = 0;
                Sacado::Fad::DFad<real> soln_ext = 0;
                std::vector<Sacado::Fad::DFad<real>> soln_grad_int(dim);
                std::vector<Sacado::Fad::DFad<real>> soln_grad_ext(dim);

                // u and v on given quadrature point on both sides of the face
                for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                    soln_int += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_value(itrial, iquad);
                }
                for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                    soln_ext += current_solution_ad[itrial] * fe_values_face_current->shape_value(itrial, iquad);
                }
                // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //       and zero everywhere else.
                const real test_int = fe_values_face_neighbor->shape_value(itest_neighbor, iquad);
                const real test_ext = 0;

                for (unsigned int idim=0; idim<dim; ++idim) {
                    soln_grad_int[idim] = 0;
                    soln_grad_ext[idim] = 0;
                    for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                        soln_grad_int[idim] += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_grad(itrial,iquad)[idim];
                    }
                    for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                        soln_grad_ext[idim] += current_solution_ad[itrial] * fe_values_face_current->shape_grad(itrial,iquad)[idim];
                    }
                    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                    //       and zero everywhere else.
                    const real test_grad_int = fe_values_face_neighbor->shape_grad(itest_neighbor, iquad)[idim];
                    const real test_grad_ext = 0;

                    // Using normals from other side
                    // So opposite and equal value
                    const real flipped_normal1 = -normals_int[iquad][idim];
                    const Sacado::Fad::DFad<real> soln_jump      = (soln_int - soln_ext) * flipped_normal1;
                    const Sacado::Fad::DFad<real> soln_grad_avg  = 0.5*(soln_grad_int[idim] + soln_grad_ext[idim]);
                    const real test_jump                         = (test_int - test_ext) * flipped_normal1;
                    const real test_grad_avg                     = 0.5*(test_grad_int + test_grad_ext);
                
                    rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];
                }
            }
            // *******************
            neighbor_cell_rhs(itest_neighbor) += rhs.val();
            for (unsigned int itrial = 0; itrial < n_dofs_current_cell; ++itrial) {
                dR2_dW1[itrial] = rhs.dx(itrial);
            }
            for (unsigned int itrial = 0; itrial < n_dofs_neighbor_cell; ++itrial) {
                dR2_dW2[itrial] = rhs.dx(n_dofs_current_cell+itrial);
            }
            system_matrix.add(neighbor_dofs_indices[itest_neighbor], current_dofs_indices, dR2_dW1);
            system_matrix.add(neighbor_dofs_indices[itest_neighbor], neighbor_dofs_indices, dR2_dW2);
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, double>::assemble_face_term_implicit(
        const FEValuesBase<PHILIP_DIM,PHILIP_DIM>     *fe_values_face_current,
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM>     *fe_values_face_neighbor,
        const double penalty,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        const std::vector<types::global_dof_index> &neighbor_dofs_indices,
        Vector<double>          &current_cell_rhs,
        Vector<double>          &neighbor_cell_rhs);
} // end of PHiLiP namespace

