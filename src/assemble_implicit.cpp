#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include <Sacado.hpp>
//#include <deal.II/differentiation/ad/sacado_math.h>
//#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "dg.h"
#include "boundary.h"

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
        using ADtype = Sacado::Fad::DFad<real>;

        const unsigned int n_quad_pts      = fe_values->n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values->dofs_per_cell;

        AssertDimension (n_dofs_cell, current_dofs_indices.size());

        const std::vector<real> &JxW = fe_values->get_JxW_values ();

        // AD variable
        std::vector< ADtype > solution_ad(n_dofs_cell);
        for (unsigned int i = 0; i < n_dofs_cell; ++i) {
            solution_ad[i] = solution(current_dofs_indices[i]);
            solution_ad[i].diff(i, n_dofs_cell);
        }
        std::vector<real> residual_derivatives(n_dofs_cell);

        std::vector< ADtype > soln_at_q(n_quad_pts);
        std::vector< Tensor< 1, dim, ADtype > > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

        std::vector< Tensor< 1, dim, ADtype > > conv_phys_flux_at_q(n_quad_pts);
        std::vector< Tensor< 1, dim, ADtype > > diss_phys_flux_at_q(n_quad_pts);
        std::vector< ADtype > source_at_q(n_quad_pts);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Interpolate solution to the face quadrature points
            soln_at_q[iquad]      = 0;
            soln_grad_at_q[iquad] = 0;
            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_cell; itrial++) {
                soln_at_q[iquad]      += solution_ad[itrial] * fe_values->shape_value(itrial, iquad);
                soln_grad_at_q[iquad] += solution_ad[itrial] * fe_values->shape_grad(itrial, iquad);
            }

            // Evaluate physical convective flux and source term
            pde_physics->convective_flux (soln_at_q[iquad], conv_phys_flux_at_q[iquad]);
            pde_physics->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad], diss_phys_flux_at_q[iquad]);
            pde_physics->source_term (fe_values->quadrature_point(iquad), soln_at_q[iquad], source_at_q[iquad]);
        }

        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
            ADtype rhs = 0;

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                // The right-hand side sends all the term to the side of the source term
                // Therefore, 
                // \divergence ( Fconv + Fdiss ) = source 
                // has the right-hand side
                // rhs = - \divergence( Fconv + Fdiss ) + source 

                // Weak form

                // Convective
                rhs += conv_phys_flux_at_q[iquad] * fe_values->shape_grad(itest,iquad) * JxW[iquad];
                // Diffusive
                // Note that for diffusion, the negative is defined in the physics
                rhs += diss_phys_flux_at_q[iquad] * fe_values->shape_grad(itest,iquad) * JxW[iquad];
                // Source
                rhs += source_at_q[iquad] * fe_values->shape_value(itest,iquad) * JxW[iquad];
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
        using ADtype = Sacado::Fad::DFad<real>;
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
        const std::vector< Point<dim, real> > quad_pts = fe_values_face->get_quadrature_points();
        //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        //    const Point<dim, real> x_quad = quad_pts[iquad];
        //    pde_physics->manufactured_solution(x_quad, boundary_values[iquad]);
        //}

        // AD variable
        std::vector< ADtype > solution_ad(n_dofs_cell);
        for (unsigned int i = 0; i < n_dofs_cell; ++i) {
            solution_ad[i] = solution(current_dofs_indices[i]);
            solution_ad[i].diff(i, n_dofs_cell);
        }
        std::vector<real> residual_derivatives(n_dofs_cell);

        std::vector< ADtype > soln_int(n_quad_pts);

        std::vector< Tensor< 1, dim, ADtype > > soln_grad_int(n_quad_pts);

        std::vector< Tensor< 1, dim, ADtype > > conv_phys_flux_int(n_quad_pts);
        std::vector< Tensor< 1, dim, ADtype > > diss_phys_flux_int(n_quad_pts);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            // Interpolate solution to the face quadrature points
            soln_int[iquad]      = 0;
            soln_grad_int[iquad] = 0;
            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_cell; itrial++) {
                soln_int[iquad]      += solution_ad[itrial] * fe_values_face->shape_value(itrial, iquad);
                soln_grad_int[iquad] += solution_ad[itrial] * fe_values_face->shape_grad(itrial, iquad);
            }
            // Evaluate physical convective flux and source term
            pde_physics->convective_flux (soln_int[iquad], conv_phys_flux_int[iquad]);
            pde_physics->dissipative_flux (soln_int[iquad], soln_grad_int[iquad], diss_phys_flux_int[iquad]);
        }

        // Applying convection boundary condition
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

            ADtype rhs = 0;

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                const Tensor<1,dim,ADtype> normal_int = normals[iquad];

                // Obtain solution at quadrature point
                ADtype soln_ext = 0;
                // Convection
                const Tensor<1,dim,ADtype> characteristic_velocity_at_q = pde_physics->convective_eigenvalues(soln_int[iquad]);
                const ADtype vel_dot_normal = characteristic_velocity_at_q * normal_int;
                const bool inflow = (vel_dot_normal < 0.);
                if (inflow) {
                    soln_ext = boundary_values[iquad];
                } else {
                    soln_ext = soln_int[iquad];
                }
                Tensor< 1, dim, ADtype > conv_phys_flux_ext;
                pde_physics->convective_flux (soln_ext, conv_phys_flux_ext);

                // Flux average and solution jump for scalar dissipation
                const Tensor< 1, dim, ADtype > soln_jump = normal_int * (soln_int[iquad] - soln_ext);
                const Tensor< 1, dim, ADtype > flux_avg = 0.5*(conv_phys_flux_int[iquad] + conv_phys_flux_ext);

                // Evaluate spectral radius used for scalar dissipation
                const ADtype soln_avg  = 0.5*(soln_int[iquad] + soln_ext);
                const Tensor <1, dim, ADtype> conv_eig = pde_physics->convective_eigenvalues(soln_avg);
                ADtype max_abs_eig = std::abs(conv_eig[0]);
                for (int i=1; i<dim; i++) {
                    const ADtype value = std::abs(conv_eig[i]);
                    if(value > max_abs_eig) max_abs_eig = value;
                }

                // Scalar dissipation
                Tensor< 1, dim, ADtype > numerical_flux = flux_avg + 0.5 * max_abs_eig * soln_jump;

                if (inflow) {
                    numerical_flux = conv_phys_flux_ext;
                } else {
                    numerical_flux = conv_phys_flux_int[iquad];
                }

                const ADtype normal_int_numerical_flux = numerical_flux*normal_int;

                rhs += 
                    -normal_int_numerical_flux *
                    fe_values_face->shape_value(itest,iquad) *
                    JxW[iquad];

                // Diffusion
                // *******************
                // Boundary condition for the diffusion
                // Nitsche boundary condition
                soln_ext = -soln_int[iquad]+2*boundary_values[iquad];
                // Weakly imposed boundary condition
                //soln_ext = boundary_values[iquad];

                const Tensor<1,dim,ADtype> soln_grad_ext = soln_grad_int[iquad];

                // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //       and zero everywhere else.
                const real test_int = fe_values_face->shape_value(itest, iquad);
                const real test_ext = 0;
                const Tensor<1,dim,real> test_grad_int = fe_values_face->shape_grad(itest, iquad);
                const Tensor<1,dim,real> test_grad_ext;// = 0;

                //const Tensor<1,dim,ADtype> soln_jump        = (soln_int[iquad] - soln_ext) * normal_int;
                const Tensor<1,dim,ADtype> soln_grad_avg    = 0.5*(soln_grad_int[iquad] + soln_grad_ext);
                const Tensor<1,dim,ADtype> test_jump        = (test_int - test_ext) * normal_int;
                const Tensor<1,dim,ADtype> test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);

                //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW[iquad];
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
        using ADtype = Sacado::Fad::DFad<real>;

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
        std::vector<ADtype> current_solution_ad(n_dofs_current_cell);
        std::vector<ADtype> neighbor_solution_ad(n_dofs_current_cell);

        const ADtype zeroAD = 0;

        const unsigned int total_indep = n_dofs_neighbor_cell + n_dofs_neighbor_cell;

        for (unsigned int i = 0; i < n_dofs_current_cell; ++i) {
            current_solution_ad[i] = solution(current_dofs_indices[i]);
            current_solution_ad[i].diff(i, total_indep);
        }
        for (unsigned int i = 0; i < n_dofs_neighbor_cell; ++i) {
            neighbor_solution_ad[i] = solution(neighbor_dofs_indices[i]);
            neighbor_solution_ad[i].diff(i+n_dofs_current_cell, total_indep);
        }
        // Jacobian blocks
        std::vector<real> dR1_dW1(n_dofs_current_cell);
        std::vector<real> dR1_dW2(n_dofs_neighbor_cell);
        std::vector<real> dR2_dW1(n_dofs_current_cell);
        std::vector<real> dR2_dW2(n_dofs_neighbor_cell);

        std::vector<ADtype> normal_int_numerical_flux(n_quad_pts);


        // Interpolate solution to the face quadrature points
        std::vector< ADtype > soln_int(n_quad_pts);
        std::vector< ADtype > soln_ext(n_quad_pts);

        std::vector< Tensor< 1, dim, ADtype > > soln_grad_int(n_quad_pts); // Tensor initialize with zeros
        std::vector< Tensor< 1, dim, ADtype > > soln_grad_ext(n_quad_pts); // Tensor initialize with zeros

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            soln_int[iquad]      = 0;
            soln_ext[iquad]      = 0;
            soln_grad_int[iquad] = 0;
            soln_grad_ext[iquad] = 0;

            // Interpolate solution to face
            for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                soln_int[iquad]      += current_solution_ad[itrial] * fe_values_face_current->shape_value(itrial, iquad);
                soln_grad_int[iquad] += current_solution_ad[itrial] * fe_values_face_current->shape_grad(itrial,iquad);
            }
            for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                soln_ext[iquad]      += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_value(itrial, iquad);
                soln_grad_ext[iquad] += neighbor_solution_ad[itrial] * fe_values_face_neighbor->shape_grad(itrial,iquad);
            }

            normal_int_numerical_flux[iquad] = 0;


            Tensor< 1, dim, ADtype > conv_phys_flux_int;
            Tensor< 1, dim, ADtype > conv_phys_flux_ext;

            // Numerical flux

            // Evaluate phys flux and scalar parameter for scalar dissipation
            const Tensor<1,dim,ADtype> normal1 = normals_int[iquad];
            pde_physics->convective_flux (soln_int[iquad], conv_phys_flux_int);
            pde_physics->convective_flux (soln_ext[iquad], conv_phys_flux_ext);
            
            // Scalar dissipation
            // Flux average and solution jump for scalar dissipation
            const Tensor< 1, dim, ADtype > soln_jump = normal1 * (soln_int[iquad] - soln_ext[iquad]);
            const Tensor< 1, dim, ADtype > flux_avg = 0.5*(conv_phys_flux_int + conv_phys_flux_ext);
            // Evaluate spectral radius used for scalar dissipation
            const ADtype soln_avg  = 0.5*(soln_int[iquad] + soln_ext[iquad]);
            const Tensor <1, dim, ADtype> conv_eig = pde_physics->convective_eigenvalues(soln_avg);
            ADtype max_abs_eig = std::abs(conv_eig[0]);
            for (int i=1; i<dim; i++) {
                const ADtype value = std::abs(conv_eig[i]);
                if(value > max_abs_eig) max_abs_eig = value;
            }
            Tensor< 1, dim, ADtype > numerical_flux = flux_avg + 0.5 * max_abs_eig * soln_jump;
                const ADtype vel_dot_normal = conv_eig * normal1;
                const bool inflow = (vel_dot_normal < 0.);
                if (inflow) {
                    numerical_flux = conv_phys_flux_ext;
                } else {
                    numerical_flux = conv_phys_flux_int;
                }

            normal_int_numerical_flux[iquad] = numerical_flux*normal1;
        }

        for (unsigned int itest_current=0; itest_current<n_dofs_current_cell; ++itest_current) {
            // From test functions associated with current cell point of view
            // *******************
            ADtype rhs = 0;

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                // Convection
                rhs -=
                    fe_values_face_current->shape_value(itest_current,iquad) *
                    normal_int_numerical_flux[iquad] *
                    JxW_int[iquad];

                // Diffusion


                //// Note: The test function is piece-wise defined to be non-zero only on the associated cell
                ////       and zero everywhere else.
                //const Tensor< 1, dim, real > test_grad_int = fe_values_face_current->shape_grad(itest_current, iquad);
                //const Tensor< 1, dim, real > test_grad_ext; // Constructor initializes with zeros

                //const Tensor< 1, dim, real > normal1        = normals_int[iquad];
                //const Tensor< 1, dim, ADtype > soln_jump      = (soln_int[iquad] - soln_ext[iquad]) * normal1;
                //const Tensor< 1, dim, ADtype > soln_grad_avg  = 0.5*(soln_grad_ext[iquad] + soln_grad_int[iquad]);

                //const Tensor< 1, dim, real > test_jump        = (test_int - test_ext) * normal1;
                //const Tensor< 1, dim, real > test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);

                //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];

                // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //       and zero everywhere else.
                const real test_int = fe_values_face_current->shape_value(itest_current, iquad);
                const real test_ext = 0;

                for (unsigned int idim=0; idim<dim; ++idim) {
                    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                    //       and zero everywhere else.
                    const real test_grad_int = fe_values_face_current->shape_grad(itest_current, iquad)[idim];
                    const real test_grad_ext = 0;

                    // Using normals from interior point of view
                    const real normal1          = normals_int[iquad][idim];
                    const ADtype soln_jump      = (soln_int[iquad] - soln_ext[iquad]) * normal1;
                    const ADtype soln_grad_avg  = 0.5*(soln_grad_int[iquad][idim] + soln_grad_ext[iquad][idim]);
                    const real test_jump        = (test_int - test_ext) * normal1;
                    const real test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);
                
                    //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];
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
            ADtype rhs = 0;
            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
                // Convection
                rhs -=
                    fe_values_face_neighbor->shape_value(itest_neighbor,iquad) *
                    (-normal_int_numerical_flux[iquad]) *
                    JxW_int[iquad];

                // Diffusion

                // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //       and zero everywhere else.
                const real test_int = fe_values_face_neighbor->shape_value(itest_neighbor, iquad);
                const real test_ext = 0;

                for (unsigned int idim=0; idim<dim; ++idim) {
                    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                    //       and zero everywhere else.
                    const real test_grad_int = fe_values_face_neighbor->shape_grad(itest_neighbor, iquad)[idim];
                    const real test_grad_ext = 0;

                    // Using normals from other side
                    // So opposite and equal value
                    const real flipped_normal1 = -normals_int[iquad][idim];
                    // Flipped soln_ext and soln_int
                    const ADtype soln_jump      = (soln_ext[iquad] - soln_int[iquad]) * flipped_normal1;
                    const ADtype soln_grad_avg  = 0.5*(soln_grad_int[iquad][idim] + soln_grad_ext[iquad][idim]);
                    const real test_jump        = (test_int - test_ext) * flipped_normal1;
                    const real test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);
                
                    //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];
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

