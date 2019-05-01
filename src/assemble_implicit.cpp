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

namespace PHiLiP
{
    using namespace dealii;

    template <int dim, int nstate, typename real>
    void DiscontinuousGalerkin<dim,nstate,real>::assemble_cell_terms_implicit(
        const FEValues<dim,dim> *fe_values_vol,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<real> &current_cell_rhs)
    {
        using ADtype = Sacado::Fad::DFad<real>;
        using ADArray = std::array<ADtype,nstate>;
        using ADArrayVector = std::array< Tensor<1,dim,ADtype>, nstate >;

        const unsigned int n_quad_pts      = fe_values_vol->n_quadrature_points;
        const unsigned int n_dofs_cell     = fe_values_vol->dofs_per_cell;

        AssertDimension (n_dofs_cell, current_dofs_indices.size());

        const std::vector<real> &JxW = fe_values_vol->get_JxW_values ();

        // AD variable
        std::vector< ADArray > solution_ad(n_dofs_cell);
        for (unsigned int i = 0; i < n_dofs_cell; ++i) {
            const int ISTATE = 0;
            solution_ad[i][ISTATE] = solution(current_dofs_indices[i]);
            solution_ad[i][ISTATE].diff(i, n_dofs_cell);
        }
        std::vector<real> residual_derivatives(n_dofs_cell);

        std::vector< ADArray > soln_at_q(n_quad_pts);
        std::vector< ADArrayVector > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

        std::vector< ADArrayVector > conv_phys_flux_at_q(n_quad_pts);
        std::vector< ADArrayVector > diss_phys_flux_at_q(n_quad_pts);
        std::vector< ADArray > source_at_q(n_quad_pts);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            for (int istate=0; istate<nstate; istate++) { 
                // Interpolate solution to the face quadrature points
                soln_at_q[iquad][istate]      = 0;
                soln_grad_at_q[iquad][istate] = 0;
                // Interpolate solution to face
                for (unsigned int itrial=0; itrial<n_dofs_cell; itrial++) {
                    soln_at_q[iquad][istate]      += solution_ad[itrial][istate] * fe_values_vol->shape_value(itrial, iquad);
                    soln_grad_at_q[iquad][istate] += solution_ad[itrial][istate] * fe_values_vol->shape_grad(itrial, iquad);
                }
            }
            // Evaluate physical convective flux and source term
            pde_physics->convective_flux (soln_at_q[iquad], conv_phys_flux_at_q[iquad]);
            pde_physics->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad], diss_phys_flux_at_q[iquad]);
            pde_physics->source_term (fe_values_vol->quadrature_point(iquad), soln_at_q[iquad], source_at_q[iquad]);
        }

        // Weak form
        // The right-hand side sends all the term to the side of the source term
        // Therefore, 
        // \divergence ( Fconv + Fdiss ) = source 
        // has the right-hand side
        // rhs = - \divergence( Fconv + Fdiss ) + source 
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

            ADArray rhs;
            for (int istate=0; istate<nstate; istate++) {
                rhs[istate] = 0.0;
            }

            for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

                for (int istate=0; istate<nstate; istate++) {
                    // Convective
                    rhs[istate] += conv_phys_flux_at_q[iquad][istate] * fe_values_vol->shape_grad(itest,iquad) * JxW[iquad];
                    // Diffusive
                    // Note that for diffusion, the negative is defined in the physics
                    //rhs[istate] += diss_phys_flux_at_q[iquad][istate] * fe_values_vol->shape_grad(itest,iquad) * JxW[iquad];
                    // Source
                    rhs[istate] += source_at_q[iquad][istate] * fe_values_vol->shape_value(itest,iquad) * JxW[iquad];
                }
            }

            const int ISTATE = 0;
            current_cell_rhs(itest) += rhs[ISTATE].val();

            for (unsigned int itrial = 0; itrial < n_dofs_cell; ++itrial) {
                //residual_derivatives[itrial] = rhs.fastAccessDx(itrial);
                residual_derivatives[itrial] = rhs[ISTATE].dx(itrial);
            }
            system_matrix.add(current_dofs_indices[itest], current_dofs_indices, residual_derivatives);
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, 1, double>::assemble_cell_terms_implicit(
        const FEValues<PHILIP_DIM,PHILIP_DIM> *fe_values_vol,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);


    template <int dim, int nstate, typename real>
    void DiscontinuousGalerkin<dim,nstate,real>::assemble_boundary_term_implicit(
        const FEFaceValues<dim,dim> *fe_values_boundary,
        const real penalty,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<real> &current_cell_rhs)
    {
        using ADtype = Sacado::Fad::DFad<real>;
        using ADArray = std::array<ADtype,nstate>;
        using ADArrayVector = std::array< Tensor<1,dim,ADtype>, nstate >;

        const unsigned int n_dofs_cell = fe_values_boundary->dofs_per_cell;
        const unsigned int n_face_quad_pts = fe_values_boundary->n_quadrature_points;

        AssertDimension (n_dofs_cell, current_dofs_indices.size());

        const std::vector<real> &JxW = fe_values_boundary->get_JxW_values ();
        const std::vector<Tensor<1,dim>> &normals = fe_values_boundary->get_normal_vectors ();

        // Recover boundary values at quadrature points
        //std::vector<real> boundary_values(n_face_quad_pts);
        //static Boundary<dim> boundary_function;
        //const unsigned int dummy = 0; // Virtual function that requires 3 arguments
        //boundary_function.value_list (fe_values_boundary->get_quadrature_points(), boundary_values, dummy);

        std::vector<ADArray> boundary_values(n_face_quad_pts);
        const std::vector< Point<dim,real> > quad_pts = fe_values_boundary->get_quadrature_points();
        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            const Point<dim, real> x_quad = quad_pts[iquad];
            pde_physics->manufactured_solution(x_quad, boundary_values[iquad]);
        }

        // AD variable
        std::vector< ADArray > solution_ad(n_dofs_cell);
        for (unsigned int i = 0; i < n_dofs_cell; ++i) {
            const int ISTATE=0;
            solution_ad[i][ISTATE] = solution(current_dofs_indices[i]);
            solution_ad[i][ISTATE].diff(i, n_dofs_cell);
        }
        std::vector<real> residual_derivatives(n_dofs_cell);

        std::vector<ADArray> soln_int(n_face_quad_pts);
        std::vector<ADArray> soln_ext(n_face_quad_pts);

        std::vector<ADArrayVector> soln_grad_int(n_face_quad_pts);
        std::vector<ADArrayVector> soln_grad_ext(n_face_quad_pts);

        std::vector<ADArrayVector> diss_phys_flux_int(n_face_quad_pts);
        std::vector<ADArrayVector> diss_phys_flux_ext(n_face_quad_pts);


        std::vector<ADArray> convective_num_flux_dot_n(n_face_quad_pts);

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            const Tensor<1,dim,ADtype> normal_int = normals[iquad];
            ADArray characteristic_dot_n_at_q = pde_physics->convective_eigenvalues(boundary_values[iquad], normal_int);

            for (int istate=0; istate<nstate; istate++) {
                // Interpolate solution to the face quadrature points
                soln_int[iquad][istate]      = 0;
                soln_grad_int[iquad][istate] = 0;
                // Interpolate solution to face
                for (unsigned int itrial=0; itrial<n_dofs_cell; itrial++) {
                    soln_int[iquad][istate]      += solution_ad[itrial][istate] * fe_values_boundary->shape_value(itrial, iquad);
                    soln_grad_int[iquad][istate] += solution_ad[itrial][istate] * fe_values_boundary->shape_grad(itrial, iquad);
                }

                const bool inflow = (characteristic_dot_n_at_q[istate] < 0.);

                if (inflow) { // Dirichlet boundary condition
                    soln_ext[iquad][istate] = boundary_values[iquad][istate];
                    // For some reason, Nitsche boundary type is adjoint inconsistent
                    //soln_ext[iquad][istate] = -soln_int[iquad][istate]+2*boundary_values[iquad][istate];
                } else { // Neumann boundary condition
                    soln_ext[iquad][istate] = soln_int[iquad][istate];
                }
            }
            // Evaluate physical convective flux and source term
            pde_physics->dissipative_flux (soln_int[iquad], soln_grad_int[iquad], diss_phys_flux_int[iquad]);

            convective_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        }

        // Applying convection boundary condition
        for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

            ADArray rhs;
            for (int istate=0; istate<nstate; istate++) {
                rhs[istate] = 0.0;
            }

            for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

                for (int istate=0; istate<nstate; istate++) {
                    rhs[istate] -= 
                        convective_num_flux_dot_n[iquad][istate] *
                        fe_values_boundary->shape_value(itest,iquad) *
                        JxW[iquad];
                }

                //     // Diffusion
                //     // *******************
                //     // Boundary condition for the diffusion
                //     // Nitsche boundary condition
                //     soln_ext = -soln_int[iquad]+2*boundary_values[iquad];
                //     // Weakly imposed boundary condition
                //     //soln_ext = boundary_values[iquad];

                //     const ADArrayVector soln_grad_ext = soln_grad_int[iquad];

                //     // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //     //       and zero everywhere else.
                //     const real test_int = fe_values_boundary->shape_value(itest, iquad);
                //     const real test_ext = 0;
                //     const Tensor<1,dim,real> test_grad_int = fe_values_boundary->shape_grad(itest, iquad);
                //     const Tensor<1,dim,real> test_grad_ext;// = 0;

                //     //const Tensor<1,dim,ADtype> soln_jump        = (soln_int[iquad] - soln_ext) * normal_int;
                //     const ADArrayVector soln_grad_avg    = 0.5*(soln_grad_int[iquad] + soln_grad_ext);
                //     const ADArrayVector test_jump        = (test_int - test_ext) * normal_int;
                //     const ADArrayVector test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);

                //     //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW[iquad];
            }
            // *******************

            const int ISTATE = 0;
            current_cell_rhs(itest) += rhs[ISTATE].val();

            for (unsigned int itrial = 0; itrial < n_dofs_cell; ++itrial) {
                //residual_derivatives[itrial] = rhs[ISTATE].fastAccessDx(itrial);
                residual_derivatives[itrial] = rhs[ISTATE].dx(itrial);
            }
            system_matrix.add(current_dofs_indices[itest], current_dofs_indices, residual_derivatives);
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, 1, double>::assemble_boundary_term_implicit(
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM> *fe_values_boundary,
        const double penalty,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        Vector<double> &current_cell_rhs);

    template <int dim, int nstate, typename real>
    void DiscontinuousGalerkin<dim,nstate,real>::assemble_face_term_implicit(
        const FEValuesBase<dim,dim>     *fe_values_int,
        const FEFaceValues<dim,dim>     *fe_values_ext,
        const real penalty,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        const std::vector<types::global_dof_index> &neighbor_dofs_indices,
        Vector<real>          &current_cell_rhs,
        Vector<real>          &neighbor_cell_rhs)
    {
        using ADtype = Sacado::Fad::DFad<real>;
        using ADArray = std::array<ADtype,nstate>;
        using ADArrayVector = std::array< Tensor<1,dim,ADtype>, nstate >;

        // Use quadrature points of neighbor cell
        // Might want to use the maximum n_quad_pts1 and n_quad_pts2
        const unsigned int n_face_quad_pts = fe_values_ext->n_quadrature_points;

        const unsigned int n_dofs_current_cell = fe_values_int->dofs_per_cell;
        const unsigned int n_dofs_neighbor_cell = fe_values_ext->dofs_per_cell;

        AssertDimension (n_dofs_current_cell, current_dofs_indices.size());
        AssertDimension (n_dofs_neighbor_cell, neighbor_dofs_indices.size());

        // Jacobian and normal should always be consistent between two elements
        // even for non-conforming meshes?
        const std::vector<real> &JxW_int = fe_values_int->get_JxW_values ();
        const std::vector<Tensor<1,dim> > &normals_int = fe_values_int->get_normal_vectors ();

        // AD variable
        std::vector<ADArray> current_solution_ad(n_dofs_current_cell);
        std::vector<ADArray> neighbor_solution_ad(n_dofs_current_cell);

        const unsigned int total_indep = n_dofs_neighbor_cell + n_dofs_neighbor_cell;

        for (unsigned int i = 0; i < n_dofs_current_cell; ++i) {
            const int ISTATE = 0;
            current_solution_ad[i][ISTATE] = solution(current_dofs_indices[i]);
            current_solution_ad[i][ISTATE].diff(i, total_indep);
        }
        for (unsigned int i = 0; i < n_dofs_neighbor_cell; ++i) {
            const int ISTATE = 0;
            neighbor_solution_ad[i][ISTATE] = solution(neighbor_dofs_indices[i]);
            neighbor_solution_ad[i][ISTATE].diff(i+n_dofs_current_cell, total_indep);
        }
        // Jacobian blocks
        std::vector<real> dR1_dW1(n_dofs_current_cell);
        std::vector<real> dR1_dW2(n_dofs_neighbor_cell);
        std::vector<real> dR2_dW1(n_dofs_current_cell);
        std::vector<real> dR2_dW2(n_dofs_neighbor_cell);

        std::vector<ADArray> normal_int_numerical_flux(n_face_quad_pts);

        // Interpolate solution to the face quadrature points
        std::vector< ADArray > soln_int(n_face_quad_pts);
        std::vector< ADArray > soln_ext(n_face_quad_pts);

        std::vector< ADArrayVector > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
        std::vector< ADArrayVector > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            const Tensor<1,dim,ADtype> normal_int = normals_int[iquad];

            for (int istate=0; istate<nstate; ++istate) {
                soln_int[iquad][istate]      = 0;
                soln_ext[iquad][istate]      = 0;
                soln_grad_int[iquad][istate] = 0;
                soln_grad_ext[iquad][istate] = 0;

                // Interpolate solution to face
                for (unsigned int itrial=0; itrial<n_dofs_current_cell; itrial++) {
                    soln_int[iquad][istate]      += current_solution_ad[itrial][istate] * fe_values_int->shape_value(itrial, iquad);
                    soln_grad_int[iquad][istate] += current_solution_ad[itrial][istate] * fe_values_int->shape_grad(itrial, iquad);
                }
                for (unsigned int itrial=0; itrial<n_dofs_neighbor_cell; itrial++) {
                    soln_ext[iquad][istate]      += neighbor_solution_ad[itrial][istate] * fe_values_ext->shape_value(itrial, iquad);
                    soln_grad_ext[iquad][istate] += neighbor_solution_ad[itrial][istate] * fe_values_ext->shape_grad(itrial, iquad);
                }
            }
            normal_int_numerical_flux[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        }

        for (unsigned int itest_current=0; itest_current<n_dofs_current_cell; ++itest_current) {
            // From test functions associated with current cell point of view
            // *******************
            ADArray rhs;
            for (int istate=0; istate<nstate; istate++) {
                rhs[istate] = 0.0;
            }

            for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
                // Convection
                for (int istate=0; istate<nstate; istate++) {
                    rhs[istate] -=
                        fe_values_int->shape_value(itest_current,iquad) *
                        normal_int_numerical_flux[iquad][istate] *
                        JxW_int[iquad];
                }

                //    // Diffusion


                //    //// Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //    ////       and zero everywhere else.
                //    //const Tensor< 1, dim, real > test_grad_int = fe_values_int->shape_grad(itest_current, iquad);
                //    //const Tensor< 1, dim, real > test_grad_ext; // Constructor initializes with zeros

                //    //const Tensor< 1, dim, real > normal1        = normals_int[iquad];
                //    //const Tensor< 1, dim, ADtype > soln_jump      = (soln_int[iquad] - soln_ext[iquad]) * normal1;
                //    //const Tensor< 1, dim, ADtype > soln_grad_avg  = 0.5*(soln_grad_ext[iquad] + soln_grad_int[iquad]);

                //    //const Tensor< 1, dim, real > test_jump        = (test_int - test_ext) * normal1;
                //    //const Tensor< 1, dim, real > test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);

                //    //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];

                //    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //    //       and zero everywhere else.
                //    const real test_int = fe_values_int->shape_value(itest_current, iquad);
                //    const real test_ext = 0;

                //    for (unsigned int idim=0; idim<dim; ++idim) {
                //        // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //        //       and zero everywhere else.
                //        const real test_grad_int = fe_values_int->shape_grad(itest_current, iquad)[idim];
                //        const real test_grad_ext = 0;

                //        // Using normals from interior point of view
                //        const real normal1          = normals_int[iquad][idim];
                //        const ADtype soln_jump      = (soln_int[iquad] - soln_ext[iquad]) * normal1;
                //        const ADtype soln_grad_avg  = 0.5*(soln_grad_int[iquad][idim] + soln_grad_ext[iquad][idim]);
                //        const real test_jump        = (test_int - test_ext) * normal1;
                //        const real test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);
                //    
                //        //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];
                //    }
            }
            // *******************

            const int ISTATE = 0;
            current_cell_rhs(itest_current) += rhs[ISTATE].val();
            for (unsigned int itrial = 0; itrial < n_dofs_current_cell; ++itrial) {
                dR1_dW1[itrial] = rhs[ISTATE].dx(itrial);
            }
            for (unsigned int itrial = 0; itrial < n_dofs_neighbor_cell; ++itrial) {
                dR1_dW2[itrial] = rhs[ISTATE].dx(n_dofs_current_cell+itrial);
            }
            system_matrix.add(current_dofs_indices[itest_current], current_dofs_indices, dR1_dW1);
            system_matrix.add(current_dofs_indices[itest_current], neighbor_dofs_indices, dR1_dW2);
        }

        for (unsigned int itest_neighbor=0; itest_neighbor<n_dofs_neighbor_cell; ++itest_neighbor) {
            // From test functions associated with neighbour cell point of view
            // *******************
            ADArray rhs;
            for (int istate=0; istate<nstate; istate++) {
                rhs[istate] = 0.0;
            }

            for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
                // Convection
                for (int istate=0; istate<nstate; istate++) {
                    rhs[istate] -=
                        fe_values_ext->shape_value(itest_neighbor,iquad) *
                        (-normal_int_numerical_flux[iquad][istate]) *
                        JxW_int[iquad];
                }

                //    // Diffusion

                //    // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //    //       and zero everywhere else.
                //    const real test_int = fe_values_ext->shape_value(itest_neighbor, iquad);
                //    const real test_ext = 0;

                //    for (unsigned int idim=0; idim<dim; ++idim) {
                //        // Note: The test function is piece-wise defined to be non-zero only on the associated cell
                //        //       and zero everywhere else.
                //        const real test_grad_int = fe_values_ext->shape_grad(itest_neighbor, iquad)[idim];
                //        const real test_grad_ext = 0;

                //        // Using normals from other side
                //        // So opposite and equal value
                //        const real flipped_normal1 = -normals_int[iquad][idim];
                //        // Flipped soln_ext and soln_int
                //        const ADtype soln_jump      = (soln_ext[iquad] - soln_int[iquad]) * flipped_normal1;
                //        const ADtype soln_grad_avg  = 0.5*(soln_grad_int[iquad][idim] + soln_grad_ext[iquad][idim]);
                //        const real test_jump        = (test_int - test_ext) * flipped_normal1;
                //        const real test_grad_avg    = 0.5*(test_grad_int + test_grad_ext);
                //    
                //        //rhs += (soln_jump * test_grad_avg + soln_grad_avg * test_jump - penalty*soln_jump*test_jump) * JxW_int[iquad];
                //    }
            }
            // *******************
            const int ISTATE = 0;
            neighbor_cell_rhs(itest_neighbor) += rhs[ISTATE].val();
            for (unsigned int itrial = 0; itrial < n_dofs_current_cell; ++itrial) {
                dR2_dW1[itrial] = rhs[ISTATE].dx(itrial);
            }
            for (unsigned int itrial = 0; itrial < n_dofs_neighbor_cell; ++itrial) {
                dR2_dW2[itrial] = rhs[ISTATE].dx(n_dofs_current_cell+itrial);
            }
            system_matrix.add(neighbor_dofs_indices[itest_neighbor], current_dofs_indices, dR2_dW1);
            system_matrix.add(neighbor_dofs_indices[itest_neighbor], neighbor_dofs_indices, dR2_dW2);
        }
    }
    template void DiscontinuousGalerkin<PHILIP_DIM, 1, double>::assemble_face_term_implicit(
        const FEValuesBase<PHILIP_DIM,PHILIP_DIM>     *fe_values_int,
        const FEFaceValues<PHILIP_DIM,PHILIP_DIM>     *fe_values_ext,
        const double penalty,
        const std::vector<types::global_dof_index> &current_dofs_indices,
        const std::vector<types::global_dof_index> &neighbor_dofs_indices,
        Vector<double>          &current_cell_rhs,
        Vector<double>          &neighbor_cell_rhs);
} // end of PHiLiP namespace

