#include <deal.II/base/tensor.h>
#include <deal.II/base/table.h>

#include <deal.II/base/qprojector.h>

#include <deal.II/lac/full_matrix.templates.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include <Sacado.hpp>
//#include <deal.II/differentiation/ad/sacado_math.h>
//#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "dg.h"
#include "physics/physics_factory.h"

namespace PHiLiP {

#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
    template <int dim> using Triangulation = dealii::Triangulation<dim>;
#else
    template <int dim> using Triangulation = dealii::parallel::distributed::Triangulation<dim>;
#endif

template <int dim, int nstate, typename real>
DGWeak<dim,nstate,real>::DGWeak(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    Triangulation *const triangulation_input)
    : DGBase<dim,real>::DGBase(nstate, parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input) // Use DGBase constructor
{
    using ADtype = Sacado::Fad::DFad<real>;
    pde_physics = Physics::PhysicsFactory<dim,nstate,ADtype> ::create_Physics(parameters_input);
    conv_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, ADtype> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics);
    diss_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, ADtype> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics);

    pde_physics_double = Physics::PhysicsFactory<dim,nstate,real> ::create_Physics(parameters_input);
    conv_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics_double);
    diss_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics_double);
}
// Destructor
template <int dim, int nstate, typename real>
DGWeak<dim,nstate,real>::~DGWeak ()
{ 
    pcout << "Destructing DGWeak..." << std::endl;
    delete conv_num_flux;
    delete diss_num_flux;

    delete conv_num_flux_double;
    delete diss_num_flux_double;
}

template <int dim, typename real>
std::vector<dealii::Tensor<2,dim,real>> evaluate_metric_jacobian (
    const std::vector<dealii::Point<dim>> &points,
    const std::vector<real> &coords_coeff,
    const dealii::FESystem<dim,dim> &fe)
{
    const unsigned int n_dofs = fe.dofs_per_cell;
    const unsigned int n_pts = points.size();

    AssertDimension(n_dofs, coords_coeff.size());

    std::vector<dealii::Tensor<2,dim,real>> metric_jacobian(n_pts);

    for (unsigned int ipoint=0; ipoint<n_pts; ++ipoint) {
        std::array< dealii::Tensor<1,dim,real>, dim > coords_grad; // Tensor initialize with zeros
        for (unsigned int idof=0; idof<n_dofs; ++idof) {
            const unsigned int axis = fe.system_to_component_index(idof).first;
            coords_grad[axis] += coords_coeff[idof] * fe.shape_grad (idof, points[ipoint]);
        }
        for (int row=0;row<dim;++row) {
            for (int col=0;col<dim;++col) {
                metric_jacobian[ipoint][row][col] = coords_grad[row][col];
            }
        }
    }
    return metric_jacobian;
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_terms_dRdX(
    const dealii::FEValues<dim,dim> &,//fe_values_vol,
    const dealii::FESystem<dim,dim> &fe,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const unsigned int n_quad_pts      = quadrature.size();
    const unsigned int n_dofs_cell     = fe.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<dealii::Point<dim>> &points = quadrature.get_points ();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector< ADtype > coords_coeff(n_metric_dofs_cell);

    const unsigned int n_total_indep = n_metric_dofs_cell;
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff[idof] = this->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
        coords_coeff[idof].diff(idof, n_total_indep);
    }
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jacobian = evaluate_metric_jacobian ( points, coords_coeff, fe_metric);
    std::vector<ADtype> jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        const ADtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

        jac_det[iquad] = jacobian_determinant;
        jac_inv_tran[iquad] = jacobian_transpose_inverse;

// #ifndef NDEBUG
//         const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();
//         const real jacdet1 = jacobian_determinant.val();
//         const real jacdet2 = JxW[iquad]/quadrature.weight(iquad);
//         Assert( std::abs((jacdet1-jacdet2)/jacdet1) < 1e-12 ,
//                 dealii::ExcMessage( "Jacobian " + std::to_string(jacdet1)
//                                     +" is not the same as the one computed by FEValues "
//                                     + std::to_string(jacdet2)
//                                   )
//               );
// 
//         const dealii::DerivativeForm<1,dim,dim,real> jacobian_inverse2 = fe_values_vol.inverse_jacobian(iquad);
//         for (int d=0;d<dim;++d) {
//             for (int e=0;e<dim;++e) {
// 
//                 //std::cout << jacobian_transpose_inverse[d][e].val() << " " << jacobian_inverse2[e][d] << std::endl;
//                 const real jactrans1 = jacobian_transpose_inverse[d][e].val();
//                 const real jactrans2 = jacobian_inverse2[e][d];
//                 Assert( std::abs((jactrans1-jactrans2)/jactrans1) < 1e-12 ,
//                         dealii::ExcMessage( "Jacobian inverse transpose" + std::to_string(jactrans1) 
//                                            +" is not the same as the one computed by FEValues " + std::to_string(jactrans2)));
//             }
//         }
// #endif
    }

    std::vector< std::array<ADtype,nstate> > soln_at_q(n_quad_pts);
    std::vector< std::array< dealii::Tensor<1,dim,ADtype>, nstate > > conv_phys_flux_at_q(n_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< std::array<ADtype,nstate> > source_at_q(n_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
    }

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();
    dealii::FullMatrix<ADtype> interpolation_operator(n_dofs_cell,n_quad_pts);
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    // Might want to have the dimension as the innermost index
    // Need a contiguous 2d-array structure
    std::array<dealii::FullMatrix<ADtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(n_dofs_cell, n_quad_pts);
    }
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe.shape_grad(idof,points[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator[d][idof][iquad] = phys_shape_grad[d];
            }
        }
    }

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const unsigned int istate = fe.system_to_component_index(idof).first;
            soln_at_q[iquad][istate]      += soln_coeff[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_at_q[iquad][istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }
        conv_phys_flux_at_q[iquad] = pde_physics->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            dealii::Point<dim,ADtype> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = 0.0;}
            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                ad_point[iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }
            source_at_q[iquad] = pde_physics->source_term (ad_point, soln_at_q[iquad]);
        }
    }

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    std::vector<real> residual_derivatives(n_metric_dofs_cell);
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0;

        const unsigned int istate = fe.system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const ADtype JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

            for (int d=0;d<dim;++d) {
                // Convective
                rhs = rhs + gradient_operator[d][itest][iquad] * conv_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
                //// Diffusive
                //// Note that for diffusion, the negative is defined in the physics
                rhs = rhs + gradient_operator[d][itest][iquad] * diss_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
            }
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                rhs = rhs + interpolation_operator[itest][iquad]* source_at_q[iquad][istate] * JxW_iquad;
            }
        }

        local_rhs_int_cell(itest) += rhs.val();

        for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
            residual_derivatives[inode] = rhs.fastAccessDx(inode);
        }
        this->dRdXv.add(cell_dofs_indices[itest], cell_metric_dofs_indices, residual_derivatives);
    }

}


template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_boundary_term_dRdX(
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    //const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    std::vector<dealii::Tensor<1,dim,ADtype>> normals(n_quad_pts);

    const dealii::Quadrature<dim> face_quadrature = dealii::QProjector<dim>::project_to_face(quadrature,face_number);
    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    std::cout << "dim-1 weight: " << quadrature.weight(iquad) << " dim weight: " << face_quadrature.weight(iquad) << std::endl;
    //}

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = face_quadrature.get_points();
    std::vector<dealii::Point<dim,ADtype>> real_quad_pts(unit_quad_pts.size());

    //const std::vector<dealii::Point<dim>> &fevaluespoints = fe_values_boundary.get_quadrature_points();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector< ADtype > coords_coeff(n_metric_dofs_cell);

    const unsigned int n_total_indep = n_metric_dofs_cell;
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff[idof] = this->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
        coords_coeff[idof].diff(idof, n_total_indep);
    }
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jacobian = evaluate_metric_jacobian (unit_quad_pts, coords_coeff, fe_metric);
    std::vector<ADtype> jac_det(n_quad_pts);
    std::vector<ADtype> surface_jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran(n_quad_pts);

    const dealii::Tensor<1,dim,ADtype> unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];
    //std::cout << "list of their JxW/weight: " << std::endl;
    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    std::cout << fe_values_boundary.JxW(iquad) / fe_values_boundary.get_quadrature().weight(iquad) << std::endl;
    //}

    //std::cout << "list of their normals: " << std::endl;
    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    std::cout << fe_values_boundary.normal_vector(iquad) << std::endl;
    //}
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int d=0;d<dim;++d) { real_quad_pts[iquad][d] = 0;}
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            const int iaxis = fe_metric.system_to_component_index(idof).first;
            real_quad_pts[iquad][iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
        }

        const ADtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));


        jac_det[iquad] = jacobian_determinant;
        jac_inv_tran[iquad] = jacobian_transpose_inverse;

        const dealii::Tensor<1,dim,ADtype> normal = dealii::contract<1,0>(jacobian_transpose_inverse, unit_normal);
        const ADtype area = normal.norm();

        // Technically the normals have jac_det multiplied.
        // However, we use normalized normals by convention, so the the term
        // ends up appearing in the surface jacobian.
        normals[iquad] = normal / normal.norm();
        surface_jac_det[iquad] = normal.norm()*jac_det[iquad];

        //std::cout << "my JxW/J" << surface_jac_det[iquad].val() << std::endl;
        //std::cout << "my normals again ";
        //for (int d=0;d<dim;++d) {
        //    std::cout << normals[iquad][d].val() << " ";
        //}
        //std::cout << std::endl;
    }

    std::vector<ADArray> conv_num_flux_dot_n(n_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_quad_pts); // sigma*

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    }

    dealii::FullMatrix<ADtype> interpolation_operator(n_dofs_cell,n_quad_pts);
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    std::array<dealii::FullMatrix<ADtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(n_dofs_cell, n_quad_pts);
    }
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe.shape_grad(idof,unit_quad_pts[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator[d][idof][iquad] = phys_shape_grad[d];
            }
        }
    }
    // Interpolate solution to face
    // std::cout << "Their solution and gradient: " << std::endl;
    // for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //     std::array< ADtype, nstate > their_soln_int;
    //     std::array< dealii::Tensor<1,dim,ADtype>, nstate > their_soln_grad_int;
    //     for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
    //         const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
    //         their_soln_int[istate] += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof,iquad,istate);
    //         their_soln_grad_int[istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
    //     }

    //     for (unsigned int istate=0;istate<nstate;++istate) {
    //         std::cout << their_soln_int[istate].val() << std::endl;
    //         for (int d=0;d<dim;++d) {
    //             std::cout << their_soln_grad_int[istate][d].val() << " ";
    //         }
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "My solution and gradient: " << std::endl;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals[iquad];

        std::array<ADtype,nstate> soln_int;
        std::array<ADtype,nstate> soln_ext;
        std::array< dealii::Tensor<1,dim,ADtype>, nstate > soln_grad_int;
        std::array< dealii::Tensor<1,dim,ADtype>, nstate > soln_grad_ext;
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[istate]      = 0;
            soln_grad_int[istate] = 0;
        }
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[istate] += soln_coeff_int[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_int[istate][d] += soln_coeff_int[idof] * gradient_operator[d][idof][iquad];
            }
        }
        // for (unsigned int istate=0;istate<nstate;++istate) {
        //     std::cout << soln_int[istate].val() << std::endl;
        //     for (int d=0;d<dim;++d) {
        //         std::cout << soln_grad_int[istate][d].val() << " ";
        //     }
        // }
        // std::cout << std::endl;

        pde_physics->boundary_face_values (boundary_id, real_quad_pts[iquad], normal_int, soln_int, soln_grad_int, soln_ext, soln_grad_ext);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_ext, soln_ext, normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int, diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_int, penalty, true);
    }
    // std::cout << "Conv num flux" << std::endl;
    // for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //     for (unsigned int istate=0;istate<nstate;++istate) {
    //         std::cout << conv_num_flux_dot_n[iquad][istate].val() << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Applying convection boundary condition
    std::vector<real> residual_derivatives(n_metric_dofs_cell);
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const ADtype JxW_iquad = surface_jac_det[iquad] * face_quadrature.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator[itest][iquad] * conv_num_flux_dot_n[iquad][istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator[itest][iquad] * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator[d][itest][iquad] * diss_flux_jump_int[iquad][istate][d] * JxW_iquad;
            }
            // std::cout << "my JxW: " << JxW_iquad.val() << std::endl;
            // std::cout << "their JxW: " << fe_values_boundary.JxW(iquad) << std::endl;
        }
        // *******************

        local_rhs_int_cell(itest) += rhs.val();

        for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
            residual_derivatives[inode] = rhs.fastAccessDx(inode);
        }
        this->dRdXv.add(dof_indices_int[itest], cell_metric_dofs_indices, residual_derivatives);

    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_dRdX(
    const unsigned int interior_face_number,
    const unsigned int exterior_face_number,
    const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim> &face_quadrature_int,
    const dealii::Quadrature<dim> &face_quadrature_ext,
    const std::vector<dealii::types::global_dof_index> &interior_cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &exterior_cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_int = face_quadrature_int.get_points();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_ext = face_quadrature_ext.get_points();

    const unsigned int n_face_quad_pts = unit_quad_pts_int.size();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector< ADtype > coords_coeff_int(n_metric_dofs_cell);
    std::vector< ADtype > coords_coeff_ext(n_metric_dofs_cell);

    const unsigned int n_total_indep = 2*n_metric_dofs_cell;
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff_int[idof] = this->high_order_grid.nodes[interior_cell_metric_dofs_indices[idof]];
        coords_coeff_int[idof].diff(idof, n_total_indep);
    }
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff_ext[idof] = this->high_order_grid.nodes[exterior_cell_metric_dofs_indices[idof]];
        coords_coeff_ext[idof].diff(n_metric_dofs_cell+idof, n_total_indep);
    }
    // Use the metric Jacobian from the interior cell
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jac_int = evaluate_metric_jacobian (unit_quad_pts_int, coords_coeff_int, fe_metric);
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jac_ext = evaluate_metric_jacobian (unit_quad_pts_ext, coords_coeff_ext, fe_metric);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran_int(n_face_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran_ext(n_face_quad_pts);

    const dealii::Tensor<1,dim,ADtype> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[interior_face_number];
    const dealii::Tensor<1,dim,ADtype> unit_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[exterior_face_number];

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    //const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());


    // AD variable
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);


    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);


    // Interpolate solution to the face quadrature points
    ADArray soln_int;
    ADArray soln_ext;

    ADArrayTensor1 soln_grad_int; // Tensor initialize with zeros
    ADArrayTensor1 soln_grad_ext; // Tensor initialize with zeros

    ADArray conv_num_flux_dot_n;
    ADArray diss_soln_num_flux; // u*
    ADArray diss_auxi_num_flux_dot_n; // sigma*

    ADArrayTensor1 diss_flux_jump_int; // u*-u_int
    ADArrayTensor1 diss_flux_jump_ext; // u*-u_ext
    // AD variable
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
    }

    std::vector<ADtype> interpolation_operator_int(n_dofs_int);
    std::vector<ADtype> interpolation_operator_ext(n_dofs_ext);
    std::array<std::vector<ADtype>,dim> gradient_operator_int, gradient_operator_ext;
    for (int d=0;d<dim;++d) {
        gradient_operator_int[d].resize(n_dofs_int);
        gradient_operator_ext[d].resize(n_dofs_ext);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        const ADtype jacobian_determinant_int = dealii::determinant(metric_jac_int[iquad]);
        const ADtype jacobian_determinant_ext = dealii::determinant(metric_jac_ext[iquad]);

        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse_int = dealii::transpose(dealii::invert(metric_jac_int[iquad]));
        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse_ext = dealii::transpose(dealii::invert(metric_jac_ext[iquad]));

        const ADtype jac_det_int = jacobian_determinant_int;
        const ADtype jac_det_ext = jacobian_determinant_ext;

        const dealii::Tensor<2,dim,ADtype> jac_inv_tran_int = jacobian_transpose_inverse_int;
        const dealii::Tensor<2,dim,ADtype> jac_inv_tran_ext = jacobian_transpose_inverse_ext;

        const dealii::Tensor<1,dim,ADtype> normal_int = dealii::contract<1,0>(jacobian_transpose_inverse_int, unit_normal_int);
        const dealii::Tensor<1,dim,ADtype> normal_ext = dealii::contract<1,0>(jacobian_transpose_inverse_ext, unit_normal_ext);
        const ADtype area_int = normal_int.norm();
        const ADtype area_ext = normal_ext.norm();

        // Technically the normals have jac_det multiplied.
        // However, we use normalized normals by convention, so the the term
        // ends up appearing in the surface jacobian.

        const dealii::Tensor<1,dim,ADtype> normal_normalized_int = normal_int / area_int;
        const dealii::Tensor<1,dim,ADtype> normal_normalized_ext = -normal_normalized_int;//normal_ext / area_ext; Must use opposite normal to be consistent with explicit
        const ADtype surface_jac_det_int = area_int*jac_det_int;
        const ADtype surface_jac_det_ext = area_ext*jac_det_ext;

        for (int d=0;d<dim;++d) {
            //Assert( std::abs(normal_int[d].val()+normal_ext[d].val()) < 1e-12,
            //    dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
            //        + " N1: " + std::to_string(normal_int[d].val())
            //        + " N2: " + std::to_string(normal_ext[d].val())));
            Assert( std::abs(normal_normalized_int[d].val()+normal_normalized_ext[d].val()) < 1e-12,
                dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
                    + " N1: " + std::to_string(normal_normalized_int[d].val())
                    + " N2: " + std::to_string(normal_normalized_ext[d].val())));
        }
        Assert( std::abs(surface_jac_det_ext.val()-surface_jac_det_int.val()) < 1e-12
                || std::abs(surface_jac_det_ext.val()-std::pow(2,dim-1)*surface_jac_det_int.val()) < 1e-12 ,
                dealii::ExcMessage("Inconsistent surface Jacobians. J1: " + std::to_string(surface_jac_det_int.val())
                + " J2: " + std::to_string(surface_jac_det_ext.val())));

        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            interpolation_operator_int[idof] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran_int, fe_int.shape_grad(idof,unit_quad_pts_int[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator_int[d][idof] = phys_shape_grad[d];
            }
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            interpolation_operator_ext[idof] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran_ext, fe_ext.shape_grad(idof,unit_quad_pts_ext[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator_ext[d][idof] = phys_shape_grad[d];
            }
        }

        for (int istate=0; istate<nstate; istate++) { 
            soln_int[istate]      = 0;
            soln_grad_int[istate] = 0;
            soln_ext[istate]      = 0;
            soln_grad_ext[istate] = 0;
        }

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_int.system_to_component_index(idof).first;
            soln_int[istate]      += soln_coeff_int_ad[idof] * interpolation_operator_int[idof];
            for (int d=0;d<dim;++d) {
                soln_grad_int[istate][d] += soln_coeff_int_ad[idof] * gradient_operator_int[d][idof];
            }
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_ext.system_to_component_index(idof).first;
            soln_ext[istate]      += soln_coeff_ext_ad[idof] * interpolation_operator_ext[idof];
            for (int d=0;d<dim;++d) {
                soln_grad_ext[istate][d] += soln_coeff_ext_ad[idof] * gradient_operator_ext[d][idof];
            }
        }

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_normalized_int);
        diss_soln_num_flux = diss_num_flux->evaluate_solution_flux(soln_int, soln_ext, normal_normalized_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[s] - soln_int[s]) * normal_normalized_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[s] - soln_ext[s]) * normal_normalized_ext;
        }
        diss_flux_jump_int = pde_physics->dissipative_flux (soln_int, diss_soln_jump_int);
        diss_flux_jump_ext = pde_physics->dissipative_flux (soln_ext, diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n = diss_num_flux->evaluate_auxiliary_flux(
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_normalized_int, penalty);

        // From test functions associated with interior cell point of view
        std::vector<real> residual_derivatives(n_metric_dofs_cell);
        // Jacobian blocks
        std::vector<real> dR1_dX1(n_metric_dofs_cell);
        std::vector<real> dR1_dX2(n_metric_dofs_cell);
        std::vector<real> dR2_dX1(n_metric_dofs_cell);
        std::vector<real> dR2_dX2(n_metric_dofs_cell);
        for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
            ADtype rhs = 0.0;
            const unsigned int istate = fe_int.system_to_component_index(itest_int).first;

            const ADtype JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_int[itest_int] * conv_num_flux_dot_n[istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_int[itest_int] * diss_auxi_num_flux_dot_n[istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_int[d][itest_int] * diss_flux_jump_int[istate][d] * JxW_iquad;
            }

            local_rhs_int_cell(itest_int) += rhs.val();

            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR1_dX1[inode] = rhs.fastAccessDx(inode);
            }
            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR1_dX2[inode] = rhs.fastAccessDx(n_metric_dofs_cell+inode);
            }
            this->dRdXv.add(dof_indices_int[itest_int], interior_cell_metric_dofs_indices, dR1_dX1);
            this->dRdXv.add(dof_indices_int[itest_int], exterior_cell_metric_dofs_indices, dR1_dX2);
        }

        // From test functions associated with neighbour cell point of view
        for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
            ADtype rhs = 0.0;
            const unsigned int istate = fe_ext.system_to_component_index(itest_ext).first;

            const ADtype JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-conv_num_flux_dot_n[istate]) * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-diss_auxi_num_flux_dot_n[istate]) * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_ext[d][itest_ext] * diss_flux_jump_ext[istate][d] * JxW_iquad;
            }

            local_rhs_ext_cell(itest_ext) += rhs.val();


            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR2_dX1[inode] = rhs.fastAccessDx(inode);
            }
            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR2_dX2[inode] = rhs.fastAccessDx(n_metric_dofs_cell+inode);
            }
            this->dRdXv.add(dof_indices_ext[itest_ext], interior_cell_metric_dofs_indices, dR2_dX1);
            this->dRdXv.add(dof_indices_ext[itest_ext], exterior_cell_metric_dofs_indices, dR2_dX2);
        }
    } // Quadrature point loop
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_terms_implicit(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< ADArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< ADArray > source_at_q(n_quad_pts);


    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
        soln_coeff[idof].diff(idof, n_dofs_cell);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }
        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = pde_physics->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> real_point = fe_values_vol.quadrature_point(iquad);
            dealii::Point<dim,ADtype> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = real_point[d]; }
            source_at_q[iquad] = pde_physics->source_term (ad_point, soln_at_q[iquad]);
        }
    }

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0;

        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Convective
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * conv_phys_flux_at_q[iquad][istate] * JxW[iquad];
            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
            }
        }

        local_rhs_int_cell(itest) += rhs.val();

        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(cell_dofs_indices[itest], cell_dofs_indices, residual_derivatives);
        }
    }
}


template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_boundary_term_implicit(
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();

    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector<ADArray> soln_int(n_face_quad_pts);
    std::vector<ADArray> soln_ext(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
        soln_coeff_int[idof].diff(idof, n_total_indep);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    //std::cout << "list of their jacs: " << std::endl;
    //for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //    std::cout << fe_values_boundary.jacobian(iquad).determinant() << std::endl;
    //}
    //std::cout << "list of their normals: " << std::endl;
    //for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //    std::cout << fe_values_boundary.normal_vector(iquad) << std::endl;
    //}
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals[iquad];

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        dealii::Point<dim, ADtype> ad_point;
        for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
        pde_physics->boundary_face_values (boundary_id, ad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }
    // std::cout << "Conv num flux" << std::endl;
    // for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
    //     for (unsigned int istate=0;istate<nstate;++istate) {
    //         std::cout << conv_num_flux_dot_n[iquad][istate].val() << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // Applying convection boundary condition
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * conv_num_flux_dot_n[iquad][istate] * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
            // std::cout << "jac_det: " << fe_values_boundary.jacobian(iquad).determinant() << std::endl;
            // std::cout << "weight: " << fe_values_boundary.get_quadrature().weight(iquad) << std::endl;


            // std::cout << "JxW: " << JxW[iquad] << std::endl;
            // std::cout << "JxW / jac: " << JxW[iquad]/fe_values_boundary.jacobian(iquad).determinant() << std::endl;
            // std::cout << "JxW / weight: " << JxW[iquad]/fe_values_boundary.get_quadrature().weight(iquad) << std::endl;
        }
        // *******************

        local_rhs_int_cell(itest) += rhs.val();

        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(dof_indices_int[itest], dof_indices_int, residual_derivatives);
        }
    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_implicit(
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // In the case of the non-conforming mesh, we should be using the Jacobian
    // of the smaller face since it would be "half" of the larger one.
    // However, their curvature should match.
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);


    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< ADArray > soln_int(n_face_quad_pts);
    std::vector< ADArray > soln_ext(n_face_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

    std::vector<ADArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<ADArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
    // AD variable
    const unsigned int n_total_indep = n_dofs_int + n_dofs_ext;
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
        soln_coeff_int_ad[idof].diff(idof, n_total_indep);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
        soln_coeff_ext_ad[idof].diff(idof+n_dofs_int, n_total_indep);
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0;
            soln_grad_ext[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
        }
        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = pde_physics->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        ADtype rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            //std::cout << "quadpts: " << fe_values_int.quadrature_point(iquad) << " other " << fe_values_ext.quadrature_point(iquad) << std::endl;
            //std::cout << "Jxw: " << JxW_int[iquad] << " other " << JxW_int[iquad] << std::endl;
            Assert( ( fe_values_int.quadrature_point(iquad).distance(fe_values_ext.quadrature_point(iquad)) < 1e-12 )
                    , dealii::ExcMessage("Quadrature point should be at the same location.") );
            //Assert( JxW_int[iquad] == JxW_int[iquad], dealii::ExcMessage("JxW should be the same at interface.") );
            // Convection
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * conv_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_int_cell(itest_int) += rhs.val();
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR1_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR1_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(dof_indices_int[itest_int], dof_indices_int, dR1_dW1);
            this->system_matrix.add(dof_indices_int[itest_int], dof_indices_ext, dR1_dW2);
        }
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        ADtype rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-conv_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_ext_cell(itest_ext) += rhs.val();
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
                dR2_dW1[idof] = rhs.fastAccessDx(idof);
            }
            for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
                dR2_dW2[idof] = rhs.fastAccessDx(n_dofs_int+idof);
            }
            this->system_matrix.add(dof_indices_ext[itest_ext], dof_indices_int, dR2_dW1);
            this->system_matrix.add(dof_indices_ext[itest_ext], dof_indices_ext, dR2_dW2);
        }
    }
}


template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_terms_explicit(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    using doubleArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    std::vector< doubleArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts);

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< doubleArray > source_at_q(n_quad_pts);


    // AD variable
    std::vector< real > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }
        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = pde_physics_double->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> point = fe_values_vol.quadrature_point(iquad);
            source_at_q[iquad] = pde_physics_double->source_term (point, soln_at_q[iquad]);
        }
    }

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        real rhs = 0;

        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Convective
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * conv_phys_flux_at_q[iquad][istate] * JxW[iquad];
            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics_double
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
            }
        }

        local_rhs_int_cell(itest) += rhs;

    }
}


template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_boundary_term_explicit(
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    using doubleArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim>> &normals = fe_values_boundary.get_normal_vectors ();


    std::vector<doubleArray> soln_int(n_face_quad_pts);
    std::vector<doubleArray> soln_ext(n_face_quad_pts);

    std::vector<ADArrayTensor1> soln_grad_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> soln_grad_ext(n_face_quad_pts);

    std::vector<doubleArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<doubleArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<doubleArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    // AD variable
    std::vector< real > soln_coeff_int(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real> normal_int = normals[iquad];

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        pde_physics_double->boundary_face_values (boundary_id, real_quad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux_double->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux_double->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_double->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }

    // Applying convection boundary condition
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        real rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * conv_num_flux_dot_n[iquad][istate] * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
        }
        // *******************

        local_rhs_int_cell(itest) += rhs;
    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_explicit(
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    using doubleArray = std::array<real,nstate>;
    using doubleArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // In the case of the non-conforming mesh, we should be using the Jacobian
    // of the smaller face since it would be "half" of the larger one.
    // However, their curvature should match.
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<real> soln_coeff_int_ad(n_dofs_int);
    std::vector<real> soln_coeff_ext_ad(n_dofs_ext);

    std::vector<doubleArray> conv_num_flux_dot_n(n_face_quad_pts);

    // Interpolate solution to the face quadrature points
    std::vector< doubleArray > soln_int(n_face_quad_pts);
    std::vector< doubleArray > soln_ext(n_face_quad_pts);

    std::vector< doubleArrayTensor1 > soln_grad_int(n_face_quad_pts); // Tensor initialize with zeros
    std::vector< doubleArrayTensor1 > soln_grad_ext(n_face_quad_pts); // Tensor initialize with zeros

    std::vector<doubleArray> diss_soln_num_flux(n_face_quad_pts); // u*
    std::vector<doubleArray> diss_auxi_num_flux_dot_n(n_face_quad_pts); // sigma*

    std::vector<doubleArrayTensor1> diss_flux_jump_int(n_face_quad_pts); // u*-u_int
    std::vector<doubleArrayTensor1> diss_flux_jump_ext(n_face_quad_pts); // u*-u_ext
    // AD variable
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0;
            soln_grad_ext[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,real> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
        }

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n[iquad] = conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = diss_num_flux_double->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        doubleArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext;
        }
        diss_flux_jump_int[iquad] = pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = pde_physics_double->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_double->evaluate_auxiliary_flux(
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        real rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * conv_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_int_cell(itest_int) += rhs;
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        real rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-conv_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
        }

        local_rhs_ext_cell(itest_ext) += rhs;
    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_terms_hessian(
    const dealii::FEValues<dim,dim> &,//fe_values_vol,
    const dealii::FESystem<dim,dim> &fe,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const unsigned int n_quad_pts      = quadrature.size();
    const unsigned int n_dofs_cell     = fe.dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const std::vector<dealii::Point<dim>> &points = quadrature.get_points ();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector< ADtype > coords_coeff(n_metric_dofs_cell);

    const unsigned int n_total_indep = n_metric_dofs_cell;
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff[idof] = this->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
        coords_coeff[idof].diff(idof, n_total_indep);
    }
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jacobian = evaluate_metric_jacobian ( points, coords_coeff, fe_metric);
    std::vector<ADtype> jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        const ADtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

        jac_det[iquad] = jacobian_determinant;
        jac_inv_tran[iquad] = jacobian_transpose_inverse;
    }

    std::vector< std::array<ADtype,nstate> > soln_at_q(n_quad_pts);
    std::vector< std::array< dealii::Tensor<1,dim,ADtype>, nstate > > conv_phys_flux_at_q(n_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< std::array<ADtype,nstate> > source_at_q(n_quad_pts);

    // AD variable
    std::vector< ADtype > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(cell_dofs_indices[idof]);
    }

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();
    dealii::FullMatrix<ADtype> interpolation_operator(n_dofs_cell,n_quad_pts);
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    // Might want to have the dimension as the innermost index
    // Need a contiguous 2d-array structure
    std::array<dealii::FullMatrix<ADtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(n_dofs_cell, n_quad_pts);
    }
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe.shape_grad(idof,points[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator[d][idof][iquad] = phys_shape_grad[d];
            }
        }
    }

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const unsigned int istate = fe.system_to_component_index(idof).first;
            soln_at_q[iquad][istate]      += soln_coeff[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_at_q[iquad][istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }
        conv_phys_flux_at_q[iquad] = pde_physics->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            dealii::Point<dim,ADtype> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = 0.0;}
            for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                ad_point[iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }
            source_at_q[iquad] = pde_physics->source_term (ad_point, soln_at_q[iquad]);
        }
    }

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    std::vector<real> residual_derivatives(n_metric_dofs_cell);
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0;

        const unsigned int istate = fe.system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const ADtype JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

            for (int d=0;d<dim;++d) {
                // Convective
                rhs = rhs + gradient_operator[d][itest][iquad] * conv_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
                //// Diffusive
                //// Note that for diffusion, the negative is defined in the physics
                rhs = rhs + gradient_operator[d][itest][iquad] * diss_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
            }
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                rhs = rhs + interpolation_operator[itest][iquad]* source_at_q[iquad][istate] * JxW_iquad;
            }
        }

        local_rhs_int_cell(itest) += rhs.val();

        for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
            residual_derivatives[inode] = rhs.fastAccessDx(inode);
        }
        this->dRdXv.add(cell_dofs_indices[itest], cell_metric_dofs_indices, residual_derivatives);
    }

}


template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_boundary_term_hessian(
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_dofs_cell, dof_indices_int.size());

    //const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    std::vector<dealii::Tensor<1,dim,ADtype>> normals(n_quad_pts);

    const dealii::Quadrature<dim> face_quadrature = dealii::QProjector<dim>::project_to_face(quadrature,face_number);
    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    std::cout << "dim-1 weight: " << quadrature.weight(iquad) << " dim weight: " << face_quadrature.weight(iquad) << std::endl;
    //}

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = face_quadrature.get_points();
    std::vector<dealii::Point<dim,ADtype>> real_quad_pts(unit_quad_pts.size());

    //const std::vector<dealii::Point<dim>> &fevaluespoints = fe_values_boundary.get_quadrature_points();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector< ADtype > coords_coeff(n_metric_dofs_cell);

    const unsigned int n_total_indep = n_metric_dofs_cell;
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff[idof] = this->high_order_grid.nodes[cell_metric_dofs_indices[idof]];
        coords_coeff[idof].diff(idof, n_total_indep);
    }
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jacobian = evaluate_metric_jacobian (unit_quad_pts, coords_coeff, fe_metric);
    std::vector<ADtype> jac_det(n_quad_pts);
    std::vector<ADtype> surface_jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran(n_quad_pts);

    const dealii::Tensor<1,dim,ADtype> unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];
    //std::cout << "list of their JxW/weight: " << std::endl;
    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    std::cout << fe_values_boundary.JxW(iquad) / fe_values_boundary.get_quadrature().weight(iquad) << std::endl;
    //}

    //std::cout << "list of their normals: " << std::endl;
    //for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
    //    std::cout << fe_values_boundary.normal_vector(iquad) << std::endl;
    //}
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int d=0;d<dim;++d) { real_quad_pts[iquad][d] = 0;}
        for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
            const int iaxis = fe_metric.system_to_component_index(idof).first;
            real_quad_pts[iquad][iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
        }

        const ADtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));


        jac_det[iquad] = jacobian_determinant;
        jac_inv_tran[iquad] = jacobian_transpose_inverse;

        const dealii::Tensor<1,dim,ADtype> normal = dealii::contract<1,0>(jacobian_transpose_inverse, unit_normal);
        const ADtype area = normal.norm();

        // Technically the normals have jac_det multiplied.
        // However, we use normalized normals by convention, so the the term
        // ends up appearing in the surface jacobian.
        normals[iquad] = normal / normal.norm();
        surface_jac_det[iquad] = normal.norm()*jac_det[iquad];

        //std::cout << "my JxW/J" << surface_jac_det[iquad].val() << std::endl;
        //std::cout << "my normals again ";
        //for (int d=0;d<dim;++d) {
        //    std::cout << normals[iquad][d].val() << " ";
        //}
        //std::cout << std::endl;
    }

    std::vector<ADArray> conv_num_flux_dot_n(n_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_quad_pts); // sigma*

    // AD variable
    std::vector< ADtype > soln_coeff_int(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    }

    dealii::FullMatrix<ADtype> interpolation_operator(n_dofs_cell,n_quad_pts);
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    std::array<dealii::FullMatrix<ADtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(n_dofs_cell, n_quad_pts);
    }
    for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe.shape_grad(idof,unit_quad_pts[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator[d][idof][iquad] = phys_shape_grad[d];
            }
        }
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADtype> normal_int = normals[iquad];

        std::array<ADtype,nstate> soln_int;
        std::array<ADtype,nstate> soln_ext;
        std::array< dealii::Tensor<1,dim,ADtype>, nstate > soln_grad_int;
        std::array< dealii::Tensor<1,dim,ADtype>, nstate > soln_grad_ext;
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[istate]      = 0;
            soln_grad_int[istate] = 0;
        }
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[istate] += soln_coeff_int[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_int[istate][d] += soln_coeff_int[idof] * gradient_operator[d][idof][iquad];
            }
        }

        pde_physics->boundary_face_values (boundary_id, real_quad_pts[iquad], normal_int, soln_int, soln_grad_int, soln_ext, soln_grad_ext);

        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux->evaluate_solution_flux(soln_ext, soln_ext, normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics->dissipative_flux (soln_int, diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux->evaluate_auxiliary_flux(
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_int, penalty, true);
    }

    // Applying convection boundary condition
    std::vector<real> residual_derivatives(n_metric_dofs_cell);
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        ADtype rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const ADtype JxW_iquad = surface_jac_det[iquad] * face_quadrature.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator[itest][iquad] * conv_num_flux_dot_n[iquad][istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator[itest][iquad] * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator[d][itest][iquad] * diss_flux_jump_int[iquad][istate][d] * JxW_iquad;
            }
        }
        // *******************

        local_rhs_int_cell(itest) += rhs.val();

        for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
            residual_derivatives[inode] = rhs.fastAccessDx(inode);
        }
        this->dRdXv.add(dof_indices_int[itest], cell_metric_dofs_indices, residual_derivatives);

    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_hessian(
    const unsigned int interior_face_number,
    const unsigned int exterior_face_number,
    const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &,//fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim> &face_quadrature_int,
    const dealii::Quadrature<dim> &face_quadrature_ext,
    const std::vector<dealii::types::global_dof_index> &interior_cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &exterior_cell_metric_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_int = face_quadrature_int.get_points();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_ext = face_quadrature_ext.get_points();

    const unsigned int n_face_quad_pts = unit_quad_pts_int.size();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs_cell = fe_metric.dofs_per_cell;
    std::vector< ADtype > coords_coeff_int(n_metric_dofs_cell);
    std::vector< ADtype > coords_coeff_ext(n_metric_dofs_cell);

    const unsigned int n_total_indep = 2*n_metric_dofs_cell;
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff_int[idof] = this->high_order_grid.nodes[interior_cell_metric_dofs_indices[idof]];
        coords_coeff_int[idof].diff(idof, n_total_indep);
    }
    for (unsigned int idof = 0; idof < n_metric_dofs_cell; ++idof) {
        coords_coeff_ext[idof] = this->high_order_grid.nodes[exterior_cell_metric_dofs_indices[idof]];
        coords_coeff_ext[idof].diff(n_metric_dofs_cell+idof, n_total_indep);
    }
    // Use the metric Jacobian from the interior cell
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jac_int = evaluate_metric_jacobian (unit_quad_pts_int, coords_coeff_int, fe_metric);
    std::vector<dealii::Tensor<2,dim,ADtype>> metric_jac_ext = evaluate_metric_jacobian (unit_quad_pts_ext, coords_coeff_ext, fe_metric);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran_int(n_face_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADtype>> jac_inv_tran_ext(n_face_quad_pts);

    const dealii::Tensor<1,dim,ADtype> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[interior_face_number];
    const dealii::Tensor<1,dim,ADtype> unit_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[exterior_face_number];

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    //const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());


    // AD variable
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);


    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);


    // Interpolate solution to the face quadrature points
    ADArray soln_int;
    ADArray soln_ext;

    ADArrayTensor1 soln_grad_int; // Tensor initialize with zeros
    ADArrayTensor1 soln_grad_ext; // Tensor initialize with zeros

    ADArray conv_num_flux_dot_n;
    ADArray diss_soln_num_flux; // u*
    ADArray diss_auxi_num_flux_dot_n; // sigma*

    ADArrayTensor1 diss_flux_jump_int; // u*-u_int
    ADArrayTensor1 diss_flux_jump_ext; // u*-u_ext
    // AD variable
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real>::solution(dof_indices_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real>::solution(dof_indices_ext[idof]);
    }

    std::vector<ADtype> interpolation_operator_int(n_dofs_int);
    std::vector<ADtype> interpolation_operator_ext(n_dofs_ext);
    std::array<std::vector<ADtype>,dim> gradient_operator_int, gradient_operator_ext;
    for (int d=0;d<dim;++d) {
        gradient_operator_int[d].resize(n_dofs_int);
        gradient_operator_ext[d].resize(n_dofs_ext);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        const ADtype jacobian_determinant_int = dealii::determinant(metric_jac_int[iquad]);
        const ADtype jacobian_determinant_ext = dealii::determinant(metric_jac_ext[iquad]);

        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse_int = dealii::transpose(dealii::invert(metric_jac_int[iquad]));
        const dealii::Tensor<2,dim,ADtype> jacobian_transpose_inverse_ext = dealii::transpose(dealii::invert(metric_jac_ext[iquad]));

        const ADtype jac_det_int = jacobian_determinant_int;
        const ADtype jac_det_ext = jacobian_determinant_ext;

        const dealii::Tensor<2,dim,ADtype> jac_inv_tran_int = jacobian_transpose_inverse_int;
        const dealii::Tensor<2,dim,ADtype> jac_inv_tran_ext = jacobian_transpose_inverse_ext;

        const dealii::Tensor<1,dim,ADtype> normal_int = dealii::contract<1,0>(jacobian_transpose_inverse_int, unit_normal_int);
        const dealii::Tensor<1,dim,ADtype> normal_ext = dealii::contract<1,0>(jacobian_transpose_inverse_ext, unit_normal_ext);
        const ADtype area_int = normal_int.norm();
        const ADtype area_ext = normal_ext.norm();

        // Technically the normals have jac_det multiplied.
        // However, we use normalized normals by convention, so the the term
        // ends up appearing in the surface jacobian.

        const dealii::Tensor<1,dim,ADtype> normal_normalized_int = normal_int / area_int;
        const dealii::Tensor<1,dim,ADtype> normal_normalized_ext = -normal_normalized_int;//normal_ext / area_ext; Must use opposite normal to be consistent with explicit
        const ADtype surface_jac_det_int = area_int*jac_det_int;
        const ADtype surface_jac_det_ext = area_ext*jac_det_ext;

        for (int d=0;d<dim;++d) {
            //Assert( std::abs(normal_int[d].val()+normal_ext[d].val()) < 1e-12,
            //    dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
            //        + " N1: " + std::to_string(normal_int[d].val())
            //        + " N2: " + std::to_string(normal_ext[d].val())));
            Assert( std::abs(normal_normalized_int[d].val()+normal_normalized_ext[d].val()) < 1e-12,
                dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
                    + " N1: " + std::to_string(normal_normalized_int[d].val())
                    + " N2: " + std::to_string(normal_normalized_ext[d].val())));
        }
        Assert( std::abs(surface_jac_det_ext.val()-surface_jac_det_int.val()) < 1e-12
                || std::abs(surface_jac_det_ext.val()-std::pow(2,dim-1)*surface_jac_det_int.val()) < 1e-12 ,
                dealii::ExcMessage("Inconsistent surface Jacobians. J1: " + std::to_string(surface_jac_det_int.val())
                + " J2: " + std::to_string(surface_jac_det_ext.val())));

        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            interpolation_operator_int[idof] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran_int, fe_int.shape_grad(idof,unit_quad_pts_int[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator_int[d][idof] = phys_shape_grad[d];
            }
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            interpolation_operator_ext[idof] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
            const dealii::Tensor<1,dim,ADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran_ext, fe_ext.shape_grad(idof,unit_quad_pts_ext[iquad]));
            for (int d=0;d<dim;++d) {
                gradient_operator_ext[d][idof] = phys_shape_grad[d];
            }
        }

        for (int istate=0; istate<nstate; istate++) { 
            soln_int[istate]      = 0;
            soln_grad_int[istate] = 0;
            soln_ext[istate]      = 0;
            soln_grad_ext[istate] = 0;
        }

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = fe_int.system_to_component_index(idof).first;
            soln_int[istate]      += soln_coeff_int_ad[idof] * interpolation_operator_int[idof];
            for (int d=0;d<dim;++d) {
                soln_grad_int[istate][d] += soln_coeff_int_ad[idof] * gradient_operator_int[d][idof];
            }
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = fe_ext.system_to_component_index(idof).first;
            soln_ext[istate]      += soln_coeff_ext_ad[idof] * interpolation_operator_ext[idof];
            for (int d=0;d<dim;++d) {
                soln_grad_ext[istate][d] += soln_coeff_ext_ad[idof] * gradient_operator_ext[d][idof];
            }
        }

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_normalized_int);
        diss_soln_num_flux = diss_num_flux->evaluate_solution_flux(soln_int, soln_ext, normal_normalized_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[s] - soln_int[s]) * normal_normalized_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[s] - soln_ext[s]) * normal_normalized_ext;
        }
        diss_flux_jump_int = pde_physics->dissipative_flux (soln_int, diss_soln_jump_int);
        diss_flux_jump_ext = pde_physics->dissipative_flux (soln_ext, diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n = diss_num_flux->evaluate_auxiliary_flux(
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_normalized_int, penalty);

        // From test functions associated with interior cell point of view
        std::vector<real> residual_derivatives(n_metric_dofs_cell);
        // Jacobian blocks
        std::vector<real> dR1_dX1(n_metric_dofs_cell);
        std::vector<real> dR1_dX2(n_metric_dofs_cell);
        std::vector<real> dR2_dX1(n_metric_dofs_cell);
        std::vector<real> dR2_dX2(n_metric_dofs_cell);
        for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
            ADtype rhs = 0.0;
            const unsigned int istate = fe_int.system_to_component_index(itest_int).first;

            const ADtype JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_int[itest_int] * conv_num_flux_dot_n[istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_int[itest_int] * diss_auxi_num_flux_dot_n[istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_int[d][itest_int] * diss_flux_jump_int[istate][d] * JxW_iquad;
            }

            local_rhs_int_cell(itest_int) += rhs.val();

            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR1_dX1[inode] = rhs.fastAccessDx(inode);
            }
            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR1_dX2[inode] = rhs.fastAccessDx(n_metric_dofs_cell+inode);
            }
            this->dRdXv.add(dof_indices_int[itest_int], interior_cell_metric_dofs_indices, dR1_dX1);
            this->dRdXv.add(dof_indices_int[itest_int], exterior_cell_metric_dofs_indices, dR1_dX2);
        }

        // From test functions associated with neighbour cell point of view
        for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
            ADtype rhs = 0.0;
            const unsigned int istate = fe_ext.system_to_component_index(itest_ext).first;

            const ADtype JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-conv_num_flux_dot_n[istate]) * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-diss_auxi_num_flux_dot_n[istate]) * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_ext[d][itest_ext] * diss_flux_jump_ext[istate][d] * JxW_iquad;
            }

            local_rhs_ext_cell(itest_ext) += rhs.val();


            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR2_dX1[inode] = rhs.fastAccessDx(inode);
            }
            for (unsigned int inode = 0; inode < n_metric_dofs_cell; ++inode) {
                dR2_dX2[inode] = rhs.fastAccessDx(n_metric_dofs_cell+inode);
            }
            this->dRdXv.add(dof_indices_ext[itest_ext], interior_cell_metric_dofs_indices, dR2_dX1);
            this->dRdXv.add(dof_indices_ext[itest_ext], exterior_cell_metric_dofs_indices, dR2_dX2);
        }
    } // Quadrature point loop
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::set_physics(
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, real > >pde_physics_double_input)
{
    pde_physics_double = pde_physics_double_input;
    conv_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_convective_numerical_flux (DGBase<dim,real>::all_parameters->conv_num_flux_type, pde_physics_double);
    diss_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_dissipative_numerical_flux (DGBase<dim,real>::all_parameters->diss_num_flux_type, pde_physics_double);

}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::set_physics(
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, Sacado::Fad::DFad<real> > >pde_physics_input)
{
    pde_physics = pde_physics_input;
    conv_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, Sacado::Fad::DFad<real>> ::create_convective_numerical_flux (DGBase<dim,real>::all_parameters->conv_num_flux_type, pde_physics);
    diss_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, Sacado::Fad::DFad<real>> ::create_dissipative_numerical_flux (DGBase<dim,real>::all_parameters->diss_num_flux_type, pde_physics);
}

template class DGWeak <PHILIP_DIM, 1, double>;
template class DGWeak <PHILIP_DIM, 2, double>;
template class DGWeak <PHILIP_DIM, 3, double>;
template class DGWeak <PHILIP_DIM, 4, double>;
template class DGWeak <PHILIP_DIM, 5, double>;

} // PHiLiP namespace

