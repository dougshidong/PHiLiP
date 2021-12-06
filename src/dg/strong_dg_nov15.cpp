#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include <deal.II/fe/fe_dgq.h> // Used for flux interpolation

#include "strong_dg.hpp"

namespace PHiLiP {

template <int dim, int nstate, typename real, typename MeshType>
DGStrong<dim,nstate,real,MeshType>::DGStrong(
    const Parameters::AllParameters *const parameters_input,
    const unsigned int degree,
    const unsigned int max_degree_input,
    const unsigned int grid_degree_input,
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real,MeshType>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
{ }
// Destructor
template <int dim, int nstate, typename real, typename MeshType>
DGStrong<dim,nstate,real,MeshType>::~DGStrong ()
{
    pcout << "Destructing DGStrong..." << std::endl;
}

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int ,//face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim-1> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const bool compute_dRdW,
    const bool compute_dRdX,
    const bool compute_d2R)
{ 
    (void) current_cell_index;
    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;
 
    const unsigned int n_dofs_cell = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;
 
    AssertDimension (n_dofs_cell, soln_dof_indices.size());
 
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
 
    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);
 
    // AD variable
    std::vector< FadType > soln_coeff_int(n_dofs_cell);
    const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices[idof]);
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
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
 
        const dealii::Tensor<1,dim,FadType> normal_int = normals[iquad];
        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;
 
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }
 
        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        dealii::Point<dim,FadType> ad_point;
        for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
        this->pde_physics_fad->boundary_face_values (boundary_id, ad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);
 
        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
 
        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
 
        // Used for strong form
        // Which physical convective flux to use?
        conv_phys_flux[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);
 
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);
 
        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
   }
        }
        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
 
        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }
 
    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {
 
        FadType rhs = 0.0;
 
        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;
 
        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
 
            // Convection
            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
        }
        // *******************
 
        local_rhs_int_cell(itest) += rhs.val();
 
        if (this->all_parameters->ode_solver_param.ode_solver_type == Parameters::ODESolverParam::ODESolverEnum::implicit_solver) {
            for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
                //residual_derivatives[idof] = rhs.fastAccessDx(idof);
                residual_derivatives[idof] = rhs.fastAccessDx(idof);
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
    }
}
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &,//fe,
    const dealii::Quadrature<dim> &,//quadrature,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const bool compute_dRdW,
    const bool compute_dRdX,
    const bool compute_d2R)
{
    (void) current_cell_index;
    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;

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
    std::vector< FadType > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real,MeshType>::solution(cell_dofs_indices[idof]);
        soln_coeff[idof].diff(idof, n_dofs_cell);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the volume quadrature points
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
        conv_phys_flux_at_q[iquad] = this->pde_physics_fad->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = this->pde_physics_fad->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> real_quad_point = fe_values_vol.quadrature_point(iquad);
            dealii::Point<dim,FadType> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
            source_at_q[iquad] = this->pde_physics_fad->source_term (ad_point, soln_at_q[iquad], DGBase<dim,real,MeshType>::current_time);
        }
    }


    // Evaluate flux divergence by interpolating the flux
    // Since we have nodal values of the flux, we use the Lagrange polynomials to obtain the gradients at the quadrature points.
    //const dealii::FEValues<dim,dim> &fe_values_lagrange = this->fe_values_collection_volume_lagrange.get_present_fe_values();
    std::vector<ADArray> flux_divergence(n_quad_pts);

    std::array<std::array<std::vector<FadType>,nstate>,dim> f;
    std::array<std::array<std::vector<FadType>,nstate>,dim> g;

    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            flux_divergence[iquad][istate] = 0.0;
            for ( unsigned int flux_basis = 0; flux_basis < n_quad_pts; ++flux_basis ) {
                flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate] * fe_values_lagrange.shape_grad(flux_basis,iquad);
            }

        }
    }

    // Strong form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        FadType rhs = 0;


        const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // Convective
            // Now minus such 2 integrations by parts
            assert(JxW[iquad] - fe_values_lagrange.JxW(iquad) < 1e-14);

            rhs = rhs - fe_values_vol.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW[iquad];

            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
            // Source

            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
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
template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_derivatives(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::pair<unsigned int, int> /*face_subface_int*/,
    const std::pair<unsigned int, int> /*face_subface_ext*/,
    const typename dealii::QProjector<dim>::DataSetDescriptor /*face_data_set_int*/,
    const typename dealii::QProjector<dim>::DataSetDescriptor /*face_data_set_ext*/,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &,//fe_int,
    const dealii::FESystem<dim,dim> &,//fe_ext,
    const dealii::Quadrature<dim-1> &,//face_quadrature_int,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &,//metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW,
    const bool compute_dRdX,
    const bool compute_d2R)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    assert(compute_dRdW); assert(!compute_dRdX); assert(!compute_d2R);
    (void) compute_dRdW; (void) compute_dRdX; (void) compute_d2R;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_dofs_ext, soln_dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<FadType> soln_coeff_int_ad(n_dofs_int);
    std::vector<FadType> soln_coeff_ext_ad(n_dofs_ext);


    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_int(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_phys_flux_ext(n_face_quad_pts);

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
        soln_coeff_int_ad[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_int[idof]);
        soln_coeff_int_ad[idof].diff(idof, n_total_indep);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real,MeshType>::solution(soln_dof_indices_ext[idof]);
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

        const dealii::Tensor<1,dim,FadType> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;

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
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        conv_phys_flux_int[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);
        conv_phys_flux_ext[iquad] = this->pde_physics_fad->convective_flux (soln_ext[iquad]);

        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
    diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
   }
        }
        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = this->pde_physics_fad->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
        FadType rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux_int[iquad][istate]*normals_int[iquad];
            rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * flux_diff * JxW_int[iquad];
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
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, dR1_dW1);
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, dR1_dW2);
        }
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
        FadType rhs = 0.0;
        const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const FadType flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - conv_phys_flux_ext[iquad][istate]*(-normals_int[iquad]);
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * flux_diff * JxW_int[iquad];
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
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, dR2_dW1);
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, dR2_dW2);
        }
    }
}

/*******************************************************
 *
 *              EXPLICIT
 *
 *              **********************************************/

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_volume_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const std::vector<dealii::types::global_dof_index> &cell_dofs_indices,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const unsigned int poly_degree,
    const unsigned int grid_degree,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    (void) current_cell_index;
    //std::cout << "assembling cell terms" << std::endl;
    using realtype = real;
    using realArray = std::array<realtype,nstate>;
    using realArrayTensor1 = std::array< dealii::Tensor<1,dim,realtype>, nstate >;

//    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
//    const unsigned int n_dofs_cell     = fe_values_vol.dofs_per_cell;

    const unsigned int n_quad_pts = this->operators.volume_quadrature_collection[poly_degree].size();
    const unsigned int n_dofs_cell = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;

    AssertDimension (n_dofs_cell, cell_dofs_indices.size());

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();


    std::vector<real> residual_derivatives(n_dofs_cell);

    std::vector< realArray > soln_at_q(n_quad_pts);
    std::vector< realArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< realArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< realArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< realArray > source_at_q(n_quad_pts);



    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points(dim);
    for(int idim=0; idim<dim; idim++){
        mapping_support_points[idim].resize(n_metric_dofs/dim);
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree +1);
    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
            const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points[istate][igrid_node] += val * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate); 
        }
    }
#if 0
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = (this->high_order_grid->volume_nodes[metric_dof_indices[idof]]);
        const unsigned int istate = fe_metric.system_to_component_index(idof).first; 
        const unsigned int ishape = fe_metric.system_to_component_index(idof).second; 
        mapping_support_points[istate][ishape] = val; 
    }
#endif
    std::vector<dealii::FullMatrix<real>> metric_cofactor(n_quad_pts);
    std::vector<real> determinant_Jacobian(n_quad_pts);
    for(unsigned int iquad=0;iquad<n_quad_pts; iquad++){
        metric_cofactor[iquad].reinit(dim, dim);
    }
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts, n_metric_dofs/dim, mapping_support_points, determinant_Jacobian, metric_cofactor);

    //build physical gradient operator based on skew-symmetric form
    //get physical split grdient in covariant basis
    std::vector<std::vector<dealii::FullMatrix<real>>> physical_gradient(nstate);
    for(unsigned int istate=0; istate<nstate; istate++){
        physical_gradient[istate].resize(dim);
        for(int idim=0; idim<dim; idim++){
            physical_gradient[istate][idim].reinit(n_quad_pts, n_quad_pts);    
        }
    }
    this->operators.get_Jacobian_scaled_physical_gradient(this->operators.gradient_flux_basis[poly_degree], metric_cofactor, n_quad_pts, nstate, physical_gradient); 



    // AD variable
    std::vector< realtype > soln_coeff(n_dofs_cell);
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff[idof] = DGBase<dim,real,MeshType>::solution(cell_dofs_indices[idof]);
    }
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the volume quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
             // const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
             // soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_at_q[iquad][istate]      += soln_coeff[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
              //soln_at_q[iquad][istate]      += soln_coeff[idof] * operators.fe_collection[poly_degree].shape_value_component(idof, iquad, istate);

             // soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
             // soln_grad_at_q[iquad][istate] += soln_coeff[idof] * operators.fe_collection[poly_degree].shape_grad_component(idof, iquad, istate);//should be auxiliary variable leave it for now
        }
        //std::cout << "Density " << soln_at_q[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum " << soln_at_q[iquad][1] << std::endl;
        //std::cout << "Energy " << soln_at_q[iquad][nstate-1] << std::endl;
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = this->pde_physics_double->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = this->pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
            dealii::Point<dim> quad_point;
            for(unsigned int imetric_dof=0; imetric_dof<n_metric_dofs/dim; imetric_dof++){
                for(int idim=0; idim<dim; idim++){
                    quad_point[idim] += this->operators.mapping_shape_functions_vol_flux_nodes[grid_degree][poly_degree][iquad][imetric_dof]
                                                        * mapping_support_points[idim][imetric_dof];
                }
            }
           // source_at_q[iquad] = DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->source_term (fe_values_vol.quadrature_point(iquad), soln_at_q[iquad], DGBase<dim,real,MeshType>::current_time);
            source_at_q[iquad] = this->pde_physics_double->source_term (quad_point, soln_at_q[iquad], DGBase<dim,real,MeshType>::current_time);
        }
    }


    // Evaluate flux divergence by interpolating the flux
    // Since we have nodal values of the flux, we use the Lagrange polynomials to obtain the gradients at the quadrature points.
    //const dealii::FEValues<dim,dim> &fe_values_lagrange = this->fe_values_collection_volume_lagrange.get_present_fe_values();

   // #if 0//commented out the one from github
    std::vector<realArray> flux_divergence(n_quad_pts);
 //   std::vector<realArray> flux_div_weight_norm(n_quad_pts);
    for (int istate = 0; istate<nstate; ++istate) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            flux_divergence[iquad][istate] = 0.0;
      //      flux_div_weight_norm[iquad][istate] = 0.0;
            for ( unsigned int flux_basis = 0; flux_basis < n_quad_pts; ++flux_basis ) {
                if (this->all_parameters->use_split_form == true)
                {
                   // flux_divergence[iquad][istate] += 2* DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->convective_numerical_split_flux(soln_at_q[iquad],soln_at_q[flux_basis])[istate] *  fe_values_lagrange.shape_grad(flux_basis,iquad);
                    for(int idim=0; idim<dim; idim++){
                        flux_divergence[iquad][istate] += 2* DGBaseState<dim,nstate,real,MeshType>::pde_physics_double->convective_numerical_split_flux(soln_at_q[iquad],soln_at_q[flux_basis])[istate][idim] *  physical_gradient[istate][idim][iquad][flux_basis];
                    }
                }
                else
                {
                  //  flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate] * fe_values_lagrange.shape_grad(flux_basis,iquad);
                   // flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate] * fe_values_lagrange.shape_grad(flux_basis,iquad) * determinant_Jacobian[iquad];
                    for(int idim=0; idim<dim; idim++){
                        flux_divergence[iquad][istate] += conv_phys_flux_at_q[flux_basis][istate][idim] * physical_gradient[istate][idim][iquad][flux_basis];
                    }
                }
            }
        }
    }
   // #endif







    // Strong form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        realtype rhs = 0;
     //   realtype split = 0;

       // const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(itest).first;
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {


            // Convective
            // Now minus such 2 integrations by parts
         //   rhs = rhs - fe_values_vol.shape_value_component(itest,iquad,istate) * flux_divergence[iquad][istate] * JxW[iquad];
            rhs = rhs - this->operators.vol_integral_basis[poly_degree][iquad][itest]  * flux_divergence[iquad][istate];


            //rhs = rhs - ESFR_filter[itest][iquad] * flux_divergence[iquad][istate] * JxW[iquad];
           // rhs = rhs - ESFR_filter[itest][iquad] * alpha * flux_divergence[iquad][istate] * JxW[iquad];
           // rhs = rhs -  fe_values_vol.shape_value_component(itest,iquad,istate)* alpha * flux_divergence[iquad][istate] * JxW[iquad];

            //// Diffusive
            //// Note that for diffusion, the negative is defined in the physics
            rhs = rhs + fe_values_vol.shape_grad_component(itest,iquad,istate) * diss_phys_flux_at_q[iquad][istate] * JxW[iquad];
            // Source

            if(this->all_parameters->manufactured_convergence_study_param.manufactured_solution_param.use_manufactured_source_term) {
#if 0
//for the correct ESFR filt on source
        dealii::FullMatrix<real> ESFR_filter(n_dofs_cell);
        dealii::FullMatrix<real> Mass_matrix(n_dofs_cell);
        const unsigned int current_fe_index = fe_values_vol.get_fe().tensor_degree();
        this->operators.build_local_Mass_Matrix(JxW, n_dofs_cell, n_quad_pts, current_fe_index, Mass_matrix); 
        dealii::FullMatrix<real> K_operator(n_dofs_cell);
        this->operators.build_local_K_operator(Mass_matrix, n_dofs_cell, current_fe_index, K_operator);
        dealii::FullMatrix<real> temp(n_dofs_cell);
        temp.add(1.0, Mass_matrix, 1.0, K_operator);
        dealii::FullMatrix<real> m_inv(n_dofs_cell);
        m_inv.invert(Mass_matrix);
        dealii::FullMatrix<real> temp2(n_dofs_cell);
        temp.mmult(temp2, m_inv);
        temp2.mTmult(ESFR_filter, this->operators.basis_at_vol_cubature[current_fe_index]);
#endif

               // rhs = rhs + fe_values_vol.shape_value_component(itest,iquad,istate) * source_at_q[iquad][istate] * JxW[iquad];
                rhs = rhs + this->operators.vol_integral_basis[poly_degree][iquad][itest] * source_at_q[iquad][istate] * determinant_Jacobian[iquad];
                //rhs = rhs + ESFR_filter[itest][iquad] * source_at_q[iquad][istate] * JxW[iquad];
            }
        }

        local_rhs_int_cell(itest) += rhs;
    }
}


template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_boundary_term_explicit(
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    (void) current_cell_index;
    using ADArray = std::array<FadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadType>, nstate >;

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

    std::vector<ADArrayTensor1> conv_phys_flux(n_face_quad_pts);

    // AD variable
    std::vector< FadType > soln_coeff_int(n_dofs_cell);
    const unsigned int n_total_indep = n_dofs_cell;
    for (unsigned int idof = 0; idof < n_dofs_cell; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real,MeshType>::solution(dof_indices_int[idof]);
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
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,FadType> normal_int = normals[iquad];
        const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;

        for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        dealii::Point<dim,FadType> ad_point;
        for (int d=0;d<dim;++d) { ad_point[d] = real_quad_point[d]; }
        this->pde_physics_fad->boundary_face_values (boundary_id, ad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

        //
        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        // Used for strong form
        // Which physical convective flux to use?
        conv_phys_flux[iquad] = this->pde_physics_fad->convective_flux (soln_int[iquad]);

        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
   }
        }
        diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }

    // Boundary integral
    for (unsigned int itest=0; itest<n_dofs_cell; ++itest) {

        FadType rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const FadType flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_phys_flux[iquad][istate]*normals[iquad];
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * flux_diff * JxW[iquad];
            // Diffusive
            rhs = rhs - fe_values_boundary.shape_value_component(itest,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW[iquad];
            rhs = rhs + fe_values_boundary.shape_grad_component(itest,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW[iquad];
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

template <int dim, int nstate, typename real, typename MeshType>
void DGStrong<dim,nstate,real,MeshType>::assemble_face_term_explicit(
    const unsigned int iface, const unsigned int neighbor_iface, 
    typename dealii::DoFHandler<dim>::active_cell_iterator /*cell*/,
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int poly_degree, const unsigned int grid_degree,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
    //std::cout << "assembling face terms" << std::endl;
    using ADtype = real;
    using ADArray = std::array<ADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADtype>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
   // const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;
    const unsigned int n_face_quad_pts = this->operators.face_quadrature_collection[poly_degree].size();
    const unsigned int n_quad_pts_vol = this->operators.volume_quadrature_collection[poly_degree].size();

    const unsigned int n_dofs_int = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;
    const unsigned int n_dofs_ext = this->operators.fe_collection_basis[poly_degree].dofs_per_cell;
//    const unsigned int n_dofs_int = fe_values_int.dofs_per_cell;
//    const unsigned int n_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_dofs_int, dof_indices_int.size());
    AssertDimension (n_dofs_ext, dof_indices_ext.size());


    const dealii::FESystem<dim> &fe_metric = this->high_order_grid->fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;





    // Jacobian and normal should always be consistent between two elements
    // even for non-conforming meshes?
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
   // const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();


    //Compute metric terms (cofactor and normals)

    //get local cofactor matrix
    std::vector<std::vector<real>> mapping_support_points_int(dim);
    std::vector<std::vector<real>> mapping_support_points_ext(dim);
    for(int idim=0; idim<dim; idim++){
        mapping_support_points_int[idim].resize(n_metric_dofs/dim);
        mapping_support_points_ext[idim].resize(n_metric_dofs/dim);
    }
    dealii::QGaussLobatto<dim> vol_GLL(grid_degree + 1);
    //get the mapping support points int and ext
    for (unsigned int igrid_node = 0; igrid_node< n_metric_dofs/dim; ++igrid_node) {
        for (unsigned int idof = 0; idof< n_metric_dofs; ++idof) {
            const real val_int = (this->high_order_grid->volume_nodes[metric_dof_indices_int[idof]]);
            const unsigned int istate_int = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points_int[istate_int][igrid_node] += val_int * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate_int); 

            const real val_ext = (this->high_order_grid->volume_nodes[metric_dof_indices_ext[idof]]);
            const unsigned int istate_ext = fe_metric.system_to_component_index(idof).first; 
            mapping_support_points_ext[istate_ext][igrid_node] += val_ext * fe_metric.shape_value_component(idof,vol_GLL.point(igrid_node),istate_ext); 
        }
    }

    std::vector<dealii::FullMatrix<real>> metric_cofactor_int(n_quad_pts_vol);
    std::vector<dealii::FullMatrix<real>> metric_cofactor_ext(n_quad_pts_vol);
    std::vector<dealii::FullMatrix<real>> metric_cofactor_face(n_face_quad_pts);
    std::vector<real> determinant_Jacobian_face(n_face_quad_pts);
    std::vector<real> determinant_Jacobian(n_quad_pts_vol);//note we dont actually need this so use same for int and ext
    for(unsigned int iquad=0;iquad<n_face_quad_pts; iquad++){
        metric_cofactor_face[iquad].reinit(dim, dim);
    }
    //surface metric cofactor
    this->operators.build_local_face_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, iface,
                                                                        n_face_quad_pts, n_metric_dofs / dim, mapping_support_points_int, 
                                                                        determinant_Jacobian_face, metric_cofactor_face);

    //get physical normal scaled by jac
    const dealii::Tensor<1,dim, real> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[iface];
    std::vector<dealii::Tensor<1,dim, real> > normal_phys_int(n_face_quad_pts);
   // const std::vector<dealii::Tensor<1,dim> > &normals_dealii = fe_values_int.get_normal_vectors ();
//pcout<<" quad points "<<n_face_quad_pts<<std::endl;
    for(unsigned int iquad=0; iquad<n_face_quad_pts; iquad++){
        this->operators.compute_reference_to_physical(unit_normal_int, metric_cofactor_face[iquad], normal_phys_int[iquad]); 
    //       pcout<<"MY NORMAL "<<normal_phys_int[iquad][idim]/determinant_Jacobian_face[iquad]<<"  NORMAL DEALII  "<<normals_dealii[iquad][idim]<<" FOR DIM "<<idim<<" iquad "<<iquad<<std::endl;
    //       pcout<<" dealii jac "<<JxW_int[iquad]/this->operators.face_quadrature_collection[poly_degree].weight(iquad)<<" MY JAC "<<determinant_Jacobian_face[iquad]<<std::endl;
    }


    for(unsigned int iquad=0;iquad<n_quad_pts_vol; iquad++){
        metric_cofactor_int[iquad].reinit(dim, dim);
        metric_cofactor_ext[iquad].reinit(dim, dim);
    }
    //interior and exterior volume cofactors
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts_vol, n_metric_dofs/dim, mapping_support_points_int, determinant_Jacobian, metric_cofactor_int);
//    for(unsigned int iquad=0;iquad<n_quad_pts_vol; iquad++){
//        determinant_Jacobian[iquad] = 0.0;
//    }
    this->operators.build_local_vol_metric_cofactor_matrix_and_det_Jac(grid_degree, poly_degree, n_quad_pts_vol, n_metric_dofs/dim, mapping_support_points_ext, determinant_Jacobian, metric_cofactor_ext);


    // AD variable
   // std::vector<FadType> soln_coeff_int_ad(n_dofs_int);
   // std::vector<FadType> soln_coeff_ext_ad(n_dofs_ext);
    std::vector<ADtype> soln_coeff_int_ad(n_dofs_int);
    std::vector<ADtype> soln_coeff_ext_ad(n_dofs_ext);


    // Jacobian blocks
    std::vector<real> dR1_dW1(n_dofs_int);
    std::vector<real> dR1_dW2(n_dofs_ext);
    std::vector<real> dR2_dW1(n_dofs_int);
    std::vector<real> dR2_dW2(n_dofs_ext);

    std::vector<ADArray> conv_num_flux_dot_n(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_ref_flux_int_on_face(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_ref_flux_ext_on_face(n_face_quad_pts);

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
    for (unsigned int idof = 0; idof < n_dofs_int; ++idof) {
        soln_coeff_int_ad[idof] = DGBase<dim,real,MeshType>::solution(dof_indices_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_dofs_ext; ++idof) {
        soln_coeff_ext_ad[idof] = DGBase<dim,real,MeshType>::solution(dof_indices_ext[idof]);
    }


//START GET THE FLUX IN THE VOLUME AND INTERP TO THE FACET

//get modal coefficients of flux to interpolate to face
    std::vector< ADArray > soln_at_q_int(n_quad_pts_vol);
    std::vector< ADArray > soln_at_q_ext(n_quad_pts_vol);
    for (unsigned int iquad=0; iquad<n_quad_pts_vol; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_at_q_int[iquad][istate]      = 0.0;
            soln_at_q_ext[iquad][istate]      = 0.0;
        }
    }

    std::vector< ADArrayTensor1 > conv_ref_flux_int(n_quad_pts_vol);
    std::vector< ADArrayTensor1 > conv_ref_flux_ext(n_quad_pts_vol);
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_at_q_int[iquad][istate]      += soln_coeff_int_ad[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
        }
        //get phys flux in vol quad
        ADArrayTensor1 phys_flux = this->pde_physics_double->convective_flux (soln_at_q_int[iquad]);
        //transform to a reference flux
        for(int istate=0; istate<nstate; istate++){
            this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_int[iquad], conv_ref_flux_int[iquad][istate]);
        }
    }
    for(unsigned int iquad=0; iquad<n_quad_pts_vol; iquad++){
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_at_q_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * this->operators.basis_at_vol_cubature[poly_degree][iquad][idof];
        }
        //get phys flux in vol quad
        ADArrayTensor1 phys_flux = this->pde_physics_double->convective_flux (soln_at_q_ext[iquad]);
        //transform to a reference flux
        for(int istate=0; istate<nstate; istate++){
            this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_ext[iquad], conv_ref_flux_ext[iquad][istate]);
        }
    }

    std::vector<ADArrayTensor1> conv_ref_flux_int_vol(n_face_quad_pts);
    std::vector<ADArrayTensor1> conv_ref_flux_ext_vol(n_face_quad_pts);
    for(int istate=0; istate<nstate; istate++){
        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            conv_ref_flux_int_vol[iquad][istate]      = 0.0;
            conv_ref_flux_ext_vol[iquad][istate]      = 0.0;
            for (unsigned int iflux=0; iflux<n_quad_pts_vol; ++iflux) {
                conv_ref_flux_int_vol[iquad][istate] += conv_ref_flux_int[iflux][istate] * this->operators.flux_basis_at_facet_cubature[poly_degree][istate][iface][iquad][iflux];
                conv_ref_flux_ext_vol[iquad][istate] += conv_ref_flux_ext[iflux][istate] * this->operators.flux_basis_at_facet_cubature[poly_degree][istate][neighbor_iface][iquad][iflux];
            }
        }
    }
//END GET THE FLUX IN THE VOLUME AND INTERP TO THE FACET




    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0.0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0.0;
            soln_grad_ext[iquad][istate] = 0;
        }
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

       // const dealii::Tensor<1,dim,FadType> normal_int = normals_int[iquad];
       // const dealii::Tensor<1,dim,FadType> normal_int = normal_phys_int[iquad];
       // const dealii::Tensor<1,dim,FadType> normal_ext = -normal_int;
        const dealii::Tensor<1,dim,ADtype> normal_int = normal_phys_int[iquad];
        const dealii::Tensor<1,dim,ADtype> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_dofs_int; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int_ad[idof] * this->operators.basis_at_facet_cubature[poly_degree][iface][iquad][idof];
            soln_grad_int[iquad][istate] += soln_coeff_int_ad[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_dofs_ext; ++idof) {
            const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(idof).first;
           // const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
           // soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_ext[iquad][istate]      += soln_coeff_ext_ad[idof] * this->operators.basis_at_facet_cubature[poly_degree][neighbor_iface][iquad][idof];
            soln_grad_ext[iquad][istate] += soln_coeff_ext_ad[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
        }
        //std::cout << "Density int" << soln_int[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum int" << soln_int[iquad][1] << std::endl;
        //std::cout << "Energy int" << soln_int[iquad][nstate-1] << std::endl;
        //std::cout << "Density ext" << soln_ext[iquad][0] << std::endl;
        //if(nstate>1) std::cout << "Momentum ext" << soln_ext[iquad][1] << std::endl;
        //std::cout << "Energy ext" << soln_ext[iquad][nstate-1] << std::endl;

        // Evaluate physical convective flux, physical dissipative flux, and source term

        //std::cout <<"evaluating numerical fluxes" <<std::endl;
       // conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_fad->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        if (this->all_parameters->use_split_form == true && this->all_parameters->use_curvilinear_split_form == false){
            ADArrayTensor1 conv_surface_ref_flux;
            ADArrayTensor1 phys_flux = this->pde_physics_double->convective_flux (soln_int[iquad]);
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
            }
            conv_ref_flux_int_on_face[iquad] = this->pde_physics_double->convective_surface_numerical_split_flux(conv_surface_ref_flux, conv_ref_flux_int_vol[iquad]); 
            phys_flux = this->pde_physics_double->convective_flux (soln_ext[iquad]);
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
            }
            conv_ref_flux_ext_on_face[iquad] = this->pde_physics_double->convective_surface_numerical_split_flux(conv_surface_ref_flux, conv_ref_flux_ext_vol[iquad]); 
        } else if(this->all_parameters->use_split_form == false && this->all_parameters->use_curvilinear_split_form == true){
            ADArrayTensor1 conv_surface_ref_flux;
            ADArrayTensor1 phys_flux = this->pde_physics_double->convective_flux (soln_int[iquad]);
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
                conv_ref_flux_int_on_face[iquad][istate] = 0.5 * (conv_surface_ref_flux[istate] + conv_ref_flux_int_vol[iquad][istate]);
            }

            phys_flux = this->pde_physics_double->convective_flux (soln_ext[iquad]);
            for(int istate=0; istate<nstate; istate++){
                this->operators.compute_physical_to_reference(phys_flux[istate], metric_cofactor_face[iquad], conv_surface_ref_flux[istate]);
                conv_ref_flux_ext_on_face[iquad][istate] = 0.5 * (conv_surface_ref_flux[istate] + conv_ref_flux_ext_vol[iquad][istate]);
            }

        } else {
            conv_ref_flux_int_on_face[iquad] = conv_ref_flux_int_vol[iquad];
            conv_ref_flux_ext_on_face[iquad] = conv_ref_flux_ext_vol[iquad];
        }

       // std::cout <<"done evaluating numerical fluxes" <<std::endl;


       // diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
   for (int d=0; d<dim; d++) {
    diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
    diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
   }
        }
       // diss_flux_jump_int[iquad] = this->pde_physics_fad->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
       // diss_flux_jump_ext[iquad] = this->pde_physics_fad->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);
        diss_flux_jump_int[iquad] = this->pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = this->pde_physics_double->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

       // diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_fad->evaluate_auxiliary_flux(
        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real,MeshType>::diss_num_flux_double->evaluate_auxiliary_flux(
            0.0, 0.0,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

//const unsigned int poly_degree = pow(n_dofs_int,1.0/dim)-1.0;
//const double pi = atan(1)*4.0;
    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_dofs_int; ++itest_int) {
       // FadType rhs = 0.0;
        ADtype rhs = 0.0;
      //  const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_int).first;
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest_int).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

            // Convection
            const ADtype flux_diff = conv_num_flux_dot_n[iquad][istate] - conv_ref_flux_int_on_face[iquad][istate]*unit_normal_int;
            rhs = rhs - this->operators.face_integral_basis[poly_degree][iface][iquad][itest_int] * flux_diff;
           // rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * JxW_int[iquad]* flux_diff;


           // rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * flux_diff * JxW_int[iquad]
           //             *(poly_degree+1.0)*pi;
            // Diffusive
           // rhs = rhs - fe_values_int.shape_value_component(itest_int,iquad,istate) * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_int[iquad];
            rhs = rhs - this->operators.face_integral_basis[poly_degree][iface][iquad][itest_int] * diss_auxi_num_flux_dot_n[iquad][istate];
            rhs = rhs + fe_values_int.shape_grad_component(itest_int,iquad,istate) * diss_flux_jump_int[iquad][istate] * JxW_int[iquad];
        }

       // local_rhs_int_cell(itest_int) += rhs.val();
        local_rhs_int_cell(itest_int) += rhs;
    #if 0
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
    #endif
    }

    // From test functions associated with neighbour cell point of view
    for (unsigned int itest_ext=0; itest_ext<n_dofs_ext; ++itest_ext) {
       // FadType rhs = 0.0;
        ADtype rhs = 0.0;
       // const unsigned int istate = fe_values_int.get_fe().system_to_component_index(itest_ext).first;
        const unsigned int istate = this->operators.fe_collection_basis[poly_degree].system_to_component_index(itest_ext).first;

        for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
            // Convection
            const ADtype flux_diff = (-conv_num_flux_dot_n[iquad][istate]) - conv_ref_flux_ext_on_face[iquad][istate]*(-unit_normal_int);
            rhs = rhs - this->operators.face_integral_basis[poly_degree][neighbor_iface][iquad][itest_ext] * flux_diff;
            //rhs = rhs -  fe_values_ext.shape_value_component(itest_ext,iquad,istate) * JxW_int[iquad]* flux_diff;


          //  rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * flux_diff * JxW_int[iquad]
          //              *(poly_degree+1.0)*pi;
            // Diffusive
            rhs = rhs - fe_values_ext.shape_value_component(itest_ext,iquad,istate) * (-diss_auxi_num_flux_dot_n[iquad][istate]) * JxW_int[iquad];
            rhs = rhs + fe_values_ext.shape_grad_component(itest_ext,iquad,istate) * diss_flux_jump_ext[iquad][istate] * JxW_int[iquad];
        }

       // local_rhs_ext_cell(itest_ext) += rhs.val();
        local_rhs_ext_cell(itest_ext) += rhs;
    #if 0
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
    #endif
    }
}

// using default MeshType = Triangulation
// 1D: dealii::Triangulation<dim>;
// OW: dealii::parallel::distributed::Triangulation<dim>;
template class DGStrong <PHILIP_DIM, 1, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::Triangulation<PHILIP_DIM>>;

template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::shared::Triangulation<PHILIP_DIM>>;

#if PHILIP_DIM!=1
template class DGStrong <PHILIP_DIM, 1, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 2, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 3, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 4, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
template class DGStrong <PHILIP_DIM, 5, double, dealii::parallel::distributed::Triangulation<PHILIP_DIM>>;
#endif

} // PHiLiP namespace

