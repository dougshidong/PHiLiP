#include <deal.II/base/tensor.h>
#include <deal.II/base/table.h>

#include <deal.II/base/qprojector.h>

//#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/full_matrix.h>

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
    pde_physics_double = Physics::PhysicsFactory<dim,nstate,real> ::create_Physics(parameters_input);
    conv_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics_double);
    diss_num_flux_double = NumericalFlux::NumericalFluxFactory<dim, nstate, real> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics_double);

    using ADtype = Sacado::Fad::DFad<real>;
    pde_physics = Physics::PhysicsFactory<dim,nstate,ADtype> ::create_Physics(parameters_input);
    conv_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, ADtype> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics);
    diss_num_flux = NumericalFlux::NumericalFluxFactory<dim, nstate, ADtype> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics);

    using ADADtype = Sacado::Fad::DFad<ADtype>;
    pde_physics_fad_fad = Physics::PhysicsFactory<dim,nstate,ADADtype> ::create_Physics(parameters_input);
    conv_num_flux_fad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, ADADtype> ::create_convective_numerical_flux (parameters_input->conv_num_flux_type, pde_physics_fad_fad);
    diss_num_flux_fad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, ADADtype> ::create_dissipative_numerical_flux (parameters_input->diss_num_flux_type, pde_physics_fad_fad);
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

    delete conv_num_flux_fad_fad;
    delete diss_num_flux_fad_fad;
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
real DGWeak<dim,nstate,real>::evaluate_CFL (
    std::vector< std::array<real,nstate> > soln_at_q,
    const real cell_diameter
    )
{
    const unsigned int n_pts = soln_at_q.size();
    std::vector< real > convective_eigenvalues(n_pts);
    for (unsigned int isol = 0; isol < n_pts; ++isol) {
        convective_eigenvalues[isol] = pde_physics_double->max_convective_eigenvalue (soln_at_q[isol]);
        //viscosities[isol] = pde_physics_double->compute_diffusion_coefficient (soln_at_q[isol]);
    }
    const real max_eig = *(std::max_element(convective_eigenvalues.begin(), convective_eigenvalues.end()));

    return cell_diameter / max_eig;
}



template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_terms_explicit(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    using doubleArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const unsigned int n_quad_pts      = fe_values_vol.n_quadrature_points;
    const unsigned int n_soln_dofs_int     = fe_values_vol.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());

    const std::vector<real> &JxW = fe_values_vol.get_JxW_values ();

    std::vector< doubleArray > soln_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts);

    std::vector< ADArrayTensor1 > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< doubleArray > source_at_q(n_quad_pts);


    // AD variable
    std::vector< real > soln_coeff(n_soln_dofs_int);
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        soln_coeff[idof] = DGBase<dim,real>::solution(soln_dof_indices_int[idof]);
    }

    const double cell_diameter = fe_values_vol.get_cell()->diameter();
    const real artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                       this->discontinuity_sensor(cell_diameter, soln_coeff, fe_values_vol.get_fe())
                                       : 0.0;

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
    }
    // Interpolate solution to face
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
              const unsigned int istate = fe_values_vol.get_fe().system_to_component_index(idof).first;
              soln_at_q[iquad][istate]      += soln_coeff[idof] * fe_values_vol.shape_value_component(idof, iquad, istate);
              soln_grad_at_q[iquad][istate] += soln_coeff[idof] * fe_values_vol.shape_grad_component(idof, iquad, istate);
        }
        // Evaluate physical convective flux and source term
        conv_phys_flux_at_q[iquad] = pde_physics_double->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        if(this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_phys_flux_at_q = pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff, soln_at_q[iquad], soln_grad_at_q[iquad]);
            for (int istate=0; istate<nstate; istate++) { 
                diss_phys_flux_at_q[iquad][istate] += artificial_diss_phys_flux_at_q[istate];
            }
        }
        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> point = fe_values_vol.quadrature_point(iquad);
            source_at_q[iquad] = pde_physics_double->source_term (point, soln_at_q[iquad]);
            //std::array<real,nstate> artificial_source_at_q = pde_physics_double->artificial_source_term (artificial_diss_coeff, point, soln_at_q[iquad]);
            //for (int s=0;s<nstate;++s) source_at_q[iquad][s] += artificial_source_at_q[s];
        }
    }

    const unsigned int cell_index = fe_values_vol.get_cell()->active_cell_index();
    this->max_dt_cell[cell_index] = evaluate_CFL ( soln_at_q, cell_diameter );

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore, 
    // \divergence ( Fconv + Fdiss ) = source 
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source 
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {

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
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    using doubleArray = std::array<real,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    const unsigned int n_soln_dofs_int = fe_values_boundary.dofs_per_cell;
    const unsigned int n_face_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());

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
    std::vector< real > soln_coeff_int(n_soln_dofs_int);
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(soln_dof_indices_int[idof]);
    }

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
        }
    }

    const double cell_diameter = fe_values_boundary.get_cell()->diameter();
    const real artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                       this->discontinuity_sensor(cell_diameter, soln_coeff_int, fe_values_boundary.get_fe())
                                       : 0.0;

    // Interpolate solution to face
    const std::vector< dealii::Point<dim,real> > quad_pts = fe_values_boundary.get_quadrature_points();
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real> normal_int = normals[iquad];

        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_boundary.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_boundary.shape_grad_component(idof, iquad, istate);
        }

        const dealii::Point<dim, real> real_quad_point = quad_pts[iquad];
        pde_physics_double->boundary_face_values (boundary_id, real_quad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

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
        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff, soln_int[iquad], diss_soln_jump_int);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_double->evaluate_auxiliary_flux(
            artificial_diss_coeff,
            artificial_diss_coeff,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty, true);
    }

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {

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
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    using doubleArray = std::array<real,nstate>;
    using doubleArrayTensor1 = std::array< dealii::Tensor<1,dim,real>, nstate >;

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

    const unsigned int n_soln_dofs_int = fe_values_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_values_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    // Jacobian and normal should always be consistent between two elements
    // In the case of the non-conforming mesh, we should be using the Jacobian
    // of the smaller face since it would be "half" of the larger one.
    // This should be consistent with the DGBase decision of which cells is reponsible for
    // the face.
    // However, their curvature should match.
    const std::vector<real> &JxW_int = fe_values_int.get_JxW_values ();
    const std::vector<dealii::Tensor<1,dim> > &normals_int = fe_values_int.get_normal_vectors ();

    // AD variable
    std::vector<real> soln_coeff_int(n_soln_dofs_int);
    std::vector<real> soln_coeff_ext(n_soln_dofs_ext);

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
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        soln_coeff_int[idof] = DGBase<dim,real>::solution(soln_dof_indices_int[idof]);
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        soln_coeff_ext[idof] = DGBase<dim,real>::solution(soln_dof_indices_ext[idof]);
    }
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[iquad][istate]      = 0;
            soln_grad_int[iquad][istate] = 0;
            soln_ext[iquad][istate]      = 0;
            soln_grad_ext[iquad][istate] = 0;
        }
    }

    const double cell_diameter_int = fe_values_int.get_cell()->diameter();
    const double cell_diameter_ext = fe_values_ext.get_cell()->diameter();
    const real artificial_diss_coeff_int = this->all_parameters->add_artificial_dissipation ?
                                           this->discontinuity_sensor(cell_diameter_int, soln_coeff_int, fe_values_int.get_fe())
                                           : 0.0;
    const real artificial_diss_coeff_ext = this->all_parameters->add_artificial_dissipation ?
                                           this->discontinuity_sensor(cell_diameter_ext, soln_coeff_ext, fe_values_ext.get_fe())
                                           : 0.0;

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,real> normal_int = normals_int[iquad];
        const dealii::Tensor<1,dim,real> normal_ext = -normal_int;

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            const unsigned int istate = fe_values_int.get_fe().system_to_component_index(idof).first;
            soln_int[iquad][istate]      += soln_coeff_int[idof] * fe_values_int.shape_value_component(idof, iquad, istate);
            soln_grad_int[iquad][istate] += soln_coeff_int[idof] * fe_values_int.shape_grad_component(idof, iquad, istate);
        }
        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
            const unsigned int istate = fe_values_ext.get_fe().system_to_component_index(idof).first;
            soln_ext[iquad][istate]      += soln_coeff_ext[idof] * fe_values_ext.shape_value_component(idof, iquad, istate);
            soln_grad_ext[iquad][istate] += soln_coeff_ext[idof] * fe_values_ext.shape_grad_component(idof, iquad, istate);
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

        if (this->all_parameters->add_artificial_dissipation) {
            const doubleArrayTensor1 artificial_diss_flux_jump_int = pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff_int, soln_int[iquad], diss_soln_jump_int);
            const doubleArrayTensor1 artificial_diss_flux_jump_ext = pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff_ext, soln_ext[iquad], diss_soln_jump_ext);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
                diss_flux_jump_ext[iquad][s] += artificial_diss_flux_jump_ext[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_double->evaluate_auxiliary_flux(
            artificial_diss_coeff_int,
            artificial_diss_coeff_ext,
            soln_int[iquad], soln_ext[iquad],
            soln_grad_int[iquad], soln_grad_ext[iquad],
            normal_int, penalty);
    }

    // From test functions associated with interior cell point of view
    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
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
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
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
void DGWeak<dim,nstate,real>::assemble_boundary_term_derivatives(
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADADtype = Sacado::Fad::DFad<ADtype>;
    using ADArray = std::array<ADADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADADtype>, nstate >;

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;

    const unsigned int n_soln_dofs = fe_values_boundary.dofs_per_cell;
    const unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    //const std::vector<real> &JxW = fe_values_boundary.get_JxW_values ();
    std::vector<dealii::Tensor<1,dim,ADADtype>> normals(n_quad_pts);

    const dealii::Quadrature<dim> face_quadrature = dealii::QProjector<dim>::project_to_face(quadrature,face_number);

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = face_quadrature.get_points();
    std::vector<dealii::Point<dim,ADADtype>> real_quad_pts(unit_quad_pts.size());

    //const std::vector<dealii::Point<dim>> &fevaluespoints = fe_values_boundary.get_quadrature_points();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector< ADADtype > coords_coeff(n_metric_dofs);
    std::vector< ADADtype > soln_coeff(n_soln_dofs);

    // Derivatives are ordered such that w comes first with index 0, then x.
    // If derivatives with respect to w are not needed, then derivatives
    // with respect to x will start at index 0.
    unsigned int w_start = 0, w_end = 0, x_start = 0, x_end = 0;
    if (compute_d2R || (compute_dRdW && compute_dRdX)) {
        w_start = 0;
        w_end   = w_start + n_soln_dofs;
        x_start = w_end;
        x_end   = x_start + n_metric_dofs;
    } else if (compute_dRdW) {
        w_start = 0;
        w_end   = w_start + n_soln_dofs;
        x_start = w_end;
        x_end   = x_start + 0;
    } else if (compute_dRdX) {
        w_start = 0;
        w_end   = w_start + 0;
        x_start = w_end;
        x_end   = x_start + n_metric_dofs;
    } else {
        std::cout << "Called the derivative version of the residual without requesting the derivative" << std::endl;
    }

    unsigned int i_derivative = 0;
    const unsigned int n_total_indep = x_end;

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;
        soln_coeff[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
        coords_coeff[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdX || compute_d2R) i_derivative++;
    }

    AssertDimension(i_derivative, n_total_indep);

    std::vector<dealii::Tensor<2,dim,ADADtype>> metric_jacobian = evaluate_metric_jacobian (unit_quad_pts, coords_coeff, fe_metric);
    std::vector<ADADtype> jac_det(n_quad_pts);
    std::vector<ADADtype> surface_jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADADtype>> jac_inv_tran(n_quad_pts);

    const dealii::Tensor<1,dim,ADADtype> unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        if (compute_metric_derivatives) {
            for (int d=0;d<dim;++d) { real_quad_pts[iquad][d] = 0;}
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                real_quad_pts[iquad][iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }

            const ADADtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
            const dealii::Tensor<2,dim,ADADtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

            jac_det[iquad] = jacobian_determinant;
            jac_inv_tran[iquad] = jacobian_transpose_inverse;

            const dealii::Tensor<1,dim,ADADtype> normal = dealii::contract<1,0>(jacobian_transpose_inverse, unit_normal);
            const ADADtype area = normal.norm();

            surface_jac_det[iquad] = normal.norm()*jac_det[iquad];
            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.
            normals[iquad] = normal / normal.norm();

            // Exact mapping
            // real_quad_pts[iquad] = fe_values_boundary.quadrature_point(iquad);
            // surface_jac_det[iquad] = fe_values_boundary.JxW(iquad) / face_quadrature.weight(iquad);
            // normals[iquad] = fe_values_boundary.normal_vector(iquad);

        } else {
            real_quad_pts[iquad] = fe_values_boundary.quadrature_point(iquad);
            surface_jac_det[iquad] = fe_values_boundary.JxW(iquad) / face_quadrature.weight(iquad);
            normals[iquad] = fe_values_boundary.normal_vector(iquad);
        }

    }

    std::vector<ADArray> conv_num_flux_dot_n(n_quad_pts);
    std::vector<ADArray> diss_soln_num_flux(n_quad_pts); // u*
    std::vector<ADArrayTensor1> diss_flux_jump_int(n_quad_pts); // u*-u_int
    std::vector<ADArray> diss_auxi_num_flux_dot_n(n_quad_pts); // sigma*

    dealii::FullMatrix<real> interpolation_operator(n_soln_dofs,n_quad_pts);
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    //std::array<dealii::FullMatrix<ADADtype>,dim> gradient_operator;
    // for (int d=0;d<dim;++d) {
    //     gradient_operator[d].reinit(n_soln_dofs, n_quad_pts);
    // }
    std::array<dealii::Table<2,ADADtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(dealii::TableIndices<2>(n_soln_dofs, n_quad_pts));
    }
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            if (compute_metric_derivatives) {
                const dealii::Tensor<1,dim,ADADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe.shape_grad(idof,unit_quad_pts[iquad]));
                for (int d=0;d<dim;++d) {
                    gradient_operator[d][idof][iquad] = phys_shape_grad[d];
                }

                // Exact mapping
                // for (int d=0;d<dim;++d) {
                //     const unsigned int istate = fe.system_to_component_index(idof).first;
                //     gradient_operator[d][idof][iquad] = fe_values_boundary.shape_grad_component(idof, iquad, istate)[d];
                // }
            } else {
                for (int d=0;d<dim;++d) {
                    const unsigned int istate = fe.system_to_component_index(idof).first;
                    gradient_operator[d][idof][iquad] = fe_values_boundary.shape_grad_component(idof, iquad, istate)[d];
                }
            }
        }
    }

    const double cell_diameter = fe_values_boundary.get_cell()->diameter();
    const ADADtype artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                           this->discontinuity_sensor(cell_diameter, soln_coeff, fe_values_boundary.get_fe())
                                           : 0.0;

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,ADADtype> normal_int = normals[iquad];

        std::array<ADADtype,nstate> soln_int;
        std::array<ADADtype,nstate> soln_ext;
        std::array< dealii::Tensor<1,dim,ADADtype>, nstate > soln_grad_int;
        std::array< dealii::Tensor<1,dim,ADADtype>, nstate > soln_grad_ext;
        for (int istate=0; istate<nstate; istate++) { 
            soln_int[istate]      = 0;
            soln_grad_int[istate] = 0;
        }
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
            const int istate = fe_values_boundary.get_fe().system_to_component_index(idof).first;
            soln_int[istate] += soln_coeff[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_int[istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }

        pde_physics_fad_fad->boundary_face_values (boundary_id, real_quad_pts[iquad], normal_int, soln_int, soln_grad_int, soln_ext, soln_grad_ext);

        // Evaluate physical convective flux, physical dissipative flux
        // Following the the boundary treatment given by 
        //      Hartmann, R., Numerical Analysis of Higher Order Discontinuous Galerkin Finite Element Methods,
        //      Institute of Aerodynamics and Flow Technology, DLR (German Aerospace Center), 2008.
        //      Details given on page 93
        //conv_num_flux_dot_n[iquad] = conv_num_flux_fad_fad->evaluate_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        // So, I wasn't able to get Euler manufactured solutions to converge when F* = F*(Ubc, Ubc)
        // Changing it back to the standdard F* = F*(Uin, Ubc)
        // This is known not be adjoint consistent as per the paper above. Page 85, second to last paragraph.
        // Losing 2p+1 OOA on functionals for all PDEs.
        conv_num_flux_dot_n[iquad] = conv_num_flux_fad_fad->evaluate_flux(soln_int, soln_ext, normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux_fad_fad->evaluate_solution_flux(soln_ext, soln_ext, normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[iquad][s] - soln_int[s]) * normal_int;
        }
        diss_flux_jump_int[iquad] = pde_physics_fad_fad->dissipative_flux (soln_int, diss_soln_jump_int);

        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = pde_physics_fad_fad->artificial_dissipative_flux (artificial_diss_coeff, soln_int, diss_soln_jump_int);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux_fad_fad->evaluate_auxiliary_flux(
            artificial_diss_coeff,
            artificial_diss_coeff,
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_int, penalty, true);
    }

    // Applying convection boundary condition
    ADADtype dual_dot_residual = 0.0;
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        ADADtype rhs = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const ADADtype JxW_iquad = surface_jac_det[iquad] * face_quadrature.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator[itest][iquad] * conv_num_flux_dot_n[iquad][istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator[itest][iquad] * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator[d][itest][iquad] * diss_flux_jump_int[iquad][istate][d] * JxW_iquad;
            }
        }


        local_rhs_cell(itest) += rhs.val().val();

        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = rhs.dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = rhs.dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        if (compute_d2R) {
            const unsigned int global_residual_row = soln_dof_indices[itest];
            dual_dot_residual += this->dual[global_residual_row]*rhs;
        }

    }

    if (compute_d2R) {
        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;
            const ADtype dWi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }
        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;
            const ADtype dXi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = dXi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }
    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_derivatives(
    const unsigned int interior_face_number,
    const unsigned int exterior_face_number,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::Quadrature<dim> &face_quadrature_int,
    const dealii::Quadrature<dim> &face_quadrature_ext,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices_ext,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADADtype = Sacado::Fad::DFad<ADtype>;
    using ADArray = std::array<ADADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADADtype>, nstate >;

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_int = face_quadrature_int.get_points();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_ext = face_quadrature_ext.get_points();

    const unsigned int n_face_quad_pts = unit_quad_pts_int.size();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    std::vector< ADADtype > coords_coeff_int(n_metric_dofs);
    std::vector< ADADtype > coords_coeff_ext(n_metric_dofs);
    std::vector< ADADtype > soln_coeff_int(n_soln_dofs_int);
    std::vector< ADADtype > soln_coeff_ext(n_soln_dofs_ext);

    // Current derivative order is: soln_int, soln_ext, metric_int, metric_ext
    unsigned int w_int_start = 0, w_int_end = 0, w_ext_start = 0, w_ext_end = 0,
                 x_int_start = 0, x_int_end = 0, x_ext_start = 0, x_ext_end = 0;
    if (compute_d2R || (compute_dRdW && compute_dRdX)) {
        w_int_start = 0;
        w_int_end   = w_int_start + n_soln_dofs_int;
        w_ext_start = w_int_end;
        w_ext_end   = w_ext_start + n_soln_dofs_ext;
        x_int_start = w_ext_end;
        x_int_end   = x_int_start + n_metric_dofs;
        x_ext_start = x_int_end;
        x_ext_end   = x_ext_start + n_metric_dofs;
    } else if (compute_dRdW) {
        w_int_start = 0;
        w_int_end   = w_int_start + n_soln_dofs_int;
        w_ext_start = w_int_end;
        w_ext_end   = w_ext_start + n_soln_dofs_ext;
        x_int_start = w_ext_end;
        x_int_end   = x_int_start + 0;
        x_ext_start = x_int_end;
        x_ext_end   = x_ext_start + 0;
    } else if (compute_dRdX) {
        w_int_start = 0;
        w_int_end   = w_int_start + 0;
        w_ext_start = w_int_end;
        w_ext_end   = w_ext_start + 0;
        x_int_start = w_ext_end;
        x_int_end   = x_int_start + n_metric_dofs;
        x_ext_start = x_int_end;
        x_ext_end   = x_ext_start + n_metric_dofs;
    } else {
        std::cout << "Called the derivative version of the residual without requesting the derivative" << std::endl;
    }

    const unsigned int n_total_indep = x_ext_end;
    unsigned int i_derivative = 0;

    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        const real val = this->solution(soln_dof_indices_int[idof]);
        soln_coeff_int[idof] = val;
        soln_coeff_int[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff_int[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff_int[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        const real val = this->solution(soln_dof_indices_ext[idof]);
        soln_coeff_ext[idof] = val;
        soln_coeff_ext[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff_ext[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff_ext[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.nodes[metric_dof_indices_int[idof]];
        coords_coeff_int[idof] = val;
        coords_coeff_int[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff_int[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff_int[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdX || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.nodes[metric_dof_indices_ext[idof]];
        coords_coeff_ext[idof] = val;
        coords_coeff_ext[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff_ext[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff_ext[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdX || compute_d2R) i_derivative++;
    }
    AssertDimension(i_derivative, n_total_indep);

    // Use the metric Jacobian from the interior cell
    std::vector<dealii::Tensor<2,dim,ADADtype>> metric_jac_int = evaluate_metric_jacobian (unit_quad_pts_int, coords_coeff_int, fe_metric);
    std::vector<dealii::Tensor<2,dim,ADADtype>> metric_jac_ext = evaluate_metric_jacobian (unit_quad_pts_ext, coords_coeff_ext, fe_metric);
    std::vector<dealii::Tensor<2,dim,ADADtype>> jac_inv_tran_int(n_face_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADADtype>> jac_inv_tran_ext(n_face_quad_pts);

    const dealii::Tensor<1,dim,ADADtype> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[interior_face_number];
    const dealii::Tensor<1,dim,ADADtype> unit_normal_ext = dealii::GeometryInfo<dim>::unit_normal_vector[exterior_face_number];

    // Use quadrature points of neighbor cell
    // Might want to use the maximum n_quad_pts1 and n_quad_pts2
    //const unsigned int n_face_quad_pts = fe_values_ext.n_quadrature_points;

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

    std::vector<ADADtype> interpolation_operator_int(n_soln_dofs_int);
    std::vector<ADADtype> interpolation_operator_ext(n_soln_dofs_ext);
    std::array<std::vector<ADADtype>,dim> gradient_operator_int, gradient_operator_ext;
    for (int d=0;d<dim;++d) {
        gradient_operator_int[d].resize(n_soln_dofs_int);
        gradient_operator_ext[d].resize(n_soln_dofs_ext);
    }
    const double cell_diameter_int = fe_values_int.get_cell()->diameter();
    const double cell_diameter_ext = fe_values_ext.get_cell()->diameter();
    const ADADtype artificial_diss_coeff_int = this->all_parameters->add_artificial_dissipation ?
                                               this->discontinuity_sensor(cell_diameter_int, soln_coeff_int, fe_values_int.get_fe())
                                               : 0.0;
    const ADADtype artificial_diss_coeff_ext = this->all_parameters->add_artificial_dissipation ?
                                               this->discontinuity_sensor(cell_diameter_ext, soln_coeff_ext, fe_values_ext.get_fe())
                                               : 0.0;

    ADADtype dual_dot_residual = 0.0;
    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        dealii::Tensor<1,dim,ADADtype> normal_normalized_int;
        dealii::Tensor<1,dim,ADADtype> normal_normalized_ext;
        ADADtype surface_jac_det_int;
        ADADtype surface_jac_det_ext;
        if (compute_metric_derivatives) {
            const ADADtype jacobian_determinant_int = dealii::determinant(metric_jac_int[iquad]);
            const ADADtype jacobian_determinant_ext = dealii::determinant(metric_jac_ext[iquad]);

            const dealii::Tensor<2,dim,ADADtype> jacobian_transpose_inverse_int = dealii::transpose(dealii::invert(metric_jac_int[iquad]));
            const dealii::Tensor<2,dim,ADADtype> jacobian_transpose_inverse_ext = dealii::transpose(dealii::invert(metric_jac_ext[iquad]));

            const ADADtype jac_det_int = jacobian_determinant_int;
            const ADADtype jac_det_ext = jacobian_determinant_ext;

            const dealii::Tensor<2,dim,ADADtype> jac_inv_tran_int = jacobian_transpose_inverse_int;
            const dealii::Tensor<2,dim,ADADtype> jac_inv_tran_ext = jacobian_transpose_inverse_ext;

            const dealii::Tensor<1,dim,ADADtype> normal_int = dealii::contract<1,0>(jacobian_transpose_inverse_int, unit_normal_int);
            const dealii::Tensor<1,dim,ADADtype> normal_ext = dealii::contract<1,0>(jacobian_transpose_inverse_ext, unit_normal_ext);
            const ADADtype area_int = normal_int.norm();
            const ADADtype area_ext = normal_ext.norm();

            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.

            normal_normalized_int = normal_int / area_int;
            normal_normalized_ext = -normal_normalized_int;//normal_ext / area_ext; Must use opposite normal to be consistent with explicit

            surface_jac_det_int = area_int*jac_det_int;
            surface_jac_det_ext = area_ext*jac_det_ext;

            for (int d=0;d<dim;++d) {
                //Assert( std::abs(normal_int[d].val().val()+normal_ext[d].val().val()) < 1e-12,
                //    dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
                //        + " N1: " + std::to_string(normal_int[d].val().val())
                //        + " N2: " + std::to_string(normal_ext[d].val().val())));
                Assert( std::abs(normal_normalized_int[d].val().val()+normal_normalized_ext[d].val().val()) < 1e-12,
                    dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
                        + " N1: " + std::to_string(normal_normalized_int[d].val().val())
                        + " N2: " + std::to_string(normal_normalized_ext[d].val().val())));
            }
            Assert( std::abs(surface_jac_det_ext.val().val()-surface_jac_det_int.val().val()) < 1e-12
                    || std::abs(surface_jac_det_ext.val().val()-std::pow(2,dim-1)*surface_jac_det_int.val().val()) < 1e-12 ,
                    dealii::ExcMessage("Inconsistent surface Jacobians. J1: " + std::to_string(surface_jac_det_int.val().val())
                    + " J2: " + std::to_string(surface_jac_det_ext.val().val())));

            for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                interpolation_operator_int[idof] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
                const dealii::Tensor<1,dim,ADADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran_int, fe_int.shape_grad(idof,unit_quad_pts_int[iquad]));
                for (int d=0;d<dim;++d) {
                    gradient_operator_int[d][idof] = phys_shape_grad[d];
                }
            }
            for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                interpolation_operator_ext[idof] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
                const dealii::Tensor<1,dim,ADADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran_ext, fe_ext.shape_grad(idof,unit_quad_pts_ext[iquad]));
                for (int d=0;d<dim;++d) {
                    gradient_operator_ext[d][idof] = phys_shape_grad[d];
                }
            }

            // Exact mapping
            // for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            //     interpolation_operator_int[idof] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
            // }
            // for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
            //     interpolation_operator_ext[idof] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
            // }
            // for (int d=0;d<dim;++d) {
            //     for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            //         const unsigned int istate = fe_int.system_to_component_index(idof).first;
            //         gradient_operator_int[d][idof] = fe_values_int.shape_grad_component(idof, iquad, istate)[d];
            //     }
            //     for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
            //         const unsigned int istate = fe_ext.system_to_component_index(idof).first;
            //         gradient_operator_ext[d][idof] = fe_values_ext.shape_grad_component(idof, iquad, istate)[d];
            //     }
            // }
            // normal_normalized_int = fe_values_int.normal_vector(iquad);
            // normal_normalized_ext = -normal_normalized_int; // Must use opposite normal to be consistent with explicit
            // surface_jac_det_int = fe_values_int.JxW(iquad)/face_quadrature_int.weight(iquad);
            // surface_jac_det_ext = fe_values_ext.JxW(iquad)/face_quadrature_ext.weight(iquad);

        } else {
            for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                interpolation_operator_int[idof] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
            }
            for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                interpolation_operator_ext[idof] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
            }
            for (int d=0;d<dim;++d) {
                for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                    const unsigned int istate = fe_int.system_to_component_index(idof).first;
                    gradient_operator_int[d][idof] = fe_values_int.shape_grad_component(idof, iquad, istate)[d];
                }
                for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                    const unsigned int istate = fe_ext.system_to_component_index(idof).first;
                    gradient_operator_ext[d][idof] = fe_values_ext.shape_grad_component(idof, iquad, istate)[d];
                }
            }
            surface_jac_det_int = fe_values_int.JxW(iquad)/face_quadrature_int.weight(iquad);
            surface_jac_det_ext = fe_values_ext.JxW(iquad)/face_quadrature_ext.weight(iquad);

            normal_normalized_int = fe_values_int.normal_vector(iquad);
            normal_normalized_ext = -normal_normalized_int; // Must use opposite normal to be consistent with explicit
        }

        for (int istate=0; istate<nstate; istate++) { 
            soln_int[istate]      = 0;
            soln_grad_int[istate] = 0;
            soln_ext[istate]      = 0;
            soln_grad_ext[istate] = 0;
        }

        // Interpolate solution to face
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
            const unsigned int istate = fe_int.system_to_component_index(idof).first;
            soln_int[istate]      += soln_coeff_int[idof] * interpolation_operator_int[idof];
            for (int d=0;d<dim;++d) {
                soln_grad_int[istate][d] += soln_coeff_int[idof] * gradient_operator_int[d][idof];
            }
        }
        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
            const unsigned int istate = fe_ext.system_to_component_index(idof).first;
            soln_ext[istate]      += soln_coeff_ext[idof] * interpolation_operator_ext[idof];
            for (int d=0;d<dim;++d) {
                soln_grad_ext[istate][d] += soln_coeff_ext[idof] * gradient_operator_ext[d][idof];
            }
        }

        // Evaluate physical convective flux, physical dissipative flux, and source term
        conv_num_flux_dot_n = conv_num_flux_fad_fad->evaluate_flux(soln_int, soln_ext, normal_normalized_int);
        diss_soln_num_flux = diss_num_flux_fad_fad->evaluate_solution_flux(soln_int, soln_ext, normal_normalized_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            diss_soln_jump_int[s] = (diss_soln_num_flux[s] - soln_int[s]) * normal_normalized_int;
            diss_soln_jump_ext[s] = (diss_soln_num_flux[s] - soln_ext[s]) * normal_normalized_ext;
        }
        diss_flux_jump_int = pde_physics_fad_fad->dissipative_flux (soln_int, diss_soln_jump_int);
        diss_flux_jump_ext = pde_physics_fad_fad->dissipative_flux (soln_ext, diss_soln_jump_ext);

        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = pde_physics_fad_fad->artificial_dissipative_flux (artificial_diss_coeff_int, soln_int, diss_soln_jump_int);
            const ADArrayTensor1 artificial_diss_flux_jump_ext = pde_physics_fad_fad->artificial_dissipative_flux (artificial_diss_coeff_ext, soln_ext, diss_soln_jump_ext);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[s] += artificial_diss_flux_jump_int[s];
                diss_flux_jump_ext[s] += artificial_diss_flux_jump_ext[s];
            }
        }


        diss_auxi_num_flux_dot_n = diss_num_flux_fad_fad->evaluate_auxiliary_flux(
            artificial_diss_coeff_int,
            artificial_diss_coeff_ext,
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_normalized_int, penalty);

        // From test functions associated with interior cell point of view
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            ADADtype rhs = 0.0;
            const unsigned int istate = fe_int.system_to_component_index(itest_int).first;

            const ADADtype JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_int[itest_int] * conv_num_flux_dot_n[istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_int[itest_int] * diss_auxi_num_flux_dot_n[istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_int[d][itest_int] * diss_flux_jump_int[istate][d] * JxW_iquad;
            }

            local_rhs_int_cell(itest_int) += rhs.val().val();

            if (compute_dRdW) {
                // dR_int_dW_int
                std::vector<real> residual_derivatives(n_soln_dofs_int);
                for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                    const unsigned int i_dx = idof+w_int_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, residual_derivatives);

                // dR_int_dW_ext
                residual_derivatives.resize(n_soln_dofs_ext);
                for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                    const unsigned int i_dx = idof+w_ext_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, residual_derivatives);
            }
            if (compute_dRdX) {
                // dR_int_dX_int
                std::vector<real> residual_derivatives(n_metric_dofs);
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_int_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_int, residual_derivatives);

                // dR_int_dX_ext
                // residual_derivatives.resize(n_metric_dofs);
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_ext_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_ext, residual_derivatives);
            }
            if (compute_d2R) {
                const unsigned int global_residual_row = soln_dof_indices_int[itest_int];
                dual_dot_residual += this->dual[global_residual_row]*rhs;
            }
        }

        // From test functions associated with neighbour cell point of view
        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            ADADtype rhs = 0.0;
            const unsigned int istate = fe_ext.system_to_component_index(itest_ext).first;

            const ADADtype JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-conv_num_flux_dot_n[istate]) * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-diss_auxi_num_flux_dot_n[istate]) * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_ext[d][itest_ext] * diss_flux_jump_ext[istate][d] * JxW_iquad;
            }

            local_rhs_ext_cell(itest_ext) += rhs.val().val();

            if (compute_dRdW) {
                // dR_ext_dW_int
                std::vector<real> residual_derivatives(n_soln_dofs_int);
                for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                    const unsigned int i_dx = idof+w_int_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, residual_derivatives);

                // dR_ext_dW_ext
                residual_derivatives.resize(n_soln_dofs_ext);
                for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                    const unsigned int i_dx = idof+w_ext_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, residual_derivatives);
            }
            if (compute_dRdX) {
                // dR_ext_dX_int
                std::vector<real> residual_derivatives(n_metric_dofs);
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_int_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_int, residual_derivatives);

                // dR_ext_dX_ext
                // residual_derivatives.resize(n_metric_dofs);
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_ext_start;
                    residual_derivatives[idof] = rhs.dx(i_dx).val();
                }
                this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_ext, residual_derivatives);
            }
            if (compute_d2R) {
                const unsigned int global_residual_row = soln_dof_indices_ext[itest_ext];
                dual_dot_residual += this->dual[global_residual_row]*rhs;
            }
        }
    } // Quadrature point loop


    if (compute_d2R) {
        std::vector<real> dWidWint(n_soln_dofs_int);
        std::vector<real> dWidWext(n_soln_dofs_ext);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);
        // dWint
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {

            const unsigned int i_dx = idof+w_int_start;
            const ADtype dWi = dual_dot_residual.dx(i_dx);

            // dWint_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidWint[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_int, dWidWint);

            // dWint_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidWext[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_ext, dWidWext);

            // dWint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_int, dWidX);

            // dWint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_ext, dWidX);
        }
        // dWext
        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {

            const unsigned int i_dx = idof+w_ext_start;
            const ADtype dWi = dual_dot_residual.dx(i_dx);

            // dWext_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidWint[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_int, dWidWint);

            // dWext_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidWext[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_ext, dWidWext);

            // dWext_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_int, dWidX);

            // dWext_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_ext, dWidX);
        }

        // dXint
        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_int_start;
            const ADtype dWi = dual_dot_residual.dx(i_dx);

            // dXint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_int, dWidX);

            // dXint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_ext, dWidX);
        }
        // dXext
        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_ext_start;
            const ADtype dWi = dual_dot_residual.dx(i_dx);

            // dXext_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_int, dWidX);

            // dXext_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_ext, dWidX);
        }
    }
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_terms_derivatives(
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADADtype = Sacado::Fad::DFad<ADtype>;
    using ADArray = std::array<ADADtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,ADADtype>, nstate >;

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;

    const unsigned int n_quad_pts      = quadrature.size();
    const unsigned int n_soln_dofs     = fe.dofs_per_cell;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    const std::vector<dealii::Point<dim>> &points = quadrature.get_points ();

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector< ADADtype > coords_coeff(n_metric_dofs);
    std::vector< ADADtype > soln_coeff(n_soln_dofs);

    // Derivatives are ordered such that w comes first with index 0, then x.
    // If derivatives with respect to w are not needed, then derivatives
    // with respect to x will start at index 0.
    unsigned int w_start = 0, w_end = 0, x_start = 0, x_end = 0;
    if (compute_d2R || (compute_dRdW && compute_dRdX)) {
        w_start = 0;
        w_end   = w_start + n_soln_dofs;
        x_start = w_end;
        x_end   = x_start + n_metric_dofs;
    } else if (compute_dRdW) {
        w_start = 0;
        w_end   = w_start + n_soln_dofs;
        x_start = w_end;
        x_end   = x_start + 0;
    } else if (compute_dRdX) {
        w_start = 0;
        w_end   = w_start + 0;
        x_start = w_end;
        x_end   = x_start + n_metric_dofs;
    } else {
        std::cout << "Called the derivative version of the residual without requesting the derivative" << std::endl;
    }

    unsigned int i_derivative = 0;
    const unsigned int n_total_indep = x_end;

    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;
        soln_coeff[idof].val() = val;

        if (compute_dRdW || compute_d2R) soln_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) soln_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdW || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
        coords_coeff[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdX || compute_d2R) i_derivative++;
    }

    AssertDimension(i_derivative, n_total_indep);

    std::vector<dealii::Tensor<2,dim,ADADtype>> metric_jacobian = evaluate_metric_jacobian ( points, coords_coeff, fe_metric);
    std::vector<ADADtype> jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,ADADtype>> jac_inv_tran(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        if (compute_metric_derivatives) {
            const ADADtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
            const dealii::Tensor<2,dim,ADADtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

            jac_det[iquad] = jacobian_determinant;
            jac_inv_tran[iquad] = jacobian_transpose_inverse;

            // Exact mapping
            // jac_det[iquad] = fe_values_vol.JxW(iquad) / quadrature.weight(iquad);
        } else {
            jac_det[iquad] = fe_values_vol.JxW(iquad) / quadrature.weight(iquad);
        }
    }

    std::vector< std::array<ADADtype,nstate> > soln_at_q(n_quad_pts);
    std::vector< std::array< dealii::Tensor<1,dim,ADADtype>, nstate > > conv_phys_flux_at_q(n_quad_pts);

    std::vector< ADArrayTensor1 > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros
    std::vector< ADArrayTensor1 > diss_phys_flux_at_q(n_quad_pts);
    std::vector< std::array<ADADtype,nstate> > source_at_q(n_quad_pts);

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();
    dealii::FullMatrix<real> interpolation_operator(n_soln_dofs,n_quad_pts);
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    // Might want to have the dimension as the innermost index
    // Need a contiguous 2d-array structure
    // std::array<dealii::FullMatrix<ADADtype>,dim> gradient_operator;
    // for (int d=0;d<dim;++d) {
    //     gradient_operator[d].reinit(n_soln_dofs, n_quad_pts);
    // }
    std::array<dealii::Table<2,ADADtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(dealii::TableIndices<2>(n_soln_dofs, n_quad_pts));
    }
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            if (compute_metric_derivatives) {
                const dealii::Tensor<1,dim,ADADtype> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe.shape_grad(idof,points[iquad]));
                for (int d=0;d<dim;++d) {
                    gradient_operator[d][idof][iquad] = phys_shape_grad[d];
                }

                // Exact mapping
                // for (int d=0;d<dim;++d) {
                //     const unsigned int istate = fe.system_to_component_index(idof).first;
                //     gradient_operator[d][idof][iquad] = fe_values_vol.shape_grad_component(idof, iquad, istate)[d];
                // }
            } else {
                for (int d=0;d<dim;++d) {
                    const unsigned int istate = fe.system_to_component_index(idof).first;
                    gradient_operator[d][idof][iquad] = fe_values_vol.shape_grad_component(idof, iquad, istate)[d];
                }
            }
        }
    }

    const double cell_diameter = fe_values_vol.get_cell()->diameter();
    const ADADtype artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                           this->discontinuity_sensor(cell_diameter, soln_coeff, fe_values_vol.get_fe())
                                           : 0.0;

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) { 
            // Interpolate solution to the face quadrature points
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
            const unsigned int istate = fe.system_to_component_index(idof).first;
            soln_at_q[iquad][istate]      += soln_coeff[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_at_q[iquad][istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }
        conv_phys_flux_at_q[iquad] = pde_physics_fad_fad->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = pde_physics_fad_fad->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_phys_flux_at_q = pde_physics_fad_fad->artificial_dissipative_flux (artificial_diss_coeff, soln_at_q[iquad], soln_grad_at_q[iquad]);
            for (int s=0; s<nstate; s++) { 
                diss_phys_flux_at_q[iquad][s] += artificial_diss_phys_flux_at_q[s];
            }
        }

        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            dealii::Point<dim,ADADtype> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = 0.0;}
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                ad_point[iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }
            source_at_q[iquad] = pde_physics_fad_fad->source_term (ad_point, soln_at_q[iquad]);
            //std::array<ADADtype,nstate> artificial_source_at_q = pde_physics_fad_fad->artificial_source_term (artificial_diss_coeff, ad_point, soln_at_q[iquad]);
            //for (int s=0;s<nstate;++s) source_at_q[iquad][s] += artificial_source_at_q[s];
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
    ADADtype dual_dot_residual = 0.0;
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        ADADtype rhs = 0;

        const unsigned int istate = fe.system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const ADADtype JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

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

        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = rhs.dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = rhs.dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        if (compute_d2R) {
            const unsigned int global_residual_row = soln_dof_indices[itest];
            dual_dot_residual += this->dual[global_residual_row]*rhs;
        }

        local_rhs_cell(itest) += rhs.val().val();

    }


    if (compute_d2R) {

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;
            const ADtype dWi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = dWi.dx(j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;
            const ADtype dXi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = dXi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }
    }

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

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::set_physics(
    std::shared_ptr< Physics::PhysicsBase<dim, nstate, Sacado::Fad::DFad<Sacado::Fad::DFad<real>> > >pde_physics_input)
{
    using ADtype = Sacado::Fad::DFad<real>;
    using ADADtype = Sacado::Fad::DFad<ADtype>;
    pde_physics_fad_fad = pde_physics_input;
    conv_num_flux_fad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, ADADtype> ::create_convective_numerical_flux (DGBase<dim,real>::all_parameters->conv_num_flux_type, pde_physics_fad_fad);
    diss_num_flux_fad_fad = NumericalFlux::NumericalFluxFactory<dim, nstate, ADADtype> ::create_dissipative_numerical_flux (DGBase<dim,real>::all_parameters->diss_num_flux_type, pde_physics_fad_fad);
}

template class DGWeak <PHILIP_DIM, 1, double>;
template class DGWeak <PHILIP_DIM, 2, double>;
template class DGWeak <PHILIP_DIM, 3, double>;
template class DGWeak <PHILIP_DIM, 4, double>;
template class DGWeak <PHILIP_DIM, 5, double>;

} // PHiLiP namespace

