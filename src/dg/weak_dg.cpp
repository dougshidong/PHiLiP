#include <deal.II/base/tensor.h>
#include <deal.II/base/table.h>

#include <deal.II/base/qprojector.h>

//#include <deal.II/lac/full_matrix.templates.h>
#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/vector.h>

#include "ADTypes.hpp"

#include "weak_dg.hpp"

//#define FADFAD
template<typename real>
double getValue(const real &x) {
    if constexpr(std::is_same<real,double>::value) {
        return x;
    } else {
        return getValue(x.value());
    }
}
/// Returns y = Ax.
/** Had to rewrite this instead of 
 *  dealii::contract<1,0>(A,x);
 *  because contract doesn't allow the use of codi variables.
 */
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
    const std::shared_ptr<Triangulation> triangulation_input)
    : DGBaseState<dim,nstate,real>::DGBaseState(parameters_input, degree, max_degree_input, grid_degree_input, triangulation_input)
{ }
// Destructor
template <int dim, int nstate, typename real>
DGWeak<dim,nstate,real>::~DGWeak ()
{
    pcout << "Destructing DGWeak..." << std::endl;
}


/// Derivative indexing when only 1 cell is concerned.
/// Derivatives are ordered such that w comes first with index 0, then x.
/// If derivatives with respect to w are not needed, then derivatives
/// with respect to x will start at index 0. This function is for a single
/// cell's DoFs.
void automatic_differentiation_indexing_1(
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
    const unsigned int n_soln_dofs, const unsigned int n_metric_dofs,
    unsigned int &w_start, unsigned int &w_end,
    unsigned int &x_start, unsigned int &x_end)
{
    w_start = 0;
    w_end = 0;
    x_start = 0;
    x_end = 0;
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
}

/// Derivative indexing when 2 cells are concerned.
/// Derivatives are ordered such that w comes first with index 0, then x.
/// If derivatives with respect to w are not needed, then derivatives
/// with respect to x will start at index 0. This function is for a single
/// cell's DoFs.
void automatic_differentiation_indexing_2(
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R,
    const unsigned int n_soln_dofs_int, const unsigned int n_soln_dofs_ext, const unsigned int n_metric_dofs,
    unsigned int &w_int_start, unsigned int &w_int_end, unsigned int &w_ext_start, unsigned int &w_ext_end,
    unsigned int &x_int_start, unsigned int &x_int_end, unsigned int &x_ext_start, unsigned int &x_ext_end)
{
    // Current derivative order is: soln_int, soln_ext, metric_int, metric_ext
    w_int_start = 0; w_int_end = 0; w_ext_start = 0; w_ext_end = 0;
    x_int_start = 0; x_int_end = 0; x_ext_start = 0; x_ext_end = 0;
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
}


template <int dim, typename real>
std::vector<dealii::Tensor<2,dim,real>> evaluate_metric_jacobian (
    const std::vector<dealii::Point<dim>> &points,
    const std::vector<real> &coords_coeff,
    const dealii::FESystem<dim,dim> &fe_metric)
{
    const unsigned int n_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_pts = points.size();

    AssertDimension(n_dofs, coords_coeff.size());

    std::vector<dealii::Tensor<2,dim,real>> metric_jacobian(n_pts);

    for (unsigned int ipoint=0; ipoint<n_pts; ++ipoint) {
        std::array< dealii::Tensor<1,dim,real>, dim > coords_grad;
        for (int d=0; d<dim; ++d) {
            coords_grad[d] = 0.0;
        }
        for (unsigned int idof=0; idof<n_dofs; ++idof) {
            const unsigned int axis = fe_metric.system_to_component_index(idof).first;
            //coords_grad[axis] += coords_coeff[idof] * fe_metric.shape_grad (idof, points[ipoint]);

            dealii::Tensor<1,dim,real> shape_grad = fe_metric.shape_grad (idof, points[ipoint]);
            for (int d=0; d<dim; ++d) {
                coords_grad[axis][d] += coords_coeff[idof] * shape_grad[d];
            }
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
void DGWeak<dim,nstate,real>::assemble_volume_term_explicit(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/)
{
    (void) current_cell_index;
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

    //const real cell_diameter = fe_values_vol.get_cell()->diameter();
    real cell_diameter = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        cell_diameter = cell_diameter + JxW[iquad];
    }
    //const real artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
    //                                   this->discontinuity_sensor(cell_diameter, soln_coeff, fe_values_vol.get_fe())
    //                                   : 0.0;
    const real artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                       this->artificial_dissipation_coeffs[current_cell_index]
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
        conv_phys_flux_at_q[iquad] = DGBaseState<dim,nstate,real>::pde_physics_double->convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = DGBaseState<dim,nstate,real>::pde_physics_double->dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);
        if(this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_phys_flux_at_q = DGBaseState<dim,nstate,real>::pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff, soln_at_q[iquad], soln_grad_at_q[iquad]);
            for (int istate=0; istate<nstate; istate++) {
                diss_phys_flux_at_q[iquad][istate] += artificial_diss_phys_flux_at_q[istate];
            }
        }
        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            const dealii::Point<dim,real> point = fe_values_vol.quadrature_point(iquad);
            source_at_q[iquad] = DGBaseState<dim,nstate,real>::pde_physics_double->source_term (point, soln_at_q[iquad]);
            //std::array<real,nstate> artificial_source_at_q = DGBaseState<dim,nstate,real>::pde_physics_double->artificial_source_term (artificial_diss_coeff, point, soln_at_q[iquad]);
            //for (int s=0;s<nstate;++s) source_at_q[iquad][s] += artificial_source_at_q[s];
        }
    }

    const unsigned int cell_index = fe_values_vol.get_cell()->active_cell_index();
    this->max_dt_cell[cell_index] = DGBaseState<dim,nstate,real>::evaluate_CFL ( soln_at_q, artificial_diss_coeff, cell_diameter );

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
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    dealii::Vector<real> &local_rhs_int_cell)
{
    (void) current_cell_index;
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

    //const real cell_diameter = fe_values_boundary.get_cell()->diameter();
    //const real artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
    //                                   this->discontinuity_sensor(cell_diameter, soln_coeff_int, fe_values_boundary.get_fe())
    //                                   : 0.0;
    const real artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                       this->artificial_dissipation_coeffs[current_cell_index]
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
        DGBaseState<dim,nstate,real>::pde_physics_double->boundary_face_values (boundary_id, real_quad_point, normal_int, soln_int[iquad], soln_grad_int[iquad], soln_ext[iquad], soln_grad_ext[iquad]);

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
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real>::conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real>::diss_num_flux_double->evaluate_solution_flux(soln_ext[iquad], soln_ext[iquad], normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
            }
        }
        diss_flux_jump_int[iquad] = DGBaseState<dim,nstate,real>::pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = DGBaseState<dim,nstate,real>::pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff, soln_int[iquad], diss_soln_jump_int);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real>::diss_num_flux_double->evaluate_auxiliary_flux(
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
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_int,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices_ext,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell)
{
    (void) current_cell_index;
    (void) neighbor_cell_index;
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

    //const real cell_diameter_int = fe_values_int.get_cell()->diameter();
    //const real cell_diameter_ext = fe_values_ext.get_cell()->diameter();
    //const real artificial_diss_coeff_int = this->all_parameters->add_artificial_dissipation ?
    //                                       this->discontinuity_sensor(cell_diameter_int, soln_coeff_int, fe_values_int.get_fe())
    //                                       : 0.0;
    //const real artificial_diss_coeff_ext = this->all_parameters->add_artificial_dissipation ?
    //                                       this->discontinuity_sensor(cell_diameter_ext, soln_coeff_ext, fe_values_ext.get_fe())
    //                                       : 0.0;
    const real artificial_diss_coeff_int = this->all_parameters->add_artificial_dissipation ?
                                           this->artificial_dissipation_coeffs[current_cell_index]
                                           : 0.0;
    const real artificial_diss_coeff_ext = this->all_parameters->add_artificial_dissipation ?
                                           this->artificial_dissipation_coeffs[neighbor_cell_index]
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
        conv_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real>::conv_num_flux_double->evaluate_flux(soln_int[iquad], soln_ext[iquad], normal_int);
        diss_soln_num_flux[iquad] = DGBaseState<dim,nstate,real>::diss_num_flux_double->evaluate_solution_flux(soln_int[iquad], soln_ext[iquad], normal_int);

        doubleArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[iquad][s]) * normal_int[d];
                diss_soln_jump_ext[s][d] = (diss_soln_num_flux[iquad][s] - soln_ext[iquad][s]) * normal_ext[d];
            }
        }
        diss_flux_jump_int[iquad] = DGBaseState<dim,nstate,real>::pde_physics_double->dissipative_flux (soln_int[iquad], diss_soln_jump_int);
        diss_flux_jump_ext[iquad] = DGBaseState<dim,nstate,real>::pde_physics_double->dissipative_flux (soln_ext[iquad], diss_soln_jump_ext);

        if (this->all_parameters->add_artificial_dissipation) {
            const doubleArrayTensor1 artificial_diss_flux_jump_int = DGBaseState<dim,nstate,real>::pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff_int, soln_int[iquad], diss_soln_jump_int);
            const doubleArrayTensor1 artificial_diss_flux_jump_ext = DGBaseState<dim,nstate,real>::pde_physics_double->artificial_dissipative_flux (artificial_diss_coeff_ext, soln_ext[iquad], diss_soln_jump_ext);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
                diss_flux_jump_ext[iquad][s] += artificial_diss_flux_jump_ext[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = DGBaseState<dim,nstate,real>::diss_num_flux_double->evaluate_auxiliary_flux(
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

    // From test functions associated with neighbor cell point of view
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
template <typename adtype>
void DGWeak<dim,nstate,real>::assemble_boundary(
    const dealii::types::global_dof_index current_cell_index,
    const std::vector< adtype > &soln_coeff,
    const std::vector< adtype > &coords_coeff,
    const std::vector< real > &local_dual,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const Physics::PhysicsBase<dim, nstate, adtype> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::Quadrature<dim-1> &quadrature,
    std::vector<adtype> &rhs,
    adtype &dual_dot_residual,
    const bool compute_metric_derivatives)
{
    const unsigned int n_soln_dofs = fe_soln.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_quad_pts = fe_values_boundary.n_quadrature_points;

    dual_dot_residual = 0.0;
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        rhs[itest] = 0.0;
    }

    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const dealii::Quadrature<dim> face_quadrature
        = dealii::QProjector<dim>::project_to_face(
            dealii::ReferenceCell::get_hypercube(dim),
            quadrature,
            face_number);
    const std::vector<dealii::Point<dim,real>> &unit_quad_pts = face_quadrature.get_points();
    std::vector<dealii::Point<dim,adtype>> real_quad_pts(unit_quad_pts.size());

    std::vector<dealii::Tensor<2,dim,adtype>> metric_jacobian = evaluate_metric_jacobian (unit_quad_pts, coords_coeff, fe_metric);
    std::vector<adtype> jac_det(n_quad_pts);
    std::vector<adtype> surface_jac_det(n_quad_pts);
    std::vector<dealii::Tensor<2,dim,adtype>> jac_inv_tran(n_quad_pts);

    const dealii::Tensor<1,dim,real> unit_normal = dealii::GeometryInfo<dim>::unit_normal_vector[face_number];
    std::vector<dealii::Tensor<1,dim,adtype>> normals(n_quad_pts);

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        if (compute_metric_derivatives) {
            for (int d=0;d<dim;++d) { real_quad_pts[iquad][d] = 0;}
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                real_quad_pts[iquad][iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }

            const adtype jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
            const dealii::Tensor<2,dim,adtype> jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

            jac_det[iquad] = jacobian_determinant;
            jac_inv_tran[iquad] = jacobian_transpose_inverse;

            const dealii::Tensor<1,dim,adtype> normal = vmult(jacobian_transpose_inverse, unit_normal);
            const adtype area = norm(normal);

            surface_jac_det[iquad] = norm(normal)*jac_det[iquad];
            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.
            for (int d=0;d<dim;++d) { 
                normals[iquad][d] = normal[d] / area;
            }

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
            interpolation_operator[idof][iquad] = fe_soln.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    //std::array<dealii::FullMatrix<adtype>,dim> gradient_operator;
    // for (int d=0;d<dim;++d) {
    //     gradient_operator[d].reinit(n_soln_dofs, n_quad_pts);
    // }
    std::array<dealii::Table<2,adtype>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(dealii::TableIndices<2>(n_soln_dofs, n_quad_pts));
    }
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            if (compute_metric_derivatives) {
                const dealii::Tensor<1,dim,real> ref_shape_grad = fe_soln.shape_grad(idof,unit_quad_pts[iquad]);
                const dealii::Tensor<1,dim,adtype> phys_shape_grad = vmult(jac_inv_tran[iquad], ref_shape_grad);
                for (int d=0;d<dim;++d) {
                    gradient_operator[d][idof][iquad] = phys_shape_grad[d];
                }

                // Exact mapping
                // for (int d=0;d<dim;++d) {
                //     const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                //     gradient_operator[d][idof][iquad] = fe_values_boundary.shape_grad_component(idof, iquad, istate)[d];
                // }
            } else {
                for (int d=0;d<dim;++d) {
                    const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                    gradient_operator[d][idof][iquad] = fe_values_boundary.shape_grad_component(idof, iquad, istate)[d];
                }
            }
        }
    }

    //const adtype cell_diameter = fe_values_boundary.get_cell()->diameter();
    //const adtype artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
    //                                       this->discontinuity_sensor(cell_diameter, soln_coeff, fe_values_boundary.get_fe())
    //                                       : 0.0;
    const adtype artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                         this->artificial_dissipation_coeffs[current_cell_index]
                                         : 0.0;

    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        const dealii::Tensor<1,dim,adtype> normal_int = normals[iquad];

        std::array<adtype,nstate> soln_int;
        std::array<adtype,nstate> soln_ext;
        std::array< dealii::Tensor<1,dim,adtype>, nstate > soln_grad_int;
        std::array< dealii::Tensor<1,dim,adtype>, nstate > soln_grad_ext;
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

        physics.boundary_face_values (boundary_id, real_quad_pts[iquad], normal_int, soln_int, soln_grad_int, soln_ext, soln_grad_ext);

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
        conv_num_flux_dot_n[iquad] = conv_num_flux.evaluate_flux(soln_int, soln_ext, normal_int);
        // Notice that the flux uses the solution given by the Dirichlet or Neumann boundary condition
        diss_soln_num_flux[iquad] = diss_num_flux.evaluate_solution_flux(soln_ext, soln_ext, normal_int);

        ADArrayTensor1 diss_soln_jump_int;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[iquad][s] - soln_int[s]) * normal_int[d];
            }
        }
        diss_flux_jump_int[iquad] = physics.dissipative_flux (soln_int, diss_soln_jump_int);

        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = physics.artificial_dissipative_flux (artificial_diss_coeff, soln_int, diss_soln_jump_int);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[iquad][s] += artificial_diss_flux_jump_int[s];
            }
        }

        diss_auxi_num_flux_dot_n[iquad] = diss_num_flux.evaluate_auxiliary_flux(
            artificial_diss_coeff,
            artificial_diss_coeff,
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_int, penalty, true);
    }

    // Applying convection boundary condition
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        adtype rhs_val = 0.0;

        const unsigned int istate = fe_values_boundary.get_fe().system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const adtype JxW_iquad = surface_jac_det[iquad] * face_quadrature.weight(iquad);
            // Convection
            rhs_val = rhs_val - interpolation_operator[itest][iquad] * conv_num_flux_dot_n[iquad][istate] * JxW_iquad;
            // Diffusive
            rhs_val = rhs_val - interpolation_operator[itest][iquad] * diss_auxi_num_flux_dot_n[iquad][istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs_val = rhs_val + gradient_operator[d][itest][iquad] * diss_flux_jump_int[iquad][istate][d] * JxW_iquad;
            }
        }


        rhs[itest] = rhs_val;
        dual_dot_residual += local_dual[itest]*rhs_val;
    }
}

#ifdef FADFAD
template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_boundary_term_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    using adtype = FadFadType;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_soln_dofs = fe_values_boundary.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;
    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    std::vector< adtype > soln_coeff(n_soln_dofs);
    std::vector< adtype > coords_coeff(n_metric_dofs);

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

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
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
        coords_coeff[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdX || compute_d2R) i_derivative++;
    }

    AssertDimension(i_derivative, n_total_indep);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_dual[itest] = this->dual[soln_dof_indices[itest]];
    }

    const auto &physics = *(DGBaseState<dim,nstate,real>::pde_physics_fad_fad);
    const auto &conv_num_flux = *(DGBaseState<dim,nstate,real>::conv_num_flux_fad_fad);
    const auto &diss_num_flux = *(DGBaseState<dim,nstate,real>::diss_num_flux_fad_fad);

    std::vector<adtype> rhs(n_soln_dofs);
    adtype dual_dot_residual;
    assemble_boundary(
        current_cell_index,
        soln_coeff,
        coords_coeff,
        local_dual,
        face_number,
        boundary_id,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_boundary,
        penalty,
        fe_soln,
        fe_metric,
        quadrature,
        rhs,
        dual_dot_residual,
        compute_metric_derivatives);

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell[itest] += rhs[itest].val().val();
    }

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
                AssertIsFinite(residual_derivatives[idof]);
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }

    }

    if (compute_d2R) {
        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

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
            const FadType dXi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = dXi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }
    }
}
#endif

template <int dim, int nstate, typename real>
template <typename adtype>
void DGWeak<dim,nstate,real>::assemble_boundary_codi_taped_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    const Physics::PhysicsBase<dim, nstate, adtype> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_soln_dofs = fe_values_boundary.dofs_per_cell;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;
    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    std::vector< adtype > soln_coeff(n_soln_dofs);
    std::vector< adtype > coords_coeff(n_metric_dofs);

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

    using TH = codi::TapeHelper<adtype>;
    TH th;
    adtype::getGlobalTape();
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.startRecording();
    }
    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;

        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;

        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff[idof]);
        }
    }

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_dual[itest] = this->dual[soln_dof_indices[itest]];
    }

    std::vector<adtype> rhs(n_soln_dofs);
    adtype dual_dot_residual;
    assemble_boundary(
        current_cell_index,
        soln_coeff,
        coords_coeff,
        local_dual,
        face_number,
        boundary_id,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_boundary,
        penalty,
        fe_soln,
        fe_metric,
        quadrature,
        rhs,
        dual_dot_residual,
        compute_metric_derivatives);

    if (compute_dRdW || compute_dRdX) {
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            th.registerOutput(rhs[itest]);
        }
    } else if (compute_d2R) {
        th.registerOutput(dual_dot_residual);
    }
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.stopRecording();
    }

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell(itest) += getValue<adtype>(rhs[itest]);
        AssertIsFinite(local_rhs_cell(itest));
    }

    if (compute_dRdW) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = jac(itest,i_dx);
                AssertIsFinite(residual_derivatives[idof]);
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
        th.deleteJacobian(jac);

    }

    if (compute_dRdX) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = jac(itest,i_dx);
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        th.deleteJacobian(jac);
    }


    if (compute_d2R) {
        typename TH::HessianType& hes = th.createHessian();
        th.evalHessian(hes);

        int i_dependent = (compute_dRdW || compute_dRdX) ? n_soln_dofs : 0;

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }

        th.deleteHessian(hes);
    }

}

#ifndef FADFAD
template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_boundary_term_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const unsigned int face_number,
    const unsigned int boundary_id,
    const dealii::FEFaceValuesBase<dim,dim> &fe_values_boundary,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim-1> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    if (compute_d2R) {
        assemble_boundary_codi_taped_derivatives<codi_HessianComputationType>(
            current_cell_index,
            face_number,
            boundary_id,
            fe_values_boundary,
            penalty,
            fe_soln,
            quadrature,
            metric_dof_indices,
            soln_dof_indices,
            *(DGBaseState<dim,nstate,real>::pde_physics_rad_fad),
            *(DGBaseState<dim,nstate,real>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nstate,real>::diss_num_flux_rad_fad),
            local_rhs_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else if (compute_dRdW || compute_dRdX) {
        assemble_boundary_codi_taped_derivatives<codi_JacobianComputationType>(
            current_cell_index,
            face_number,
            boundary_id,
            fe_values_boundary,
            penalty,
            fe_soln,
            quadrature,
            metric_dof_indices,
            soln_dof_indices,
            *(DGBaseState<dim,nstate,real>::pde_physics_rad),
            *(DGBaseState<dim,nstate,real>::conv_num_flux_rad),
            *(DGBaseState<dim,nstate,real>::diss_num_flux_rad),
            local_rhs_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    }
}
#endif


template <int dim, int nstate, typename real>
template <typename real2>
void DGWeak<dim,nstate,real>::assemble_face_term(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const std::vector< real2 > &soln_coeff_int,
    const std::vector< real2 > &soln_coeff_ext,
    const std::vector< real2 > &coords_coeff_int,
    const std::vector< real2 > &coords_coeff_ext,
    const std::vector< double > &dual_int,
    const std::vector< double > &dual_ext,
    const unsigned int interior_face_number,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, real2> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, real2> &diss_num_flux,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_int,
    const dealii::FEFaceValuesBase<dim,dim>     &fe_values_ext,
    const real penalty,
    const dealii::FESystem<dim,dim> &fe_int,
    const dealii::FESystem<dim,dim> &fe_ext,
    const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::Quadrature<dim> &face_quadrature_int,
    const dealii::Quadrature<dim> &face_quadrature_ext,
    std::vector<real2> &rhs_int,
    std::vector<real2> &rhs_ext,
    real2 &dual_dot_residual,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) compute_dRdW;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    dual_dot_residual = 0.0;
    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        rhs_int[itest] = 0.0;
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        rhs_ext[itest] = 0.0;
    }

    using ADArray = std::array<real2,nstate>;
    using Tensor1D = dealii::Tensor<1,dim,real2>;
    using Tensor2D = dealii::Tensor<2,dim,real2>;
    using ADArrayTensor1 = std::array< Tensor1D, nstate >;

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;

    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_int = face_quadrature_int.get_points();
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts_ext = face_quadrature_ext.get_points();

    const unsigned int n_face_quad_pts = unit_quad_pts_int.size();


    // Use the metric Jacobian from the interior cell
    std::vector<Tensor2D> metric_jac_int = evaluate_metric_jacobian (unit_quad_pts_int, coords_coeff_int, fe_metric);
    std::vector<Tensor2D> metric_jac_ext = evaluate_metric_jacobian (unit_quad_pts_ext, coords_coeff_ext, fe_metric);
    std::vector<Tensor2D> jac_inv_tran_int(n_face_quad_pts);
    std::vector<Tensor2D> jac_inv_tran_ext(n_face_quad_pts);

    const dealii::Tensor<1,dim,real> unit_normal_int = dealii::GeometryInfo<dim>::unit_normal_vector[interior_face_number];
    const dealii::Tensor<1,dim,real> unit_normal_ext = -unit_normal_int;

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

    std::vector<real> interpolation_operator_int(n_soln_dofs_int);
    std::vector<real> interpolation_operator_ext(n_soln_dofs_ext);
    std::array<std::vector<real2>,dim> gradient_operator_int, gradient_operator_ext;
    for (int d=0;d<dim;++d) {
        gradient_operator_int[d].resize(n_soln_dofs_int);
        gradient_operator_ext[d].resize(n_soln_dofs_ext);
    }
    //const real2 cell_diameter_int = fe_values_int.get_cell()->diameter();
    //const real2 cell_diameter_ext = fe_values_ext.get_cell()->diameter();
    //const real2 artificial_diss_coeff_int = this->all_parameters->add_artificial_dissipation ?
    //                                        this->discontinuity_sensor(cell_diameter_int, soln_coeff_int, fe_values_int.get_fe())
    //                                        : 0.0;
    //const real2 artificial_diss_coeff_ext = this->all_parameters->add_artificial_dissipation ?
    //                                        this->discontinuity_sensor(cell_diameter_ext, soln_coeff_ext, fe_values_ext.get_fe())
    //                                        : 0.0;
    const real2 artificial_diss_coeff_int = this->all_parameters->add_artificial_dissipation ?
                                            this->artificial_dissipation_coeffs[current_cell_index]
                                            : 0.0;
    const real2 artificial_diss_coeff_ext = this->all_parameters->add_artificial_dissipation ?
                                            this->artificial_dissipation_coeffs[neighbor_cell_index]
                                            : 0.0;

    for (unsigned int iquad=0; iquad<n_face_quad_pts; ++iquad) {

        Tensor1D normal_normalized_int;
        Tensor1D normal_normalized_ext;
        real2 surface_jac_det_int;
        real2 surface_jac_det_ext;
        if (compute_metric_derivatives) {
            const real2 jacobian_determinant_int = dealii::determinant(metric_jac_int[iquad]);
            const real2 jacobian_determinant_ext = dealii::determinant(metric_jac_ext[iquad]);

            const Tensor2D jacobian_transpose_inverse_int = dealii::transpose(dealii::invert(metric_jac_int[iquad]));
            const Tensor2D jacobian_transpose_inverse_ext = dealii::transpose(dealii::invert(metric_jac_ext[iquad]));

            const real2 jac_det_int = jacobian_determinant_int;
            const real2 jac_det_ext = jacobian_determinant_ext;

            const Tensor2D jac_inv_tran_int = jacobian_transpose_inverse_int;
            const Tensor2D jac_inv_tran_ext = jacobian_transpose_inverse_ext;

            const Tensor1D normal_int = vmult(jacobian_transpose_inverse_int, unit_normal_int);
            const Tensor1D normal_ext = vmult(jacobian_transpose_inverse_ext, unit_normal_ext);
            const real2 area_int = norm(normal_int);
            const real2 area_ext = norm(normal_ext);

            // Technically the normals have jac_det multiplied.
            // However, we use normalized normals by convention, so the the term
            // ends up appearing in the surface jacobian.

            for (int d=0;d<dim;++d) {
                normal_normalized_int[d] = normal_int[d] / area_int;
            }
            normal_normalized_ext = -normal_normalized_int;//normal_ext / area_ext; Must use opposite normal to be consistent with explicit

            surface_jac_det_int = area_int*jac_det_int;
            surface_jac_det_ext = area_ext*jac_det_ext;

            //if (std::is_same<double,real2>::value) {
            //    for (int d=0;d<dim;++d) {
            //        Assert( std::abs(normal_normalized_int[d].val().val()+normal_normalized_ext[d].val().val()) < 1e-12,
            //            dealii::ExcMessage("Inconsistent normals. Direction " + std::to_string(d)
            //                + " N1: " + std::to_string(normal_normalized_int[d].val().val())
            //                + " N2: " + std::to_string(normal_normalized_ext[d].val().val())));
            //    }
            //    Assert( std::abs(surface_jac_det_ext.val().val()-surface_jac_det_int.val().val()) < 1e-12
            //            || std::abs(surface_jac_det_ext.val().val()-std::pow(2,dim-1)*surface_jac_det_int.val().val()) < 1e-12 ,
            //            dealii::ExcMessage("Inconsistent surface Jacobians. J1: " + std::to_string(surface_jac_det_int.val().val())
            //            + " J2: " + std::to_string(surface_jac_det_ext.val().val())));
            //}

            for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {
                interpolation_operator_int[idof] = fe_int.shape_value(idof,unit_quad_pts_int[iquad]);
                dealii::Tensor<1,dim,real> ref_shape_grad = fe_int.shape_grad(idof,unit_quad_pts_int[iquad]);
                const Tensor1D phys_shape_grad = vmult(jac_inv_tran_int, ref_shape_grad);
                for (int d=0;d<dim;++d) {
                    gradient_operator_int[d][idof] = phys_shape_grad[d];
                }
            }
            for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {
                interpolation_operator_ext[idof] = fe_ext.shape_value(idof,unit_quad_pts_ext[iquad]);
                dealii::Tensor<1,dim,real> ref_shape_grad = fe_ext.shape_grad(idof,unit_quad_pts_ext[iquad]);
                const Tensor1D phys_shape_grad = vmult(jac_inv_tran_ext, ref_shape_grad);
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
        conv_num_flux_dot_n = conv_num_flux.evaluate_flux(soln_int, soln_ext, normal_normalized_int);
        diss_soln_num_flux = diss_num_flux.evaluate_solution_flux(soln_int, soln_ext, normal_normalized_int);

        ADArrayTensor1 diss_soln_jump_int, diss_soln_jump_ext;
        for (int s=0; s<nstate; s++) {
            for (int d=0; d<dim; d++) {
                diss_soln_jump_int[s][d] = (diss_soln_num_flux[s] - soln_int[s]) * normal_normalized_int[d];
                diss_soln_jump_ext[s][d] = (diss_soln_num_flux[s] - soln_ext[s]) * normal_normalized_ext[d];
            }
        }
        diss_flux_jump_int = physics.dissipative_flux (soln_int, diss_soln_jump_int);
        diss_flux_jump_ext = physics.dissipative_flux (soln_ext, diss_soln_jump_ext);

        if (this->all_parameters->add_artificial_dissipation) {
            const ADArrayTensor1 artificial_diss_flux_jump_int = physics.artificial_dissipative_flux (artificial_diss_coeff_int, soln_int, diss_soln_jump_int);
            const ADArrayTensor1 artificial_diss_flux_jump_ext = physics.artificial_dissipative_flux (artificial_diss_coeff_ext, soln_ext, diss_soln_jump_ext);
            for (int s=0; s<nstate; s++) {
                diss_flux_jump_int[s] += artificial_diss_flux_jump_int[s];
                diss_flux_jump_ext[s] += artificial_diss_flux_jump_ext[s];
            }
        }


        diss_auxi_num_flux_dot_n = diss_num_flux.evaluate_auxiliary_flux(
            artificial_diss_coeff_int,
            artificial_diss_coeff_ext,
            soln_int, soln_ext,
            soln_grad_int, soln_grad_ext,
            normal_normalized_int, penalty);

        // From test functions associated with interior cell point of view
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            real2 rhs = 0.0;
            const unsigned int istate = fe_int.system_to_component_index(itest_int).first;

            const real2 JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_int[itest_int] * conv_num_flux_dot_n[istate] * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_int[itest_int] * diss_auxi_num_flux_dot_n[istate] * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_int[d][itest_int] * diss_flux_jump_int[istate][d] * JxW_iquad;
            }

            rhs_int[itest_int] += rhs;
            dual_dot_residual += dual_int[itest_int]*rhs;
        }

        // From test functions associated with neighbor cell point of view
        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            real2 rhs = 0.0;
            const unsigned int istate = fe_ext.system_to_component_index(itest_ext).first;

            const real2 JxW_iquad = surface_jac_det_int * face_quadrature_int.weight(iquad);
            // Convection
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-conv_num_flux_dot_n[istate]) * JxW_iquad;
            // Diffusive
            rhs = rhs - interpolation_operator_ext[itest_ext] * (-diss_auxi_num_flux_dot_n[istate]) * JxW_iquad;
            for (int d=0;d<dim;++d) {
                rhs = rhs + gradient_operator_ext[d][itest_ext] * diss_flux_jump_ext[istate][d] * JxW_iquad;
            }

            rhs_ext[itest_ext] += rhs;
            dual_dot_residual += dual_ext[itest_ext]*rhs;
        }
    } // Quadrature point loop

}

#ifdef FADFAD
template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int interior_face_number,
    const unsigned int /*exterior_face_number*/,
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
    (void) current_cell_index;
    (void) neighbor_cell_index;
    using adtype = FadFadType;
    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    std::vector< adtype > coords_coeff_int(n_metric_dofs);
    std::vector< adtype > coords_coeff_ext(n_metric_dofs);
    std::vector< adtype > soln_coeff_int(n_soln_dofs_int);
    std::vector< adtype > soln_coeff_ext(n_soln_dofs_ext);

    // Current derivative ordering is: soln_int, soln_ext, metric_int, metric_ext
    unsigned int w_int_start, w_int_end, w_ext_start, w_ext_end,
                 x_int_start, x_int_end, x_ext_start, x_ext_end;
    automatic_differentiation_indexing_2(
        compute_dRdW, compute_dRdX, compute_d2R,
        n_soln_dofs_int, n_soln_dofs_ext, n_metric_dofs,
        w_int_start, w_int_end, w_ext_start, w_ext_end,
        x_int_start, x_int_end, x_ext_start, x_ext_end);

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
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices_int[idof]];
        coords_coeff_int[idof] = val;
        coords_coeff_int[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff_int[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff_int[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdX || compute_d2R) i_derivative++;
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices_ext[idof]];
        coords_coeff_ext[idof] = val;
        coords_coeff_ext[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff_ext[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff_ext[idof].val().diff(i_derivative, n_total_indep);
        if (compute_dRdX || compute_d2R) i_derivative++;
    }
    AssertDimension(i_derivative, n_total_indep);

    std::vector<double> dual_int(n_soln_dofs_int);
    std::vector<double> dual_ext(n_soln_dofs_ext);

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_int[itest];
        dual_int[itest] = this->dual[global_residual_row];
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_ext[itest];
        dual_ext[itest] = this->dual[global_residual_row];
    }

    std::vector<adtype> rhs_int(n_soln_dofs_int);
    std::vector<adtype> rhs_ext(n_soln_dofs_ext);
    adtype dual_dot_residual;

    const auto &physics = *(DGBaseState<dim,nstate,real>::pde_physics_fad_fad);
    const auto &conv_num_flux = *(DGBaseState<dim,nstate,real>::conv_num_flux_fad_fad);
    const auto &diss_num_flux = *(DGBaseState<dim,nstate,real>::diss_num_flux_fad_fad);
    assemble_face_term(
        current_cell_index,
        neighbor_cell_index,
        soln_coeff_int,
        soln_coeff_ext,
        coords_coeff_int,
        coords_coeff_ext,
        dual_int,
        dual_ext,
        interior_face_number,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_int,
        fe_values_ext,
        penalty,
        fe_int,
        fe_ext,
        fe_metric,
        face_quadrature_int,
        face_quadrature_ext,
        rhs_int,
        rhs_ext,
        dual_dot_residual,
        compute_dRdW, compute_dRdX, compute_d2R);

    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
        local_rhs_int_cell[itest_int] += rhs_int[itest_int].val().val();
    }
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
        local_rhs_ext_cell[itest_ext] += rhs_ext[itest_ext].val().val();
    }

    if (compute_dRdW) {
        std::vector<real> residual_derivatives(n_soln_dofs_int);
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            // dR_int_dW_int
            residual_derivatives.resize(n_soln_dofs_int);
            for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                const unsigned int i_dx = idof+w_int_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, residual_derivatives);

            // dR_int_dW_ext
            residual_derivatives.resize(n_soln_dofs_ext);
            for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                const unsigned int i_dx = idof+w_ext_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, residual_derivatives);
        }

        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            // dR_ext_dW_int
            residual_derivatives.resize(n_soln_dofs_int);
            for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                const unsigned int i_dx = idof+w_int_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, residual_derivatives);

            // dR_ext_dW_ext
            residual_derivatives.resize(n_soln_dofs_ext);
            for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                const unsigned int i_dx = idof+w_ext_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, residual_derivatives);
        }
    }
    if (compute_dRdX) {
        std::vector<real> residual_derivatives(n_metric_dofs);
        for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
            // dR_int_dX_int
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_int_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_int, residual_derivatives);

            // dR_int_dX_ext
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_ext_start;
                residual_derivatives[idof] = rhs_int[itest_int].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_ext, residual_derivatives);
        }
        for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
            // dR_ext_dX_int
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_int_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_int, residual_derivatives);

            // dR_ext_dX_ext
            // residual_derivatives.resize(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_ext_start;
                residual_derivatives[idof] = rhs_ext[itest_ext].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_ext, residual_derivatives);
        }
    }

    if (compute_d2R) {
        std::vector<real> dWidWint(n_soln_dofs_int);
        std::vector<real> dWidWext(n_soln_dofs_ext);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);
        // dWint
        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {

            const unsigned int i_dx = idof+w_int_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

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
            const FadType dWi = dual_dot_residual.dx(i_dx);

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
            const FadType dWi = dual_dot_residual.dx(i_dx);

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
            const FadType dWi = dual_dot_residual.dx(i_dx);

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
#endif

#ifndef FADFAD
template <int dim, int nstate, typename real>
template <typename adtype>
void DGWeak<dim,nstate,real>::assemble_face_codi_taped_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
    const unsigned int interior_face_number,
    const unsigned int /*exterior_face_number*/,
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
    const Physics::PhysicsBase<dim, nstate, adtype> &physics,
    const NumericalFlux::NumericalFluxConvective<dim, nstate, adtype> &conv_num_flux,
    const NumericalFlux::NumericalFluxDissipative<dim, nstate, adtype> &diss_num_flux,
    dealii::Vector<real>          &local_rhs_int_cell,
    dealii::Vector<real>          &local_rhs_ext_cell,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;
    const unsigned int n_soln_dofs_int = fe_int.dofs_per_cell;
    const unsigned int n_soln_dofs_ext = fe_ext.dofs_per_cell;

    AssertDimension (n_soln_dofs_int, soln_dof_indices_int.size());
    AssertDimension (n_soln_dofs_ext, soln_dof_indices_ext.size());

    std::vector< adtype > coords_coeff_int(n_metric_dofs);
    std::vector< adtype > coords_coeff_ext(n_metric_dofs);
    std::vector< adtype > soln_coeff_int(n_soln_dofs_int);
    std::vector< adtype > soln_coeff_ext(n_soln_dofs_ext);

    // Current derivative ordering is: soln_int, soln_ext, metric_int, metric_ext
    unsigned int w_int_start, w_int_end, w_ext_start, w_ext_end,
                 x_int_start, x_int_end, x_ext_start, x_ext_end;
    automatic_differentiation_indexing_2(
        compute_dRdW, compute_dRdX, compute_d2R,
        n_soln_dofs_int, n_soln_dofs_ext, n_metric_dofs,
        w_int_start, w_int_end, w_ext_start, w_ext_end,
        x_int_start, x_int_end, x_ext_start, x_ext_end);

    using TH = codi::TapeHelper<adtype>;
    TH th;
    adtype::getGlobalTape();
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.startRecording();
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
        const real val = this->solution(soln_dof_indices_int[idof]);
        soln_coeff_int[idof] = val;
        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff_int[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff_int[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
        const real val = this->solution(soln_dof_indices_ext[idof]);
        soln_coeff_ext[idof] = val;
        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff_ext[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff_ext[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices_int[idof]];
        coords_coeff_int[idof] = val;
        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff_int[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff_int[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices_ext[idof]];
        coords_coeff_ext[idof] = val;
        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff_ext[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff_ext[idof]);
        }
    }

    std::vector<double> dual_int(n_soln_dofs_int);
    std::vector<double> dual_ext(n_soln_dofs_ext);

    for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_int[itest];
        dual_int[itest] = this->dual[global_residual_row];
    }
    for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices_ext[itest];
        dual_ext[itest] = this->dual[global_residual_row];
    }

    std::vector<adtype> rhs_int(n_soln_dofs_int);
    std::vector<adtype> rhs_ext(n_soln_dofs_ext);
    adtype dual_dot_residual;

    assemble_face_term(
        current_cell_index,
        neighbor_cell_index,
        soln_coeff_int,
        soln_coeff_ext,
        coords_coeff_int,
        coords_coeff_ext,
        dual_int,
        dual_ext,
        interior_face_number,
        physics,
        conv_num_flux,
        diss_num_flux,
        fe_values_int,
        fe_values_ext,
        penalty,
        fe_int,
        fe_ext,
        fe_metric,
        face_quadrature_int,
        face_quadrature_ext,
        rhs_int,
        rhs_ext,
        dual_dot_residual,
        compute_dRdW, compute_dRdX, compute_d2R);

    if (compute_dRdW || compute_dRdX) {
        for (unsigned int itest=0; itest<n_soln_dofs_int; ++itest) {
            th.registerOutput(rhs_int[itest]);
        }
        for (unsigned int itest=0; itest<n_soln_dofs_ext; ++itest) {
            th.registerOutput(rhs_ext[itest]);
        }
    } else if (compute_d2R) {
        th.registerOutput(dual_dot_residual);
    }
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.stopRecording();
        //adtype::getGlobalTape().printStatistics();
    }

    for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
        local_rhs_int_cell[itest_int] += getValue<adtype>(rhs_int[itest_int]);
    }
    for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {
        local_rhs_ext_cell[itest_ext] += getValue<adtype>(rhs_ext[itest_ext]);
    }

    if (compute_dRdW || compute_dRdX) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);

        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs_int);

            for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {
                int i_dependent = itest_int;

                // dR_int_dW_int
                residual_derivatives.resize(n_soln_dofs_int);
                for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                    const unsigned int i_dx = idof+w_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_int, residual_derivatives);

                // dR_int_dW_ext
                residual_derivatives.resize(n_soln_dofs_ext);
                for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                    const unsigned int i_dx = idof+w_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->system_matrix.add(soln_dof_indices_int[itest_int], soln_dof_indices_ext, residual_derivatives);
            }

            for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {

                int i_dependent = n_soln_dofs_int + itest_ext;

                // dR_ext_dW_int
                residual_derivatives.resize(n_soln_dofs_int);
                for (unsigned int idof = 0; idof < n_soln_dofs_int; ++idof) {
                    const unsigned int i_dx = idof+w_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_int, residual_derivatives);

                // dR_ext_dW_ext
                residual_derivatives.resize(n_soln_dofs_ext);
                for (unsigned int idof = 0; idof < n_soln_dofs_ext; ++idof) {
                    const unsigned int i_dx = idof+w_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->system_matrix.add(soln_dof_indices_ext[itest_ext], soln_dof_indices_ext, residual_derivatives);
            }
        }

        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);

            for (unsigned int itest_int=0; itest_int<n_soln_dofs_int; ++itest_int) {

                int i_dependent = itest_int;

                // dR_int_dX_int
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_int, residual_derivatives);

                // dR_int_dX_ext
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_int[itest_int], metric_dof_indices_ext, residual_derivatives);
            }

            for (unsigned int itest_ext=0; itest_ext<n_soln_dofs_ext; ++itest_ext) {

                int i_dependent = n_soln_dofs_int + itest_ext;

                // dR_ext_dX_int
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_int_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_int, residual_derivatives);

                // dR_ext_dX_ext
                for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                    const unsigned int i_dx = idof+x_ext_start;
                    residual_derivatives[idof] = jac(i_dependent,i_dx);
                }
                this->dRdXv.add(soln_dof_indices_ext[itest_ext], metric_dof_indices_ext, residual_derivatives);
            }
        }

        th.deleteJacobian(jac);
    }

    if (compute_d2R) {
        typename TH::HessianType& hes = th.createHessian();
        th.evalHessian(hes);

        std::vector<real> dWidW(n_soln_dofs_int);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        int i_dependent = (compute_dRdW || compute_dRdX) ? n_soln_dofs_int + n_soln_dofs_ext : 0;

        for (unsigned int idof=0; idof<n_soln_dofs_int; ++idof) {

            const unsigned int i_dx = idof+w_int_start;

            // dWint_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_int, dWidW);

            // dWint_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_int[idof], soln_dof_indices_ext, dWidW);

            // dWint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_int, dWidX);

            // dWint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_int[idof], metric_dof_indices_ext, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_int_start;

            // dXint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_int, dXidX);

            // dXint_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_int[idof], metric_dof_indices_ext, dXidX);
        }

        dWidW.resize(n_soln_dofs_ext);

        for (unsigned int idof=0; idof<n_soln_dofs_ext; ++idof) {

            const unsigned int i_dx = idof+w_ext_start;

            // dWext_dWint
            for (unsigned int jdof=0; jdof<n_soln_dofs_int; ++jdof) {
                const unsigned int j_dx = jdof+w_int_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_int, dWidW);

            // dWext_dWext
            for (unsigned int jdof=0; jdof<n_soln_dofs_ext; ++jdof) {
                const unsigned int j_dx = jdof+w_ext_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices_ext[idof], soln_dof_indices_ext, dWidW);

            // dWext_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_int, dWidX);

            // dWext_dXext
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices_ext[idof], metric_dof_indices_ext, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_ext_start;

            // dXint_dXint
            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_int_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_int, dXidX);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_ext_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices_ext[idof], metric_dof_indices_ext, dXidX);
        }

        th.deleteHessian(hes);
    }

}
#endif

template <int dim, int nstate, typename real>
template <typename real2>
void DGWeak<dim,nstate,real>::assemble_volume_term(
    const dealii::types::global_dof_index current_cell_index,
    const std::vector<real2> &soln_coeff, const std::vector<real2> &coords_coeff, const std::vector<real> &local_dual,
    const dealii::FESystem<dim,dim> &fe_soln, const dealii::FESystem<dim,dim> &fe_metric,
    const dealii::Quadrature<dim> &quadrature,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    std::vector<real2> &rhs, real2 &dual_dot_residual,
    const bool compute_metric_derivatives,
    const dealii::FEValues<dim,dim> &fe_values_vol)
{
    (void) current_cell_index;
    using Array = std::array<real2, nstate>;
    using Tensor1D = dealii::Tensor<1,dim,real2>;
    using Tensor2D = dealii::Tensor<2,dim,real2>;
    using ArrayTensor = std::array<Tensor1D, nstate>;

    const unsigned int n_quad_pts      = quadrature.size();
    const unsigned int n_soln_dofs     = fe_soln.dofs_per_cell;

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        rhs[itest] = 0;
    }
    dual_dot_residual = 0.0;

    const std::vector<dealii::Point<dim>> &points = quadrature.get_points ();

    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    // Evaluate metric terms
    std::vector<Tensor2D> metric_jacobian;
    if (compute_metric_derivatives) metric_jacobian = evaluate_metric_jacobian ( points, coords_coeff, fe_metric);
    std::vector<real2> jac_det(n_quad_pts);
    std::vector<Tensor2D> jac_inv_tran(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        if (compute_metric_derivatives) {
            const real2 jacobian_determinant = dealii::determinant(metric_jacobian[iquad]);
            const Tensor2D jacobian_transpose_inverse = dealii::transpose(dealii::invert(metric_jacobian[iquad]));

            jac_det[iquad] = jacobian_determinant;
            jac_inv_tran[iquad] = jacobian_transpose_inverse;
        } else {
            jac_det[iquad] = fe_values_vol.JxW(iquad) / quadrature.weight(iquad);
        }
    }

    // Build operators.
    const std::vector<dealii::Point<dim,double>> &unit_quad_pts = quadrature.get_points();
    dealii::FullMatrix<real> interpolation_operator(n_soln_dofs,n_quad_pts);
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
            interpolation_operator[idof][iquad] = fe_soln.shape_value(idof,unit_quad_pts[iquad]);
        }
    }
    // Might want to have the dimension as the innermost index
    // Need a contiguous 2d-array structure
    // std::array<dealii::FullMatrix<real2>,dim> gradient_operator;
    // for (int d=0;d<dim;++d) {
    //     gradient_operator[d].reinit(n_soln_dofs, n_quad_pts);
    // }
    std::array<dealii::Table<2,real2>,dim> gradient_operator;
    for (int d=0;d<dim;++d) {
        gradient_operator[d].reinit(dealii::TableIndices<2>(n_soln_dofs, n_quad_pts));
    }
    for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
         for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
             if (compute_metric_derivatives) {
                 //const dealii::Tensor<1,dim,real2> phys_shape_grad = dealii::contract<1,0>(jac_inv_tran[iquad], fe_soln.shape_grad(idof,points[iquad]));
                 const dealii::Tensor<1,dim,real2> ref_shape_grad = fe_soln.shape_grad(idof,points[iquad]);
                 dealii::Tensor<1,dim,real2> phys_shape_grad;
                 for (int dr=0;dr<dim;++dr) {
                     phys_shape_grad[dr] = 0.0;
                     for (int dc=0;dc<dim;++dc) {
                         phys_shape_grad[dr] += jac_inv_tran[iquad][dr][dc] * ref_shape_grad[dc];
                     }
                 }
                 for (int d=0;d<dim;++d) {
                     gradient_operator[d][idof][iquad] = phys_shape_grad[d];
                 }

                 // Exact mapping
                 // for (int d=0;d<dim;++d) {
                 //     const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                 //     gradient_operator[d][idof][iquad] = fe_values_vol.shape_grad_component(idof, iquad, istate)[d];
                 // }
             } else {
                 for (int d=0;d<dim;++d) {
                     const unsigned int istate = fe_soln.system_to_component_index(idof).first;
                     gradient_operator[d][idof][iquad] = fe_values_vol.shape_grad_component(idof, iquad, istate)[d];
                 }
             }
         }
     }



    //const real2 cell_diameter = fe_values_.get_cell()->diameter();
    real2 cell_diameter = 0.0;
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

        const real2 JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

        cell_diameter = cell_diameter + JxW_iquad;
    }
    //const real2 artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
    //                                    this->discontinuity_sensor(cell_diameter, soln_coeff, fe_soln)
    //                                    : 0.0;
    const real2 artificial_diss_coeff = this->all_parameters->add_artificial_dissipation ?
                                        this->artificial_dissipation_coeffs[current_cell_index]
                                        : 0.0;


    std::vector< Array > soln_at_q(n_quad_pts);
    std::vector< ArrayTensor > soln_grad_at_q(n_quad_pts); // Tensor initialize with zeros

    std::vector< ArrayTensor > conv_phys_flux_at_q(n_quad_pts);
    std::vector< ArrayTensor > diss_phys_flux_at_q(n_quad_pts);
    std::vector< Array > source_at_q(n_quad_pts);
    for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {
        for (int istate=0; istate<nstate; istate++) {
            soln_at_q[iquad][istate]      = 0;
            soln_grad_at_q[iquad][istate] = 0;
        }
        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {
            const unsigned int istate = fe_soln.system_to_component_index(idof).first;
            soln_at_q[iquad][istate]      += soln_coeff[idof] * interpolation_operator[idof][iquad];
            for (int d=0;d<dim;++d) {
                soln_grad_at_q[iquad][istate][d] += soln_coeff[idof] * gradient_operator[d][idof][iquad];
            }
        }
        conv_phys_flux_at_q[iquad] = physics.convective_flux (soln_at_q[iquad]);
        diss_phys_flux_at_q[iquad] = physics.dissipative_flux (soln_at_q[iquad], soln_grad_at_q[iquad]);

        if (this->all_parameters->add_artificial_dissipation) {
            ArrayTensor artificial_diss_phys_flux_at_q;
            artificial_diss_phys_flux_at_q = physics.artificial_dissipative_flux (artificial_diss_coeff, soln_at_q[iquad], soln_grad_at_q[iquad]);
            for (int s=0; s<nstate; s++) {
                diss_phys_flux_at_q[iquad][s] += artificial_diss_phys_flux_at_q[s];
            }
        }

        if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
            dealii::Point<dim,real2> ad_point;
            for (int d=0;d<dim;++d) { ad_point[d] = 0.0;}
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const int iaxis = fe_metric.system_to_component_index(idof).first;
                ad_point[iaxis] += coords_coeff[idof] * fe_metric.shape_value(idof,unit_quad_pts[iquad]);
            }
            source_at_q[iquad] = physics.source_term (ad_point, soln_at_q[iquad]);
            //Array artificial_source_at_q = physics.artificial_source_term (artificial_diss_coeff, ad_point, soln_at_q[iquad]);
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
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        const unsigned int istate = fe_soln.system_to_component_index(itest).first;

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            const real2 JxW_iquad = jac_det[iquad] * quadrature.weight(iquad);

            for (int d=0;d<dim;++d) {
                // Convective
                rhs[itest] = rhs[itest] + gradient_operator[d][itest][iquad] * conv_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
                //// Diffusive
                //// Note that for diffusion, the negative is defined in the physics
                rhs[itest] = rhs[itest] + gradient_operator[d][itest][iquad] * diss_phys_flux_at_q[iquad][istate][d] * JxW_iquad;
            }
            // Source
            if(this->all_parameters->manufactured_convergence_study_param.use_manufactured_source_term) {
                rhs[itest] = rhs[itest] + interpolation_operator[itest][iquad]* source_at_q[iquad][istate] * JxW_iquad;
            }
        }

        dual_dot_residual += local_dual[itest]*rhs[itest];

    }

}

#ifdef FADFAD
template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_term_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &/*fe_values_lagrange*/,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;

    using ADArray = std::array<FadFadType,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,FadFadType>, nstate >;

    const unsigned int n_soln_dofs     = fe_soln.dofs_per_cell;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector<FadFadType> coords_coeff(n_metric_dofs);
    std::vector<FadFadType> soln_coeff(n_soln_dofs);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices[itest];
        local_dual[itest] = this->dual[global_residual_row];
    }

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

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
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;
        coords_coeff[idof].val() = val;

        if (compute_dRdX || compute_d2R) coords_coeff[idof].diff(i_derivative, n_total_indep);
        if (compute_d2R) coords_coeff[idof].val().diff(i_derivative, n_total_indep);

        if (compute_dRdX || compute_d2R) i_derivative++;
    }

    AssertDimension(i_derivative, n_total_indep);

    FadFadType dual_dot_residual = 0.0;
    std::vector<FadFadType> rhs(n_soln_dofs);
    assemble_volume_term<FadFadType>(
        current_cell_index,
        soln_coeff, coords_coeff, local_dual,
        fe_soln, fe_metric, quadrature,
        *(DGBaseState<dim,nstate,real>::pde_physics_fad_fad),
        rhs, dual_dot_residual,
        compute_metric_derivatives, fe_values_vol);

    // Weak form
    // The right-hand side sends all the term to the side of the source term
    // Therefore,
    // \divergence ( Fconv + Fdiss ) = source
    // has the right-hand side
    // rhs = - \divergence( Fconv + Fdiss ) + source
    // Since we have done an integration by parts, the volume term resulting from the divergence of Fconv and Fdiss
    // is negative. Therefore, negative of negative means we add that volume term to the right-hand-side
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

        if (compute_dRdW) {
            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
                AssertIsFinite(residual_derivatives[idof]);
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
        if (compute_dRdX) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = rhs[itest].dx(i_dx).val();
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        //if (compute_d2R) {
        //    const unsigned int global_residual_row = soln_dof_indices[itest];
        //    dual_dot_residual += this->dual[global_residual_row]*rhs[itest];
        //}

        local_rhs_cell(itest) += rhs[itest].val().val();
        AssertIsFinite(local_rhs_cell(itest));

    }


    if (compute_d2R) {

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;
            const FadType dWi = dual_dot_residual.dx(i_dx);

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
            const FadType dXi = dual_dot_residual.dx(i_dx);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = dXi.dx(j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }
    }

}
#endif
#ifndef FADFAD
template <int dim, int nstate, typename real>
template <typename real2>
void DGWeak<dim,nstate,real>::assemble_volume_codi_taped_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const Physics::PhysicsBase<dim, nstate, real2> &physics,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) fe_values_lagrange;

    using adtype = real2;

    using ADArray = std::array<adtype,nstate>;
    using ADArrayTensor1 = std::array< dealii::Tensor<1,dim,adtype>, nstate >;

    const unsigned int n_soln_dofs = fe_soln.dofs_per_cell;

    AssertDimension (n_soln_dofs, soln_dof_indices.size());

    const dealii::FESystem<dim> &fe_metric = this->high_order_grid.fe_system;
    const unsigned int n_metric_dofs = fe_metric.dofs_per_cell;

    std::vector<adtype> coords_coeff(n_metric_dofs);
    std::vector<adtype> soln_coeff(n_soln_dofs);

    std::vector<real> local_dual(n_soln_dofs);
    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        const unsigned int global_residual_row = soln_dof_indices[itest];
        local_dual[itest] = this->dual[global_residual_row];
    }

    const bool compute_metric_derivatives = (!compute_dRdX && !compute_d2R) ? false : true;

    unsigned int w_start, w_end, x_start, x_end;
    automatic_differentiation_indexing_1( compute_dRdW, compute_dRdX, compute_d2R,
                                          n_soln_dofs, n_metric_dofs,
                                          w_start, w_end, x_start, x_end );

    using TH = codi::TapeHelper<adtype>;
    TH th;
    adtype::getGlobalTape();
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.startRecording();
    }
    for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
        const real val = this->solution(soln_dof_indices[idof]);
        soln_coeff[idof] = val;

        if (compute_dRdW || compute_d2R) {
            th.registerInput(soln_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(soln_coeff[idof]);
        }
    }
    for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
        const real val = this->high_order_grid.volume_nodes[metric_dof_indices[idof]];
        coords_coeff[idof] = val;

        if (compute_dRdX || compute_d2R) {
            th.registerInput(coords_coeff[idof]);
        } else {
            adtype::getGlobalTape().deactivateValue(coords_coeff[idof]);
        }
    }

    adtype dual_dot_residual = 0.0;
    std::vector<adtype> rhs(n_soln_dofs);
    assemble_volume_term<adtype>(
        current_cell_index,
        soln_coeff, coords_coeff, local_dual,
        fe_soln, fe_metric, quadrature,
        physics,
        rhs, dual_dot_residual,
        compute_metric_derivatives, fe_values_vol);

    if (compute_dRdW || compute_dRdX) {
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            th.registerOutput(rhs[itest]);
        }
    } else if (compute_d2R) {
        th.registerOutput(dual_dot_residual);
    }
    if (compute_dRdW || compute_dRdX || compute_d2R) {
        th.stopRecording();
    }

    for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
        local_rhs_cell(itest) += getValue<adtype>(rhs[itest]);
        AssertIsFinite(local_rhs_cell(itest));
    }

    if (compute_dRdW) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {

            std::vector<real> residual_derivatives(n_soln_dofs);
            for (unsigned int idof = 0; idof < n_soln_dofs; ++idof) {
                const unsigned int i_dx = idof+w_start;
                residual_derivatives[idof] = jac(itest,i_dx);
                AssertIsFinite(residual_derivatives[idof]);
            }
            this->system_matrix.add(soln_dof_indices[itest], soln_dof_indices, residual_derivatives);
        }
        th.deleteJacobian(jac);

    }

    if (compute_dRdX) {
        typename TH::JacobianType& jac = th.createJacobian();
        th.evalJacobian(jac);
        for (unsigned int itest=0; itest<n_soln_dofs; ++itest) {
            std::vector<real> residual_derivatives(n_metric_dofs);
            for (unsigned int idof = 0; idof < n_metric_dofs; ++idof) {
                const unsigned int i_dx = idof+x_start;
                residual_derivatives[idof] = jac(itest,i_dx);
            }
            this->dRdXv.add(soln_dof_indices[itest], metric_dof_indices, residual_derivatives);
        }
        th.deleteJacobian(jac);
    }


    if (compute_d2R) {
        typename TH::HessianType& hes = th.createHessian();
        th.evalHessian(hes);

        int i_dependent = (compute_dRdW || compute_dRdX) ? n_soln_dofs : 0;

        std::vector<real> dWidW(n_soln_dofs);
        std::vector<real> dWidX(n_metric_dofs);
        std::vector<real> dXidX(n_metric_dofs);

        for (unsigned int idof=0; idof<n_soln_dofs; ++idof) {

            const unsigned int i_dx = idof+w_start;

            for (unsigned int jdof=0; jdof<n_soln_dofs; ++jdof) {
                const unsigned int j_dx = jdof+w_start;
                dWidW[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdW.add(soln_dof_indices[idof], soln_dof_indices, dWidW);

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dWidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdWdX.add(soln_dof_indices[idof], metric_dof_indices, dWidX);
        }

        for (unsigned int idof=0; idof<n_metric_dofs; ++idof) {

            const unsigned int i_dx = idof+x_start;

            for (unsigned int jdof=0; jdof<n_metric_dofs; ++jdof) {
                const unsigned int j_dx = jdof+x_start;
                dXidX[jdof] = hes(i_dependent,i_dx,j_dx);
            }
            this->d2RdXdX.add(metric_dof_indices[idof], metric_dof_indices, dXidX);
        }

        th.deleteHessian(hes);
    }

}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_volume_term_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::FEValues<dim,dim> &fe_values_vol,
    const dealii::FESystem<dim,dim> &fe_soln,
    const dealii::Quadrature<dim> &quadrature,
    const std::vector<dealii::types::global_dof_index> &metric_dof_indices,
    const std::vector<dealii::types::global_dof_index> &soln_dof_indices,
    dealii::Vector<real> &local_rhs_cell,
    const dealii::FEValues<dim,dim> &fe_values_lagrange,
    const bool compute_dRdW, const bool compute_dRdX, const bool compute_d2R)
{
    (void) current_cell_index;
    (void) fe_values_lagrange;
    if (compute_d2R) {
        assemble_volume_codi_taped_derivatives<codi_HessianComputationType>(
            current_cell_index,
            fe_values_vol,
            fe_soln, quadrature,
            metric_dof_indices, soln_dof_indices,
            local_rhs_cell,
            fe_values_lagrange,
            *(DGBaseState<dim,nstate,real>::pde_physics_rad_fad),
            compute_dRdW, compute_dRdX, compute_d2R);
    } else if (compute_dRdW || compute_dRdX) {
        assemble_volume_codi_taped_derivatives<codi_JacobianComputationType>(
            current_cell_index,
            fe_values_vol,
            fe_soln, quadrature,
            metric_dof_indices, soln_dof_indices,
            local_rhs_cell,
            fe_values_lagrange,
            *(DGBaseState<dim,nstate,real>::pde_physics_rad),
            compute_dRdW, compute_dRdX, compute_d2R);
    }

    return;
}

template <int dim, int nstate, typename real>
void DGWeak<dim,nstate,real>::assemble_face_term_derivatives(
    const dealii::types::global_dof_index current_cell_index,
    const dealii::types::global_dof_index neighbor_cell_index,
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
    (void) current_cell_index;
    (void) neighbor_cell_index;
    if (compute_d2R) {
        assemble_face_codi_taped_derivatives<codi_HessianComputationType>(
            current_cell_index,
            neighbor_cell_index,
            interior_face_number,
            exterior_face_number,
            fe_values_int,
            fe_values_ext,
            penalty,
            fe_int,
            fe_ext,
            face_quadrature_int,
            face_quadrature_ext,
            metric_dof_indices_int,
            metric_dof_indices_ext,
            soln_dof_indices_int,
            soln_dof_indices_ext,
            *(DGBaseState<dim,nstate,real>::pde_physics_rad_fad),
            *(DGBaseState<dim,nstate,real>::conv_num_flux_rad_fad),
            *(DGBaseState<dim,nstate,real>::diss_num_flux_rad_fad),
            local_rhs_int_cell,
            local_rhs_ext_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    } else if (compute_dRdW || compute_dRdX) {
        assemble_face_codi_taped_derivatives<codi_JacobianComputationType>(
            current_cell_index,
            neighbor_cell_index,
            interior_face_number,
            exterior_face_number,
            fe_values_int,
            fe_values_ext,
            penalty,
            fe_int,
            fe_ext,
            face_quadrature_int,
            face_quadrature_ext,
            metric_dof_indices_int,
            metric_dof_indices_ext,
            soln_dof_indices_int,
            soln_dof_indices_ext,
            *(DGBaseState<dim,nstate,real>::pde_physics_rad),
            *(DGBaseState<dim,nstate,real>::conv_num_flux_rad),
            *(DGBaseState<dim,nstate,real>::diss_num_flux_rad),
            local_rhs_int_cell,
            local_rhs_ext_cell,
            compute_dRdW, compute_dRdX, compute_d2R);
    }

    return;
}
#endif

template class DGWeak <PHILIP_DIM, 1, double>;
template class DGWeak <PHILIP_DIM, 2, double>;
template class DGWeak <PHILIP_DIM, 3, double>;
template class DGWeak <PHILIP_DIM, 4, double>;
template class DGWeak <PHILIP_DIM, 5, double>;

} // PHiLiP namespace

