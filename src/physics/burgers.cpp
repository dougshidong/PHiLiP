#include "ADTypes.hpp"

#include "burgers.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
Burgers<dim,nstate,real>::Burgers(
    const Parameters::AllParameters *const                    parameters_input,
    const double                                              diffusion_coefficient,
    const bool                                                convection, 
    const bool                                                diffusion,
    const dealii::Tensor<2,3,double>                          input_diffusion_tensor,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const Parameters::AllParameters::TestType                 parameters_test,
    const bool                                                has_nonzero_physical_source)
    : PhysicsBase<dim,nstate,real>(parameters_input, diffusion, has_nonzero_physical_source, input_diffusion_tensor, manufactured_solution_function)
    , diffusion_scaling_coeff(diffusion_coefficient)
    , hasConvection(convection)
    , hasDiffusion(diffusion)
    , test_type(parameters_test)
{
    static_assert(nstate==dim, "Physics::Burgers() should be created with nstate==dim");
}

template <int dim, int nstate, typename real>
void Burgers<dim,nstate,real>
::boundary_face_values (
   const int /*boundary_type*/,
   const dealii::Point<dim, real> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    std::array<real,nstate> boundary_values;
    std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;
    for (int i=0; i<nstate; i++) {
        boundary_values[i] = this->manufactured_solution_function->value (pos, i);
        boundary_gradients[i] = this->manufactured_solution_function->gradient (pos, i);
    }

    for (int istate=0; istate<nstate; ++istate) {

        std::array<real,nstate> characteristic_dot_n = convective_eigenvalues(boundary_values, normal_int);
        const bool inflow = (characteristic_dot_n[istate] <= 0.);

        if (inflow || hasDiffusion) { // Dirichlet boundary condition
            // soln_bc[istate] = boundary_values[istate];
            // soln_grad_bc[istate] = soln_grad_int[istate];

            soln_bc[istate] = boundary_values[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];

        } else { // Neumann boundary condition
            // //soln_bc[istate] = soln_int[istate];
            // //soln_bc[istate] = boundary_values[istate];
            //soln_bc[istate] = -soln_int[istate]+2*boundary_values[istate];
            soln_bc[istate] = soln_int[istate];

            // **************************************************************************************************************
            // Note I don't know how to properly impose the soln_grad_bc to obtain an adjoint consistent scheme
            // Currently, Neumann boundary conditions are only imposed for the linear advection
            // Therefore, soln_grad_bc does not affect the solution
            // **************************************************************************************************************
            soln_grad_bc[istate] = soln_grad_int[istate];
            //soln_grad_bc[istate] = boundary_gradients[istate];
            //soln_grad_bc[istate] = -soln_grad_int[istate]+2*boundary_gradients[istate];
        }
    }
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Burgers<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &solution) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
        for (int s=0; s<nstate; ++s) {
            conv_flux[s][flux_dim] = 0.5*solution[flux_dim]*solution[s];
        }
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Burgers<dim,nstate,real>::convective_numerical_split_flux (
                const std::array<real,nstate> &conservative_soln1,
                const std::array<real,nstate> &conservative_soln2) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
        for (int flux_dim=0; flux_dim<dim; ++flux_dim) {
            for (int s=0; s<nstate; ++s) {
                conv_flux[s][flux_dim] = 1./6. * (conservative_soln1[flux_dim]*conservative_soln1[flux_dim] + conservative_soln1[flux_dim]*conservative_soln2[s] + conservative_soln2[s]*conservative_soln2[s]);
            }
        }
        return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Burgers<dim, nstate, real>
::compute_entropy_variables (
    const std::array<real,nstate> &conservative_soln) const
{
    return conservative_soln;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Burgers<dim, nstate, real>
::compute_conservative_variables_from_entropy_variables (
    const std::array<real,nstate> &entropy_var) const
{
    return entropy_var;
}

template <int dim, int nstate, typename real>
real Burgers<dim,nstate,real>
::diffusion_coefficient () const
{
    if(hasDiffusion) return this->diffusion_scaling_coeff;
    const real zero = 0.0;
    return zero;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Burgers<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &solution,
    const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real,nstate> eig;
    for (int i=0; i<nstate; i++) {
        eig[i] = 0.0;
        for (int d=0;d<dim;++d) {
            eig[i] += solution[d]*normal[d];
        }
    }
    return eig;
}

template <int dim, int nstate, typename real>
real Burgers<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &soln) const
{
    real max_eig = 0;
    for (int i=0; i<dim; i++) {
        //max_eig = std::max(max_eig,std::abs(soln[i]));
        const real abs_soln = abs(soln[i]);
        max_eig = std::max(max_eig, abs_soln);
        //max_eig += soln[i] * soln[i];
    }
    return max_eig;
}

template <int dim, int nstate, typename real>
real Burgers<dim,nstate,real>
::max_viscous_eigenvalue (const std::array<real,nstate> &/*conservative_soln*/) const
{
    return 0.0;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Burgers<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &solution,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    return dissipative_flux(solution, solution_gradient);
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Burgers<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    const real diff_coeff = diffusion_coefficient();
    for (int i=0; i<nstate; i++) {
        for (int d1=0; d1<dim; d1++) {
            diss_flux[i][d1] = 0.0;
            for (int d2=0; d2<dim; d2++) {
                diss_flux[i][d1] += -diff_coeff*((this->diffusion_tensor[d1][d2])*solution_gradient[i][d2]);
            }
        }
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Burgers<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &solution,
    const real current_time,
    const dealii::types::global_dof_index /*cell_index*/) const
{
    return source_term(pos,solution,current_time);
}

template <int dim, int nstate, typename real>
std::array<real,nstate> Burgers<dim,nstate,real>
::source_term (
    const dealii::Point<dim,real> &pos,
    const std::array<real,nstate> &/*solution*/,
    const real current_time) const
{
    std::array<real,nstate> source;

    using TestType = Parameters::AllParameters::TestType;

    if(this->test_type == TestType::burgers_energy_stability || this->test_type == TestType::burgers_limiter){
        for(int istate =0; istate<nstate; istate++){
            source[istate] = 0.0;
            const double pi = atan(1)*4.0;
            const real pi_xmt = pi * (pos[0] - current_time);
            const real pi_ymt = pi * (pos[1] - current_time);
            const real pi_zmt = pi * (pos[2] - current_time);

            if(dim==1)
            {
                source[istate] = pi * sin(pi_xmt)
                                   *(1.0 - cos(pi_xmt));
            }

            if(dim==2)
            {
                const real sourcedt = pi*sin(pi_xmt)*cos(pi_ymt)
                               +pi*cos(pi_xmt)*sin(pi_ymt);
                const real sourcedx = -pi*sin(pi_xmt)*(cos(pi_xmt))
                               *pow(cos(pi_ymt),2.0);
                const real sourcedy = -pi*sin(pi_ymt)*(cos(pi_ymt))
                               *pow(cos(pi_xmt),2.0);
                source[istate] = sourcedt + sourcedx + sourcedy;
            }

            if(dim==3)
            {
                const real sourcedt = pi*sin(pi_xmt)*cos(pi_ymt)*cos(pi_zmt)
                               +pi*cos(pi_xmt)*sin(pi_ymt)*cos(pi_zmt)
                               +pi*cos(pi_xmt)*cos(pi_ymt)*sin(pi_zmt);
                const real sourcedx = -pi*sin(pi_xmt)*(cos(pi_xmt))
                               *pow(cos(pi_ymt),2.0)*pow(cos(pi_zmt),2.0);
                const real sourcedy = -pi*sin(pi_ymt)*(cos(pi_ymt))
                               *pow(cos(pi_xmt),2.0)*pow(cos(pi_zmt),2.0);
                const real sourcedz = -pi*sin(pi_zmt)*(cos(pi_zmt))
                               *pow(cos(pi_xmt),2.0)*pow(cos(pi_ymt),2.0);
                source[istate] = sourcedt + sourcedx + sourcedy + sourcedz;
            }
        }
    }
    else{
    const real diff_coeff = diffusion_coefficient();
    // for (int istate=0; istate<nstate; istate++) {
    //     dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function.gradient (pos, istate);
    //     dealii::SymmetricTensor<2,dim,real> manufactured_hessian = this->manufactured_solution_function.hessian (pos, istate);
    //     source[istate] = 0.0;
    //     for (int d=0;d<dim;++d) {
    //         real manufactured_solution = this->manufactured_solution_function.value (pos, d);
    //         source[istate] += manufactured_solution*manufactured_gradient[d];
    //     }
    //     source[istate] += -diff_coeff*scalar_product((this->diffusion_tensor),manufactured_hessian);
    // }
    for (int istate=0; istate<nstate; istate++) {
        source[istate] = 0.0;
        dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, istate);
        dealii::SymmetricTensor<2,dim,real> manufactured_hessian = this->manufactured_solution_function->hessian (pos, istate);
        for (int d=0;d<dim;++d) {
            real manufactured_solution = this->manufactured_solution_function->value (pos, d);
            source[istate] += 0.5*manufactured_solution*manufactured_gradient[d];
        }
        //source[istate] += -diff_coeff*scalar_product((this->diffusion_tensor),manufactured_hessian);
        real hess = 0.0;
        for (int dr=0; dr<dim; ++dr) {
            for (int dc=0; dc<dim; ++dc) {
                hess += (this->diffusion_tensor)[dr][dc] * manufactured_hessian[dr][dc];
            }
        }
        source[istate] += -diff_coeff*hess;
    }
    for (int istate=0; istate<nstate; istate++) {
        real manufactured_solution = this->manufactured_solution_function->value (pos, istate);
        real divergence = 0.0;
        for (int d=0;d<dim;++d) {
            dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, d);
            divergence += manufactured_gradient[d];
        }
        source[istate] += 0.5*manufactured_solution*divergence;
    }

    }
    // for (int istate=0; istate<nstate; istate++) {
    //     source[istate] = 0.0;
    //     for (int d=0;d<dim;++d) {
    //         dealii::Point<dim,real> posp = pos;
    //         dealii::Point<dim,real> posm = pos;
    //         posp[d] += 1e-8;
    //         posm[d] -= 1e-8;
    //         std::array<real,nstate> solp,solm;
    //         for (int s=0; s<nstate; s++) { 
    //             solp[s] = this->manufactured_solution_function.value (posp, s);
    //             solm[s] = this->manufactured_solution_function.value (posm, s);
    //         }
    //         std::array<dealii::Tensor<1,dim,real>,nstate> convp = convective_flux (solp);
    //         std::array<dealii::Tensor<1,dim,real>,nstate> convm = convective_flux (solm);
    //         source[istate] += (convp[istate][d] - convm[istate][d]) / 2e-8;
    //     }
    // }
    return source;
}

template class Burgers < PHILIP_DIM, PHILIP_DIM, double >;
template class Burgers < PHILIP_DIM, PHILIP_DIM, FadType  >;
template class Burgers < PHILIP_DIM, PHILIP_DIM, RadType  >;
template class Burgers < PHILIP_DIM, PHILIP_DIM, FadFadType >;
template class Burgers < PHILIP_DIM, PHILIP_DIM, RadFadType >;

} // Physics namespace
} // PHiLiP namespace




