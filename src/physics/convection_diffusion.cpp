#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
void ConvectionDiffusion<dim,nstate,real>
::boundary_face_values (
   const int /*boundary_type*/,
   const dealii::Point<dim, double> &pos,
   const dealii::Tensor<1,dim,real> &normal_int,
   const std::array<real,nstate> &soln_int,
   const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
   std::array<real,nstate> &soln_bc,
   std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{

    const std::array<real,nstate> boundary_values = PhysicsBase<dim,nstate,real>::manufactured_solution(pos);
    const std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients = PhysicsBase<dim,nstate,real>::manufactured_gradient(pos);

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
            // soln_bc[istate] = -soln_int[istate]+2*boundary_values[istate];

            // //soln_grad_bc[istate] = soln_grad_int[istate];
            // soln_grad_bc[istate] = boundary_gradients[istate];

            soln_bc[istate] = soln_int[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];
            //soln_grad_bc[istate] = boundary_gradients[istate];
        }
    }
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &solution) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    const dealii::Tensor<1,dim,real> velocity_field = this->advection_speed();
    for (int i=0; i<nstate; ++i) {
        conv_flux[i] = velocity_field * solution[i];
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> ConvectionDiffusion<dim,nstate,real>
::advection_speed () const
{
    dealii::Tensor<1,dim,real> advection_speed;
    if (hasConvection) {
        if(dim >= 1) advection_speed[0] = this->velo_x;
        if(dim >= 2) advection_speed[1] = this->velo_y;
        if(dim >= 3) advection_speed[2] = this->velo_z;
    } else {
        const real zero = 0.0;
        if(dim >= 1) advection_speed[0] = zero;
        if(dim >= 2) advection_speed[1] = zero;
        if(dim >= 3) advection_speed[2] = zero;
    }
    return advection_speed;
}
template <int dim, int nstate, typename real>
real ConvectionDiffusion<dim,nstate,real>
::diffusion_coefficient () const
{
    if(hasDiffusion) return this->diff_coeff;
    const real zero = 0.0;
    return zero;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> ConvectionDiffusion<dim,nstate,real>
::convective_eigenvalues (
    const std::array<real,nstate> &/*solution*/,
    const dealii::Tensor<1,dim,real> &normal) const
{
    std::array<real,nstate> eig;
    const dealii::Tensor<1,dim,real> advection_speed = this->advection_speed();
    for (int i=0; i<nstate; i++) {
        eig[i] = advection_speed*normal;
    }
    return eig;
}

template <int dim, int nstate, typename real>
real ConvectionDiffusion<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*soln*/) const
{
    const dealii::Tensor<1,dim,real> advection_speed = this->advection_speed();
    real max_eig = 0;
    for (int i=0; i<dim; i++) {
        max_eig = std::max(max_eig,std::abs(advection_speed[0]));
    }
    return max_eig;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    const real diff_coeff = diffusion_coefficient();
    using phys = PhysicsBase<dim,nstate,real>;
    const double a11 = phys::A11, a12 = phys::A12, a13 = phys::A13;
    const double a21 = phys::A21, a22 = phys::A22, a23 = phys::A23;
    const double a31 = phys::A31, a32 = phys::A32, a33 = phys::A33;
    for (int i=0; i<nstate; i++) {
        //diss_flux[i] = -diff_coeff*1.0*solution_gradient[i];
        if (dim==1) {
            diss_flux[i] = -diff_coeff*a11*solution_gradient[i];
        } else if (dim==2) {
            diss_flux[i][0] = -diff_coeff*a11*solution_gradient[i][0]
                              -diff_coeff*a12*solution_gradient[i][1];
            diss_flux[i][1] = -diff_coeff*a21*solution_gradient[i][0]
                              -diff_coeff*a22*solution_gradient[i][1];
        } else if (dim==3) {
            diss_flux[i][0] = -diff_coeff*a11*solution_gradient[i][0]
                              -diff_coeff*a12*solution_gradient[i][1]
                              -diff_coeff*a13*solution_gradient[i][2];
            diss_flux[i][1] = -diff_coeff*a21*solution_gradient[i][0]
                              -diff_coeff*a22*solution_gradient[i][1]
                              -diff_coeff*a23*solution_gradient[i][2];
            diss_flux[i][2] = -diff_coeff*a31*solution_gradient[i][0]
                              -diff_coeff*a32*solution_gradient[i][1]
                              -diff_coeff*a33*solution_gradient[i][2];
        }

    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> ConvectionDiffusion<dim,nstate,real>
::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;
    const dealii::Tensor<1,dim,real> velocity_field = this->advection_speed();
    using phys = PhysicsBase<dim,nstate,real>;
    const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
    const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

    const real diff_coeff = diffusion_coefficient();

    const double a11 = phys::A11, a12 = phys::A12, a13 = phys::A13;
    const double a21 = phys::A21, a22 = phys::A22, a23 = phys::A23;
    const double a31 = phys::A31, a32 = phys::A32, a33 = phys::A33;

    int istate = 0;
    if (dim==1) {
        const real x = pos[0];
        source[istate] =
              velocity_field[0]*a*cos(a*x+d)
            + diff_coeff*a11*a*a*sin(a*x+d);
    } else if (dim==2) {
        const real x = pos[0], y = pos[1];
        source[istate] =
              velocity_field[0]*a*cos(a*x+d)*sin(b*y+e)
            + velocity_field[1]*b*sin(a*x+d)*cos(b*y+e)
            + diff_coeff*a11*a*a*sin(a*x+d)*sin(b*y+e)
            - diff_coeff*a12*a*b*cos(a*x+d)*cos(b*y+e)
            - diff_coeff*a21*b*a*cos(a*x+d)*cos(b*y+e)
            + diff_coeff*a22*b*b*sin(a*x+d)*sin(b*y+e);
    } else if (dim==3) {
        const real x = pos[0], y = pos[1], z = pos[2];
        source[istate] =   
              velocity_field[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f)
            + velocity_field[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f)
            + velocity_field[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f)
            + diff_coeff*a11*a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f)
            - diff_coeff*a12*a*b*cos(a*x+d)*cos(b*y+e)*sin(c*z+f)
            - diff_coeff*a13*a*c*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
            - diff_coeff*a21*b*a*cos(a*x+d)*cos(b*y+e)*sin(c*z+f)
            + diff_coeff*a22*b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f)
            - diff_coeff*a23*b*c*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
            - diff_coeff*a31*c*a*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
            - diff_coeff*a32*c*b*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
            + diff_coeff*a33*c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
    }

    istate = 1;
    if (nstate > 1) {
        int istate = 1;
        if (dim==1) {
            const real x = pos[0];
            source[istate] = -velocity_field[0]*a*sin(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source[istate] = - velocity_field[0]*a*sin(a*x+d)*cos(b*y+e)
                             - velocity_field[1]*b*cos(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];
            source[istate] =  - velocity_field[0]*a*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                              - velocity_field[1]*b*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                              - velocity_field[2]*c*cos(a*x+d)*cos(b*y+e)*sin(c*z+f);
        }
    }
    return source;
}

template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
template class ConvectionDiffusion < PHILIP_DIM, 2, double >;
template class ConvectionDiffusion < PHILIP_DIM, 2, Sacado::Fad::DFad<double>  >;

} // Physics namespace
} // PHiLiP namespace


