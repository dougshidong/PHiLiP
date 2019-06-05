#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"

namespace PHiLiP {
namespace Physics {

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

    if(dim >= 1) advection_speed[0] = this->velo_x;
    if(dim >= 2) advection_speed[1] = this->velo_y;
    if(dim >= 3) advection_speed[2] = this->velo_z;
    return advection_speed;
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

//  template <int dim, int nstate, typename real>
//  std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
//  ::apply_diffusion_matrix(
//          const std::array<real,nstate> &/*solution*/,
//          const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_grad) const
//  {
//      // deal.II tensors are initialized with zeros
//      std::array<dealii::Tensor<1,dim,real>,nstate> diffusion;
//      for (int d=0; d<dim; d++) {
//          diffusion[0][d] = 1.0*solution_grad[0][d];
//      }
//      return diffusion;
//  }

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    const double diff_coeff = this->diff_coeff;
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = -diff_coeff*1.0*solution_gradient[i];
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

    const double diff_coeff = this->diff_coeff;
    const int ISTATE = 0;
    if (dim==1) {
        const real x = pos[0];
        source[ISTATE] = velocity_field[0]*a*cos(a*x+d) +
                 diff_coeff*a*a*sin(a*x+d);
    } else if (dim==2) {
        const real x = pos[0], y = pos[1];
        source[ISTATE] = velocity_field[0]*a*cos(a*x+d)*sin(b*y+e) +
                 velocity_field[1]*b*sin(a*x+d)*cos(b*y+e) +
                 diff_coeff*a*a*sin(a*x+d)*sin(b*y+e) +
                 diff_coeff*b*b*sin(a*x+d)*sin(b*y+e);
    } else if (dim==3) {
        const real x = pos[0], y = pos[1], z = pos[2];
        source[ISTATE] =   velocity_field[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                   velocity_field[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                   velocity_field[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f) +
                   diff_coeff*a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                   diff_coeff*b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                   diff_coeff*c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
    }
    return source;
}

template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;

} // Physics namespace
} // PHiLiP namespace


