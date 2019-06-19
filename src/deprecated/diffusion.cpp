#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"
namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> Diffusion<dim, nstate, real>
::convective_flux (const std::array<real,nstate> &/*solution*/) const
{ 
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    for (int i=0; i<nstate; i++) {
        conv_flux[i] = 0;
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<real, nstate> Diffusion<dim, nstate, real>
::convective_eigenvalues(
    const std::array<real,nstate> &/*solution*/,
    const dealii::Tensor<1,dim,real> &/*normal*/) const
{
    std::array<real,nstate> eig;
    for (int i=0; i<nstate; i++) {
        eig[i] = 0;
    }
    return eig;
}
template <int dim, int nstate, typename real>
real Diffusion<dim,nstate,real>
::max_convective_eigenvalue (const std::array<real,nstate> &/*soln*/) const
{
    const real max_eig = 0;
    return max_eig;
}

//  template <int dim, int nstate, typename real>
//  std::array<dealii::Tensor<1,dim,real>,nstate> Diffusion<dim,nstate,real>
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
std::array<dealii::Tensor<1,dim,real>,nstate> Diffusion<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    const double diff_coeff = this->diff_coeff;

    using phys = PhysicsBase<dim,nstate,real>;
    //const double a11 = 10*phys::freq_x, a12 = phys::freq_y, a13 = phys::freq_z;
    //const double a21 = phys::offs_x, a22 = 10*phys::offs_y, a23 = phys::offs_z;
    //const double a31 = phys::velo_x, a32 = phys::velo_y, a33 = 10*phys::velo_z;

    const double a11 = phys::A11, a12 = phys::A12, a13 = phys::A13;
    const double a21 = phys::A11, a22 = phys::A22, a23 = phys::A23;
    const double a31 = phys::A11, a32 = phys::A32, a33 = phys::A33;
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
std::array<real,nstate> Diffusion<dim,nstate,real>
::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;
    using phys = PhysicsBase<dim,nstate,real>;
    const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
    const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
    const double diff_coeff = this->diff_coeff;
    const int ISTATE = 0;
    if (dim==1) {
        const real x = pos[0];
        source[ISTATE] = diff_coeff*a*a*sin(a*x+d);
    } else if (dim==2) {
        const real x = pos[0], y = pos[1];
        source[ISTATE] = diff_coeff*a*a*sin(a*x+d)*sin(b*y+e) +
                         diff_coeff*b*b*sin(a*x+d)*sin(b*y+e);
    } else if (dim==3) {
        const real x = pos[0], y = pos[1], z = pos[2];

        source[ISTATE] =  diff_coeff*a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                  diff_coeff*b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                  diff_coeff*c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
    }
    //const double a11 = 10*phys::freq_x, a12 = phys::freq_y, a13 = phys::freq_z;
    //const double a21 = phys::offs_x, a22 = 10*phys::offs_y, a23 = phys::offs_z;
    //const double a31 = phys::velo_x, a32 = phys::velo_y, a33 = 10*phys::velo_z;
    const double a11 = phys::A11, a12 = phys::A12, a13 = phys::A13;
    const double a21 = phys::A11, a22 = phys::A22, a23 = phys::A23;
    const double a31 = phys::A11, a32 = phys::A32, a33 = phys::A33;
    if (dim==1) {
        const real x = pos[0];
        source[ISTATE] = diff_coeff*a11*a*a*sin(a*x+d);
    } else if (dim==2) {
        const real x = pos[0], y = pos[1];
        source[ISTATE] =   diff_coeff*a11*a*a*sin(a*x+d)*sin(b*y+e)
                         - diff_coeff*a12*a*b*cos(a*x+d)*cos(b*y+e)
                         - diff_coeff*a21*b*a*cos(a*x+d)*cos(b*y+e)
                         + diff_coeff*a22*b*b*sin(a*x+d)*sin(b*y+e);
    } else if (dim==3) {
        const real x = pos[0], y = pos[1], z = pos[2];

        source[ISTATE] =   diff_coeff*a11*a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f)
                         - diff_coeff*a12*a*b*cos(a*x+d)*cos(b*y+e)*sin(c*z+f)
                         - diff_coeff*a13*a*c*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                         - diff_coeff*a21*b*a*cos(a*x+d)*cos(b*y+e)*sin(c*z+f)
                         + diff_coeff*a22*b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f)
                         - diff_coeff*a23*b*c*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                         - diff_coeff*a31*c*a*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                         - diff_coeff*a32*c*b*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                         + diff_coeff*a33*c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
    }
    return source;
}


template class Diffusion < PHILIP_DIM, 1, double >;
template class Diffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;

} // Physics namespace
} // PHiLiP namespace


