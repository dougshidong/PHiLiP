#include "physics.h"

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
LinearAdvection<dim,nstate,real>::LinearAdvection ()
{
    static_assert(nstate==1 || nstate==2, "Physics::LinearAdvection() should be created with nstate=dim+2");
}

template <int dim, int nstate, typename real>
dealii::Tensor<1,dim,real> LinearAdvection<dim,nstate,real>
::advection_speed () const
{
    dealii::Tensor<1,dim,real> advection_speed;

    if(dim >= 1) advection_speed[0] = this->velo_x;
    if(dim >= 2) advection_speed[1] = this->velo_y;
    if(dim >= 3) advection_speed[2] = this->velo_z;

    return advection_speed;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> LinearAdvection<dim,nstate,real>
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
real LinearAdvection<dim,nstate,real>
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
std::array<dealii::Tensor<1,dim,real>,nstate> LinearAdvection<dim,nstate,real>
::convective_flux (const std::array<real,nstate> &solution) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> conv_flux;
    // Assert conv_flux dimensions
    const dealii::Tensor<1,dim,real> velocity_field = this->advection_speed();
    for (int i=0; i<nstate; ++i) {
        conv_flux[i] = velocity_field * solution[i];
    }
    return conv_flux;
}

template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> LinearAdvection<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*solution*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> diss_flux;
    // No dissipation
    for (int i=0; i<nstate; i++) {
        diss_flux[i] = 0;
    }
    return diss_flux;
}

template <int dim, int nstate, typename real>
std::array<real,nstate> LinearAdvection<dim,nstate,real>
::source_term (
    const dealii::Point<dim,double> &pos,
    const std::array<real,nstate> &/*solution*/) const
{
    std::array<real,nstate> source;
    const dealii::Tensor<1,dim,real> vel = this->advection_speed();
    using phys = PhysicsBase<dim,nstate,real>;
    const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
    const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

    int istate = 0;
    if (dim==1) {
        const real x = pos[0];
        source[istate] = vel[0]*a*cos(a*x+d);
    } else if (dim==2) {
        const real x = pos[0], y = pos[1];
        source[istate] = vel[0]*a*cos(a*x+d)*sin(b*y+e) +
                         vel[1]*b*sin(a*x+d)*cos(b*y+e);
    } else if (dim==3) {
        const real x = pos[0], y = pos[1], z = pos[2];
        source[istate] =  vel[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                          vel[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                          vel[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f);
    }

    if (nstate > 1) {
        int istate = 1;
        if (dim==1) {
            const real x = pos[0];
            source[istate] = -vel[0]*a*sin(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source[istate] = - vel[0]*a*sin(a*x+d)*cos(b*y+e)
                             - vel[1]*b*cos(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];
            source[istate] =  - vel[0]*a*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                              - vel[1]*b*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                              - vel[2]*c*cos(a*x+d)*cos(b*y+e)*sin(c*z+f);
        }
    }
    return source;
}

template class LinearAdvection < PHILIP_DIM, 1, double >;
template class LinearAdvection < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
template class LinearAdvection < PHILIP_DIM, 2, double >;
template class LinearAdvection < PHILIP_DIM, 2, Sacado::Fad::DFad<double>  >;


} // Physics namespace
} // PHiLiP namespace
