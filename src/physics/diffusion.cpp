#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"

namespace PHiLiP
{
    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::convective_flux (
        const std::array<real,nstate> &/*solution*/,
        std::array<Tensor<1,dim,real>,nstate> &/*conv_flux*/) const
    { }

    template <int dim, int nstate, typename real>
    std::array<real, nstate> Diffusion<dim, nstate, real>
    ::convective_eigenvalues(
        const std::array<real,nstate> &/*solution*/,
        const Tensor<1,dim,real> &/*normal*/) const
    {
        std::array<real,nstate> eig;
        for (int i=0; i<nstate; i++) {
            eig[i] = 0;
        }
        return eig;
    }

    template <int dim, int nstate, typename real>
    std::array<Tensor<2,dim,real>,nstate> Diffusion<dim,nstate,real>
    ::diffusion_matrix ( const std::array<real,nstate> &solution ) const
    {
        // deal.II tensors are initialized with zeros
        std::array<Tensor<2,dim,real>,nstate> diff_matrix;
        for (int d=0; d<dim; d++) {
            diff_matrix[0][d][d] = 1.0;
        }
        return diff_matrix;
    }

    template <int dim, int nstate, typename real>
    void Diffusion<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<Tensor<1,dim,real>,nstate> &diss_flux) const
    {
        const double diff_coeff = this->diff_coeff;
        const std::array<Tensor<2,dim,real>,nstate> diff_matrix = diffusion_matrix(solution);
        for (int i=0; i<nstate; i++) {
            diss_flux[i] = -diff_coeff*diff_matrix[i]*solution_gradient[i];
        }
    }

    template <int dim, int nstate, typename real>
    void Diffusion<dim,nstate,real>
    ::source_term (
        const Point<dim,double> &pos,
        const std::array<real,nstate> &/*solution*/,
        std::array<real,nstate> &source) const
    {
        using phys = Physics<dim,nstate,real>;
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
    }

    template class Diffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 1, double >;

} // end of PHiLiP namespace

