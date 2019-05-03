#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"


namespace PHiLiP
{
    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim,nstate,real>
    ::convective_flux (
        const std::array<real,nstate> &solution,
        std::array<Tensor<1,dim,real>,nstate> &conv_flux) const
    {
        const Tensor<1,dim,real> velocity_field = this->advection_speed();
        for (int i=0; i<nstate; ++i) {
            conv_flux[i] = velocity_field * solution[i];
        }
    }

    template <int dim, int nstate, typename real>
    Tensor<1,dim,real> ConvectionDiffusion<dim,nstate,real>
    ::advection_speed () const
    {
        Tensor<1,dim,real> advection_speed;

        if(dim >= 1) advection_speed[0] = this->velo_x;
        if(dim >= 2) advection_speed[1] = this->velo_y;
        if(dim >= 3) advection_speed[2] = this->velo_z;
        return advection_speed;
    }

    template <int dim, int nstate, typename real>
    std::array<real,nstate> ConvectionDiffusion<dim,nstate,real>
    ::convective_eigenvalues (
        const std::array<real,nstate> &/*solution*/,
        const Tensor<1,dim,real> &normal) const
    {
        std::array<real,nstate> eig;
        const Tensor<1,dim,real> advection_speed = this->advection_speed();
        for (int i=0; i<nstate; i++) {
            eig[i] = advection_speed*normal;
        }
        return eig;
    }

    template <int dim, int nstate, typename real>
    std::array<Tensor<2,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
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
    void ConvectionDiffusion<dim,nstate,real>
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
    void ConvectionDiffusion<dim,nstate,real>
    ::source_term (
        const Point<dim,double> &pos,
        const std::array<real,nstate> &/*solution*/,
        std::array<real,nstate> &source) const
    {
        const Tensor<1,dim,real> velocity_field = this->advection_speed();
        using phys = Physics<dim,nstate,real>;
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
    }
    // Instantiate
    template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;

} // end of PHiLiP namespace


