#include <cmath>
#include <vector>

#include <Sacado.hpp>

#include "physics.h"


namespace PHiLiP
{

    template <int dim, int nstate, typename real>
    Physics<dim,nstate,real>* // returns points to base class Physics
    PhysicsFactory<dim,nstate,real>
    ::create_Physics(Parameters::AllParameters::PartialDifferentialEquation pde_type)
    {
        using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

        if (pde_type == PDE_enum::advection) {
            return new LinearAdvection<dim, nstate, real>;
        } else if (pde_type == PDE_enum::poisson) {
            return new Diffusion<dim, nstate, real>;
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return new ConvectionDiffusion<dim, nstate, real>;
        }
        std::cout << "Can't create Physics, invalid PDE type: " << pde_type << std::endl;

        return nullptr;
    }

    // Common manufactured solution for advection, diffusion, convection-diffusion
    template <int dim, int nstate, typename real>
    void Physics<dim, nstate, real>
    ::manufactured_solution (const double *const pos, real &solution)
    {
        double uexact;

        const double a = 1*0.59/dim;
        const double b = 2*0.81/dim;
        const double c = 3*0.76/dim;
        const double d = 1, e = 0.2, f = 0.5;
        if (dim==1) uexact = sin(a*pos[0]+d);
        if (dim==2) uexact = sin(a*pos[0]+d)*sin(b*pos[1]+e);
        if (dim==3) uexact = sin(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);
        solution = uexact;
    }


    // Linear advection functions
    template <int dim, int nstate, typename real>
    void LinearAdvection<dim, nstate, real>
    ::convective_flux (const real &solution, std::vector<real> &conv_flux)
    {
        // Assert conv_flux dimensions
        std::vector<real> velocity_field(dim);
        if(dim >= 1) velocity_field[0] = 1.0;
        if(dim >= 2) velocity_field[1] = 1.0;
        if(dim >= 3) velocity_field[2] = 1.0;

        
        if(dim >= 1) {
            conv_flux[0] = velocity_field[0] * solution;
        }
        if(dim >= 2) {
            conv_flux[1] = velocity_field[1] * solution;
        }
        if(dim >= 3) {
            conv_flux[2] = velocity_field[2] * solution;
        }
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim, nstate, real>
    ::dissipative_flux (const real &/*solution*/, real &/*flux*/)
    {
        return;
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim, nstate, real>
    ::source_term (double const * const pos, const real &/*solution*/, real &source)
    {
        const double a = 1*0.59/dim;
        const double b = 2*0.81/dim;
        const double c = 3*0.76/dim;
        const double d = 1, e = 0.2, f = 0.5;
        if (dim==1) {
            const double x = pos[0];
            source = a*a*sin(a*x+d);
        } else if (dim==2) {
            const double x = pos[0], y = pos[1];
            source = a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const double x = pos[0], y = pos[1], z = pos[2];

            source =  a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
    }

    // Diffusion functions
    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::convective_flux (const real &/*solution*/, std::vector<real> &/*conv_flux*/)
    {
        return;
    }

    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::dissipative_flux (const real &solution, real &flux)
    {
        flux = solution;
    }

    template <int dim, int nstate, typename real>
    void Diffusion<dim, nstate, real>
    ::source_term (double const * const pos, const real &/*solution*/, real &source)
    {
        const double a = 1*0.59/dim;
        const double b = 2*0.81/dim;
        const double c = 3*0.76/dim;
        const double d = 1, e = 0.2, f = 0.5;
        if (dim==1) {
            const double x = pos[0];
            source = a*a*sin(a*x+d);
        } else if (dim==2) {
            const double x = pos[0], y = pos[1];
            source = a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const double x = pos[0], y = pos[1], z = pos[2];

            source =  a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
    }


    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim, nstate, real>
    ::convective_flux (const real &solution, std::vector<real> &conv_flux)
    {
        // Assert conv_flux dimensions
        std::vector<real> velocity_field(dim);
        if(dim >= 1) velocity_field[0] = 1.0;
        if(dim >= 2) velocity_field[1] = 1.0;
        if(dim >= 3) velocity_field[2] = 1.0;

        
        if(dim >= 1) {
            conv_flux[0] = velocity_field[0] * solution;
        }
        if(dim >= 2) {
            conv_flux[1] = velocity_field[1] * solution;
        }
        if(dim >= 3) {
            conv_flux[2] = velocity_field[2] * solution;
        }
    }

    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim, nstate, real>
    ::dissipative_flux (const real &solution, real &flux)
    {
        flux = solution;
    }

    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim, nstate, real>
    ::source_term (double const * const pos, const real &/*solution*/, real &source)
    {
        const double a = 1*0.59/dim;
        const double b = 2*0.81/dim;
        const double c = 3*0.76/dim;
        const double d = 1, e = 0.2, f = 0.5;
        if (dim==1) {
            const double x = pos[0];
            source = a*cos(a*x+d) +
                     a*a*sin(a*x+d);
        } else if (dim==2) {
            const double x = pos[0], y = pos[1];
            source = a*cos(a*x+d)*sin(b*y+e) +
                     b*sin(a*x+d)*cos(b*y+e) +
                     a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const double x = pos[0], y = pos[1], z = pos[2];
            source =   a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                       c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f) +
                       a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
    }
    // Instantiate
    template class Physics < PHILIP_DIM, 1, double >;
    template class PhysicsFactory<PHILIP_DIM, 1, double>;
    template class LinearAdvection < PHILIP_DIM, 1, double >;
    template class Diffusion < PHILIP_DIM, 1, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 1, double >;

    template class Physics < PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class LinearAdvection < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;


} // end of PHiLiP namespace

