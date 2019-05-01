#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"


namespace PHiLiP
{
    using AllParam = Parameters::AllParameters;

    template <int dim, int nstate, typename real>
    Physics<dim,nstate,real>* // returns points to base class Physics
    PhysicsFactory<dim,nstate,real>
    ::create_Physics(AllParam::PartialDifferentialEquation pde_type)
    {
        using PDE_enum = AllParam::PartialDifferentialEquation;

        if (pde_type == PDE_enum::advection) {
            return new LinearAdvection<dim,nstate,real>;
        } else if (pde_type == PDE_enum::diffusion) {
            return new Diffusion<dim,nstate,real>;
        } else if (pde_type == PDE_enum::convection_diffusion) {
            return new ConvectionDiffusion<dim,nstate,real>;
        }
        std::cout << "Can't create Physics, invalid PDE type: " << pde_type << std::endl;
        return nullptr;
    }


    template <int dim, int nstate, typename real>
    Physics<dim,nstate,real>::~Physics() {}

    // Common manufactured solution for advection, diffusion, convection-diffusion
    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::manufactured_solution (const Point<dim,double> &pos, std::array<real,nstate> &solution) const
    {
        std::array<real,nstate> uexact;
        
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        const int ISTATE = 0;
        if (dim==1) uexact[ISTATE] = sin(a*pos[0]+d);
        if (dim==2) uexact[ISTATE] = sin(a*pos[0]+d)*sin(b*pos[1]+e);
        if (dim==3) uexact[ISTATE] = sin(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);
        solution = uexact;
    }

    template <int dim, int nstate, typename real>
    double Physics<dim,nstate,real>
    ::integral_output (bool linear) const
    {
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

        if (dim==1) { 
            // Source from Wolfram Alpha
            // https://www.wolframalpha.com/input/?i=integrate+sin(a*x%2Bd)+dx+,+x+%3D0,1
            if(linear)  return (cos(d) - cos(a + d))/a;
            else        return (sin(2.0*d)/4.0 - sin(2.0*a + 2.0*d)/4.0)/a + 1.0/2.0;
        }
        if (dim==2) {
            // Source from Wolfram Alpha
            // https://www.wolframalpha.com/input/?i=integrate+sin(a*x%2Bd)*sin(b*y%2Be)+dx+dy,+x+%3D0,1,y%3D0,1
            if(linear)  return ((cos(d) - cos(a + d))*(cos(e) - cos(b + e)))/(a*b);
            else        return ((2.0*a + sin(2.0*d) - sin(2.0*a + 2.0*d))
                                *(2.0*b + sin(2.0*e) - sin(2.0*b + 2.0*e)))
                               /(16.0*a*b);
        }
        if (dim==3) {
            // Source from Wolfram Alpha
            // https://www.wolframalpha.com/input/?i=integrate+sin(a*x%2Bd)*sin(b*y%2Be)*sin(c*z%2Bf)++dx+dy+dz,+x+%3D0,1,y%3D0,1,z%3D0,1
            if(linear)  return ( 4.0*(cos(f) - cos(c + f))
                                 * sin(a/2.0)*sin(b/2.0)*sin(a/2.0 + d)*sin(b/2.0 + e) )
                                /(a*b*c);
            else return 
                ((2.0*a + sin(2.0*d) - sin(2.0*a + 2.0*d))
                 *(2.0*b + sin(2.0*e) - sin(2.0*b + 2.0*e))
                 *(2.0*c + sin(2.0*f) - sin(2.0*c + 2.0*f)))
                 /(64.0*a*b*c);
        }
        return 0;
    }

    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::boundary_face_values (
            const int boundary_type,
            const Point<dim, double> &/*pos*/,
            const Tensor<1,dim,real> &/*normal*/,
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
    {
    }
    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::set_manufactured_dirichlet_boundary_condition (
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
    {}
    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::set_manufactured_neumann_boundary_condition (
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
    {}


    // Instantiate
    template class Physics < PHILIP_DIM, 1, double >;
    template class Physics < PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;

    template class PhysicsFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 1, double>;

} // end of PHiLiP namespace

