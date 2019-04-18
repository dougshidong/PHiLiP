#ifndef __PHYSICS__
#define __PHYSICS__

#include <deal.II/base/tensor.h>

#include "parameters.h"

namespace PHiLiP
{
    using namespace dealii;

    template <int dim, int nstate, typename real>
    class Physics
    {
    public:
        virtual void manufactured_solution (const Point<dim,double> pos, real &solution);

        // PDE is given by
        // divergence( Fconv( u ) + Fdiss( u, grad(u) ) = Source

        // Convective fluxes that will be differentiated once in space
        virtual void convective_flux (const real &solution,
                                      Tensor <1, dim, real> &conv_flux) = 0;

        // Dissipative fluxes that will be differentiated once in space
        virtual void dissipative_flux (const real & solution,
                        const Tensor<1,dim,real> &solution_gradient,
                        Tensor<1,dim,real> &diss_flux) = 0;

        // Source term that does not require differentiation
        virtual void source_term (const Point<dim,double> pos,
                                  const real &solution,
                                  real &source) = 0;
    };

    template <int dim, int nstate, typename real>
    class PhysicsFactory
    {
    public:
        static Physics<dim,nstate,real>*
            create_Physics(Parameters::AllParameters::PartialDifferentialEquation pde_type);
    };


    template <int dim, int nstate, typename real>
    class LinearAdvection : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // -\nabla \cdot (c*u) = source

        // Convective flux:  c*u
        void convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux);

        // Dissipative flux: 0
        void dissipative_flux (const real &solution,
                               const Tensor<1,dim,real> &solution_gradient,
                               Tensor<1,dim,real> &diss_flux);

        // Source term is zero or depends on manufactured solution
        void source_term (const Point<dim,double> pos, const real &solution, real &source);
    };

    template <int dim, int nstate, typename real>
    class Diffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // Fconv = 0
        // Fdiss = -grad(u)
        // -\nabla \cdot (\nabla u) = source

        // Convective flux:  0
        void convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux);

        // Dissipative flux: u
        void dissipative_flux (const real &solution,
                               const Tensor<1,dim,real> &solution_gradient,
                               Tensor<1,dim,real> &diss_flux);

        // Source term is zero or depends on manufactured solution
        void source_term (const Point<dim,double> pos, const real &solution, real &source);
    };

    template <int dim, int nstate, typename real>
    class ConvectionDiffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // Fconv = u
        // Fdiss = -grad(u)
        // \nabla \cdot (c*u) -\nabla \cdot (\nabla u) = source

        // Convective flux:  0
        void convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux);

        // Dissipative flux: u
        void dissipative_flux (const real &solution,
                               const Tensor<1,dim,real> &solution_gradient,
                               Tensor<1,dim,real> &diss_flux);

        // Source term is zero or depends on manufactured solution
        void source_term (const Point<dim,double> pos, const real &solution, real &source);
    };

} // end of PHiLiP namespace

#endif
