#ifndef __PHYSICS__
#define __PHYSICS__

#include "parameters.h"
namespace PHiLiP
{

    template <int dim, int nstate, typename real>
    class Physics
    {
    public:
        virtual void manufactured_solution (const double *const x, real &solution);

        // Convective fluxes that will be differentiated once in space
        virtual void convective_flux (const real &solution, std::vector<real> &flux) = 0;

        // Dissipative fluxes that will be differentiated once in space
        virtual void dissipative_flux (const real &solution, real &flux) = 0;

        // Source term that does not require differentiation
        virtual void source_term (const double *const x, const real &solution, real &source) = 0;
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
        void convective_flux (const real &solution, std::vector<real> &flux);

        // Dissipative flux: 0
        void dissipative_flux (const real &solution, real &flux);

        // Source term is zero or depends on manufactured solution
        void source_term (double const * const x, const real &solution, real &source);
    };

    template <int dim, int nstate, typename real>
    class Diffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // -\nabla \cdot (\nabla u) = source

        // Convective flux:  0
        void convective_flux (const real &solution, std::vector<real> &flux);

        // Dissipative flux: u
        void dissipative_flux (const real &solution, real &flux);

        // Source term is zero or depends on manufactured solution
        void source_term (double const * const x, const real &solution, real &source);
    };

    template <int dim, int nstate, typename real>
    class ConvectionDiffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // \nabla \cdot (c*u) -\nabla \cdot (\nabla u) = source

        // Convective flux:  0
        void convective_flux (const real &solution, std::vector<real> &flux);

        // Dissipative flux: u
        void dissipative_flux (const real &solution, real &flux);

        // Source term is zero or depends on manufactured solution
        void source_term (double const * const x, const real &solution, real &source);
    };

} // end of PHiLiP namespace

#endif
