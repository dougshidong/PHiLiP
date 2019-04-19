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

        // Spectral radius of convective term Jacobian
        // Used for scalar dissipation
        virtual Tensor <1, dim, real> convective_eigenvalues (const real &/*solution*/) = 0;

        // Dissipative fluxes that will be differentiated once in space
        virtual void dissipative_flux (const real & solution,
                        const Tensor<1,dim,real> &solution_gradient,
                        Tensor<1,dim,real> &diss_flux) = 0;

        // Source term that does not require differentiation
        virtual void source_term (const Point<dim,double> pos,
                                  const real &solution,
                                  real &source) = 0;
    protected:
        // Some constants used to define manufactured solution
        const double freq_x = 1*0.59/dim, freq_y = 2*0.81/dim, freq_z = 3*0.76/dim;
        const double offs_x = 1, offs_y = 0.2, offs_z = 0.5;
    };

    // This class with create a new Physics object corresponding to the pde_type
    // given as a user input
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

    public:
        // Convective flux:  c*u
        void convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux);

        // Spectral radius of convective term Jacobian is simply the maximum 'c'
        Tensor <1, dim, real> convective_eigenvalues (const real &/*solution*/);

        // Dissipative flux: 0
        void dissipative_flux (const real &solution,
                               const Tensor<1,dim,real> &solution_gradient,
                               Tensor<1,dim,real> &diss_flux);

        // Source term is zero or depends on manufactured solution
        void source_term (const Point<dim,double> pos, const real &solution, real &source);

    protected:
        // Linear advection speed:  c
        Tensor <1, dim, real> advection_speed ();

    };

    template <int dim, int nstate, typename real>
    class Diffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // Fconv = 0
        // Fdiss = -grad(u)
        // -\nabla \cdot (\nabla u) = source

    public:
        // Convective flux:  0
        void convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux);

        // Spectral radius of convective term Jacobian is 0
        Tensor <1, dim, real> convective_eigenvalues (const real &/*solution*/);

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

    public:
        // Convective flux:  0
        void convective_flux (const real &solution, Tensor <1, dim, real> &conv_flux);

        // Spectral radius of convective term Jacobian is 'c'
        Tensor <1, dim, real> convective_eigenvalues (const real &/*solution*/);

        // Dissipative flux: u
        void dissipative_flux (const real &solution,
                               const Tensor<1,dim,real> &solution_gradient,
                               Tensor<1,dim,real> &diss_flux);

        // Source term is zero or depends on manufactured solution
        void source_term (const Point<dim,double> pos, const real &solution, real &source);

    protected:
        // Linear advection speed:  c
        Tensor <1, dim, real> advection_speed ();
    };

} // end of PHiLiP namespace

#endif
