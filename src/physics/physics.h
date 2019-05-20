#ifndef __PHYSICS__
#define __PHYSICS__

#include <deal.II/base/tensor.h>

#include "parameters/all_parameters.h"

namespace PHiLiP
{
    using namespace dealii;

    /** Partial differential equation is given by the divergence of the convective and
     *  diffusive flux equal to the source term
     *
     *  \f$ \boldsymbol{\nabla} \cdot
     *         (  \mathbf{F}_{conv}( u ) 
     *          + \mathbf{F}_{diss}( u, \boldsymbol{\nabla}(u) )
     *      = s(\mathbf{x}) \f$
     */
    template <int dim, int nstate, typename real>
    class Physics
    {
    public:
        virtual ~Physics() = 0;

        /// Default manufactured solution
        ///~~~~~{.cpp}
        /// if (dim==1) uexact = sin(a*pos[0]+d);
        /// if (dim==2) uexact = sin(a*pos[0]+d)*sin(b*pos[1]+e);
        /// if (dim==3) uexact = sin(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);
        ///~~~~~
        virtual void manufactured_solution (
            const Point<dim,double> &pos,
            std::array<real,nstate> &solution) const;
        virtual void manufactured_gradient (
            const Point<dim,double> &pos,
            std::array<Tensor<1,dim,real>,nstate> &solution_gradient) const;

        /// Returns the integral of the manufactured solution over the hypercube [0,1]
        ///
        /// Either returns the linear output $\int u dV$.
        /// Or the nonlinear output $\int u^2 dV$.
        virtual double integral_output (const bool linear) const;

        // Convective fluxes that will be differentiated once in space
        virtual void convective_flux (
            const std::array<real,nstate> &solution,
            std::array<Tensor<1,dim,real>,nstate> &conv_flux) const = 0;

        // Spectral radius of convective term Jacobian
        // Used for scalar dissipation
        virtual std::array<real,nstate> convective_eigenvalues (
            const std::array<real,nstate> &/*solution*/,
            const Tensor<1,dim,real> &/*normal*/) const = 0;

        // Evaluate the diffusion matrix $ A $ such that $F_v = A \nabla u$
        virtual std::array<Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const = 0;

        // Dissipative fluxes that will be differentiated once in space
        // Evaluates the dissipative flux through the linearization F = A(u)*grad(u)
        void dissipative_flux_A_gradu (
            const real scaling,
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
            std::array<Tensor<1,dim,real>,nstate> &diss_flux) const;

        // Dissipative fluxes that will be differentiated once in space
        virtual void dissipative_flux (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
            std::array<Tensor<1,dim,real>,nstate> &diss_flux) const = 0;

        // Source term that does not require differentiation
        virtual void source_term (
            const Point<dim,double> &pos,
            const std::array<real,nstate> &solution,
            std::array<real,nstate> &source) const = 0;

        // Evaluates boundary values and gradients on the other side of the face
        virtual void boundary_face_values (
            const int /*boundary_type*/,
            const Point<dim, double> &/*pos*/,
            const Tensor<1,dim,real> &/*normal*/,
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
    protected:
        virtual void set_manufactured_dirichlet_boundary_condition (
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;
        virtual void set_manufactured_neumann_boundary_condition (
            const std::array<real,nstate> &/*soln_int*/,
            const std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
            std::array<real,nstate> &/*soln_bc*/,
            std::array<Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

        // Some constants used to define manufactured solution
        //const double freq_x = 1*1.59/dim, freq_y = 2*1.81/dim, freq_z = 3*1.76/dim;
        //const double offs_x = 1, offs_y = 1.2, offs_z = 1.5;

        const double pi = atan(1)*4.0;
        const double freq_x = 0.59/dim, freq_y = 2*0.81/dim,    freq_z = 3*0.76/dim;
        const double offs_x = 1,        offs_y = 1.2,           offs_z = 1.5;
        const double velo_x = exp(1)/2, velo_y =-pi/4.0,        velo_z = sqrt(2);
        //const double velo_x = 1.0, velo_y =-pi/4.0,        velo_z = sqrt(2);
        const double diff_coeff = 50.0;
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
        ~LinearAdvection () {};
        // Convective flux:  c*u
        void convective_flux (
            const std::array<real,nstate> &solution,
            std::array<Tensor<1,dim,real>,nstate> &conv_flux) const;

        // Spectral radius of convective term Jacobian is simply the maximum 'c'
        std::array<real,nstate> convective_eigenvalues (
            const std::array<real,nstate> &/*solution*/,
            const Tensor<1,dim,real> &normal) const;

        // Diffusion matrix: 0
        std::array<Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const;

        // Dissipative flux: 0
        void dissipative_flux (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
            std::array<Tensor<1,dim,real>,nstate> &diss_flux) const;

        // Source term is zero or depends on manufactured solution
        void source_term (
            const Point<dim,double> &pos,
            const std::array<real,nstate> &solution,
            std::array<real,nstate> &source) const;

    protected:
        // Linear advection speed:  c
        Tensor<1,dim,real> advection_speed () const;

    };

    template <int dim, int nstate, typename real>
    class Diffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // Fconv = 0
        // Fdiss = -grad(u)
        // -\nabla \cdot (\nabla u) = source

    public:
        ~Diffusion () {};
        // Convective flux:  0
        void convective_flux (
            const std::array<real,nstate> &solution,
            std::array<Tensor<1,dim,real>,nstate> &conv_flux) const;

        // Convective eigenvalues dotted with normal
        std::array<real,nstate> convective_eigenvalues (
            const std::array<real,nstate> &/*solution*/,
            const Tensor<1,dim,real> &/*normal*/) const;

        // Diffusion matrix is identity
        std::array<Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const;

        // Dissipative flux: u
        void dissipative_flux (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
            std::array<Tensor<1,dim,real>,nstate> &diss_flux) const;

        // Source term is zero or depends on manufactured solution
        void source_term (
            const Point<dim,double> &pos,
            const std::array<real,nstate> &solution,
            std::array<real,nstate> &source) const;
    };

    template <int dim, int nstate, typename real>
    class ConvectionDiffusion : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // Fconv = u
        // Fdiss = -grad(u)
        // \nabla \cdot (c*u) -\nabla \cdot (\nabla u) = source

    public:
        ~ConvectionDiffusion () {};
        // Convective flux:  0
        void convective_flux (
            const std::array<real,nstate> &solution,
            std::array<Tensor<1,dim,real>,nstate> &conv_flux) const;

        // Spectral radius of convective term Jacobian is 'c'
        std::array<real,nstate> convective_eigenvalues (
            const std::array<real,nstate> &/*solution*/,
            const Tensor<1,dim,real> &/*normal*/) const;

        // Diffusion matrix is identity
        std::array<Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const;

        // Dissipative flux: u
        void dissipative_flux (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
            std::array<Tensor<1,dim,real>,nstate> &diss_flux) const;

        // Source term is zero or depends on manufactured solution
        void source_term (
            const Point<dim,double> &pos,
            const std::array<real,nstate> &solution,
            std::array<real,nstate> &source) const;

    protected:
        // Linear advection speed:  c
        Tensor<1,dim,real> advection_speed () const;
    };

    template <int dim, int nstate, typename real>
    class LinearAdvectionVectorValued : public Physics <dim, nstate, real>
    {
        // State variable:   u
        // -\nabla \cdot (c*u) = source

    public:
        ~LinearAdvectionVectorValued () {};
        // Convective flux:  c*u
        void convective_flux (
            const std::array<real,nstate> &solution,
            std::array<Tensor<1,dim,real>,nstate> &conv_flux) const;

        // Spectral radius of convective term Jacobian is simply the maximum 'c'
        std::array<real,nstate> convective_eigenvalues (
            const std::array<real,nstate> &/*solution*/,
            const Tensor<1,dim,real> &normal) const;

        // Diffusion matrix: 0
        std::array<Tensor<1,dim,real>,nstate> apply_diffusion_matrix (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const;

        // Dissipative flux: 0
        void dissipative_flux (
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
            std::array<Tensor<1,dim,real>,nstate> &diss_flux) const;

        // Source term is zero or depends on manufactured solution
        void source_term (
            const Point<dim,double> &pos,
            const std::array<real,nstate> &solution,
            std::array<real,nstate> &source) const;

    protected:
        // Linear advection speed:  c
        Tensor<1,dim,real> advection_speed () const;

    };

} // end of PHiLiP namespace

#endif
