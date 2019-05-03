#ifndef __PHYSICS__
#define __PHYSICS__

#include <deal.II/base/tensor.h>

#include "parameters.h"

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
       /**
         *  Purpose:
         *      Interpolate VOLUME coefficients (of type 'coef_type') to FACE cubature nodes. In the case of the solution
         *      gradients, also add the appropriate partial correction based on the selected viscous numerical flux.
         *
         *  Comments:
         *      Various options are available for the computation:
         *          1) Using sum factorized operators for TP and WEDGE elements;
         *          2) Using the standard operator but exploiting sparsity;
         *          3) Using the standard approach.
         *
         *      imex_type is only used to compute Jacobian terms for coef_type = 'Q'.
         *
         *      The CDG2 and BR2 fluxes are determined according to Brdar(2012, eq. (4.3)) following the comments of section
         *      4.1 (i.e. setting eta = 0 and taking chi according to Theorem 2 part b. The area switch (eq. (4.5)) is used
         *      such that only one of the two VOLUME contributions must be accounted for in the CDG2 flux. Note, as the
         *      viscous fluxes are linear in the gradients, that this formulation corresponds exactly to that typically
         *      presented for the BR2 flux (such as in eq. (10) in Bassi(2010)) with the stabilization parameter selected
         *      according to the guidelines above. Note also that this is the analogue of the original form of the BR2 flux
         *      (eq. (21) in Bassi(2000)) when the scaling is added. It can be shown in a few steps that the r_e
         *      contribution of Brdar(2012, eq. (2.5)) is equivalent to the FACE contribution to Q (QhatF here). Briefly, we
         *      derive the contribution of r_e to the (L)eft VOLUME below:
         *
         *      \f$\int_{\Omega} r_e([[u]]) \cdot \chi  = - \int_{\Gamma} [[u]] \cdot {{chi}}                $, Brdar(2012, eq. (2.5))
		 *
         *      \f$\int_{V_L}    r_e([[u]]) \cdot \chi_L = - \int_{\Gamma} [[u]] \cdot 0.5*(\chi_L+\chi_R)        $, Restriction to V_L
		 *
         *      \f$\int_{V_L}    r_e([[u]]) \cdot \chi_L = - \int_{\Gamma} [[u]] \cdot 0.5*(\chi_L)             $, Omitting \chi_R
		 *
         *      \f$\chi_L(R_vI)'*W_vI*J_vI*\chi_L(R_vI)*\hat{r_e}([[u]]) = -0.5*\chi_L(R_fI)'*W_fI*J_fI*[[u]]_fI$, Numerical Quadrature  
		 *
         *      \f[
         *      \begin{split}
	     *      M_L*\hat{r_e}([[u]]) &  = -0.5*\chi_L(R_fI)'*W_fI*J_fI*n_fIL*(uL-uR)_fI        \\               Def. of [[u]]
         *                           &  = \chi_L(R_fI)'*W_fI*J_fI*n_fIL*0.5*(uR-uL)_fI         \\               Rearranging
         *                           &  = \chi_L(R_fI)'*W_fI*J_fI*n_fIL*({{u}}-uL)_fI          \\               Def. of {{u}}
         *                           &  = \chi_L(R_fI)'*W_fI*J_fI*n_fIL*(uNum-uL)_fI           \\               Def. of uNum
         *         \hat{r_e}([[u]])  &  = inv(M_L)*\chi_L(R_fI)'*W_fI*J_fI*n_fIL*(uNum-uL)_fI     \\             Inverting M_L
         *         \hat{r_e}([[u]])  & := QhatL
         *      \end{split}
         *      \f]
         *
         *      It is currently unclear to me where the cost savings arise when using CDG2 flux as compared to the BR2 flux
         *      as all terms must be computed for the full contribution to Qhat used in the VOLUME term. Savings were stated
         *      as being as high as 10% in Brdar(2012). Is it possible that these savings would be seen when the scheme was
         *      directly discretized in the primal formulation? (ToBeModified)
         *
         *      It is currently uncertain whether the boundary gradients should also be corrected. Currently, they are
         *      corrected, for consistency with the internal formulation. INVESTIGATE. (ToBeModified)
         *
         *      For several of the viscous boundary conditions, including NoSlip_Dirichlet_T and NoSlip_Adiabatic (as they
         *      are currently implemented), there is no dependence of boundary values on variables other than the the
         *      variable under consideration (i.e. dWB/dWL is block diagonal). This means that only the block diagonal
         *      entries of Qhat_What, Q_What are non-zero. This is currently not exploited. (ToBeModified)
         *
         *  Notation:
         *      imex_type : (im)plicit (ex)plicit (type) indicates whether this function is being called for an implicit or
         *                  explicit computation.
         *
         *      The allowed options for coef_type are 'W' (conserved variables), 'Q' (partially corrected gradients (Qp))
         *
         *      To avoid confusion with (C)ofactor terms, FACE cubature nodes are denoted with the subscript (f)ace
         *      (I)ntegration.
         *  References:
         *      Brdar(2012)-Compact and Stable Discontinuous Galerkin Methods for Convection-Diffusion Problems
         *      Bassi(2000)-A High Order Discontinuous Galerking Method for Compressible Turbulent Flows
         *      Bassi(2010)-Very High-Order Accurate Discontinuous Galerkin Computation of Transonic Turbulent Flows on
         *                  Aeronautical Configurations - Chapter 3
         */
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
        virtual std::array<Tensor<2,dim,real>,nstate> diffusion_matrix (
            const std::array<real,nstate> &solution) const = 0;

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
        const double freq_x = 1.59/dim, freq_y = 2*1.81/dim,    freq_z = 3*1.76/dim;
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
        std::array<Tensor<2,dim,real>,nstate> diffusion_matrix (
            const std::array<real,nstate> &solution) const;

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
        std::array<Tensor<2,dim,real>,nstate> diffusion_matrix (
            const std::array<real,nstate> &solution) const;

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
        std::array<Tensor<2,dim,real>,nstate> diffusion_matrix (
            const std::array<real,nstate> &solution) const;

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

} // end of PHiLiP namespace

#endif
