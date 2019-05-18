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

    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::dissipative_flux_A_gradu (
        const real scaling,
        const std::array<real,nstate> &solution,
        const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<Tensor<1,dim,real>,nstate> &dissipative_flux) const
    {
        const std::array<Tensor<1,dim,real>,nstate> dissipation = apply_diffusion_matrix(solution, solution_gradient);
        for (int s=0; s<nstate; s++) {
            dissipative_flux[s] = -scaling*dissipation[s];
        }
    }

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
    void Physics<dim,nstate,real>
    ::manufactured_gradient (const Point<dim,double> &pos, std::array<Tensor<1,dim,real>,nstate> &solution_gradient) const
    {
        std::array<real,nstate> uexact;
        
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        const int ISTATE = 0;
        if (dim==1) {
            solution_gradient[ISTATE][0] = a*cos(a*pos[0]+d);
        } else if (dim==2) {
            solution_gradient[ISTATE][0] = a*cos(a*pos[0]+d)*sin(b*pos[1]+e);
            solution_gradient[ISTATE][1] = b*sin(a*pos[0]+d)*cos(b*pos[1]+e);
        } else if (dim==3) {
            solution_gradient[ISTATE][0] = a*cos(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);
            solution_gradient[ISTATE][1] = b*sin(a*pos[0]+d)*cos(b*pos[1]+e)*sin(c*pos[2]+f);
            solution_gradient[ISTATE][2] = c*sin(a*pos[0]+d)*sin(b*pos[1]+e)*cos(c*pos[2]+f);
        }
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

    // Linear advection functions
    template <int dim, int nstate, typename real>
    Tensor<1,dim,real> LinearAdvection<dim,nstate,real>
    ::advection_speed () const
    {
        Tensor<1,dim,real> advection_speed;

        if(dim >= 1) advection_speed[0] = this->velo_x;
        if(dim >= 2) advection_speed[1] = this->velo_y;
        if(dim >= 3) advection_speed[2] = this->velo_z;

        return advection_speed;
    }

    template <int dim, int nstate, typename real>
    std::array<real,nstate> LinearAdvection<dim,nstate,real>
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
    void LinearAdvection<dim,nstate,real>
    ::convective_flux (
        const std::array<real,nstate> &solution,
        std::array<Tensor<1,dim,real>,nstate> &conv_flux) const
    {
        // Assert conv_flux dimensions
        const Tensor<1,dim,real> velocity_field = this->advection_speed();
        for (int i=0; i<nstate; ++i) {
            conv_flux[i] = velocity_field * solution[i];
        }
    }

    template <int dim, int nstate, typename real>
    std::array<Tensor<1,dim,real>,nstate> LinearAdvection<dim,nstate,real>
    ::apply_diffusion_matrix(
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const
    {
		std::array<Tensor<1,dim,real>,nstate> zero; // deal.II tensors are initialized with zeros
        return zero;
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<Tensor<1,dim,real>,nstate> &diss_flux) const
    {
        // No dissipation
        const double diff_coeff = this->diff_coeff;
        for (int i=0; i<nstate; i++) {
            diss_flux[i] = 0;
        }
    }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim,nstate,real>
    ::source_term (
        const Point<dim,double> &pos,
        const std::array<real,nstate> &/*solution*/,
        std::array<real,nstate> &source) const
    {
        const Tensor<1,dim,real> vel = this->advection_speed();
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;
        const int ISTATE = 0;
        if (dim==1) {
            const real x = pos[0];
            source[ISTATE] = vel[0]*a*cos(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source[ISTATE] = vel[0]*a*cos(a*x+d)*sin(b*y+e) +
                             vel[1]*b*sin(a*x+d)*cos(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];
            source[ISTATE] =  vel[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                              vel[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                              vel[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f);
        }
    }

    template class LinearAdvection < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class LinearAdvection < PHILIP_DIM, 1, double >;

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
    std::array<Tensor<1,dim,real>,nstate> Diffusion<dim,nstate,real>
    ::apply_diffusion_matrix(
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const
    {
        // deal.II tensors are initialized with zeros
        std::array<Tensor<1,dim,real>,nstate> diffusion;
        for (int d=0; d<dim; d++) {
            diffusion[0][d] = 1.0*solution_grad[0][d];
        }
        return diffusion;
    }

    template <int dim, int nstate, typename real>
    void Diffusion<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<Tensor<1,dim,real>,nstate> &diss_flux) const
    {
        const double diff_coeff = this->diff_coeff;
        for (int i=0; i<nstate; i++) {
            diss_flux[i] = -diff_coeff*1.0*solution_gradient[i];
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
    std::array<Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
    ::apply_diffusion_matrix(
            const std::array<real,nstate> &solution,
            const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const
    {
        // deal.II tensors are initialized with zeros
        std::array<Tensor<1,dim,real>,nstate> diffusion;
        for (int d=0; d<dim; d++) {
            diffusion[0][d] = 1.0*solution_grad[0][d];
        }
        return diffusion;
    }

    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &solution,
        const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<Tensor<1,dim,real>,nstate> &diss_flux) const
    {
        const double diff_coeff = this->diff_coeff;
        for (int i=0; i<nstate; i++) {
            diss_flux[i] = -diff_coeff*1.0*solution_gradient[i];
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

