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

        if (pde_type == PDE_enum::advection || pde_type == PDE_enum::advection_vector) {
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

    //  template <int dim, int nstate, typename real>
    //  void Physics<dim,nstate,real>
    //  ::dissipative_flux_A_gradu (
    //      const real scaling,
    //      const std::array<real,nstate> &solution,
    //      const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
    //      std::array<Tensor<1,dim,real>,nstate> &dissipative_flux) const
    //  {
    //      const std::array<Tensor<1,dim,real>,nstate> dissipation = apply_diffusion_matrix(solution, solution_gradient);
    //      for (int s=0; s<nstate; s++) {
    //          dissipative_flux[s] = -scaling*dissipation[s];
    //      }
    //  }

    // Common manufactured solution for advection, diffusion, convection-diffusion
    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    //::manufactured_solution (const Point<dim,double> &pos, std::array<real,nstate> &solution) const
    ::manufactured_solution (const Point<dim,double> &pos, real *const solution) const
    {
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

        int istate = 0;
        if (dim==1) solution[istate] = sin(a*pos[0]+d);
        if (dim==2) solution[istate] = sin(a*pos[0]+d)*sin(b*pos[1]+e);
        if (dim==3) solution[istate] = sin(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);

        if (nstate > 1) {
            istate = 1;
            if (dim==1) solution[istate] = cos(a*pos[0]+d);
            if (dim==2) solution[istate] = cos(a*pos[0]+d)*cos(b*pos[1]+e);
            if (dim==3) solution[istate] = cos(a*pos[0]+d)*cos(b*pos[1]+e)*cos(c*pos[2]+f);
        }
    }
    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::manufactured_gradient (const Point<dim,double> &pos, std::array<Tensor<1,dim,real>,nstate> &solution_gradient) const
    {
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

        int istate = 0;
        if (dim==1) {
            solution_gradient[istate][0] = a*cos(a*pos[0]+d);
        } else if (dim==2) {
            solution_gradient[istate][0] = a*cos(a*pos[0]+d)*sin(b*pos[1]+e);
            solution_gradient[istate][1] = b*sin(a*pos[0]+d)*cos(b*pos[1]+e);
        } else if (dim==3) {
            solution_gradient[istate][0] = a*cos(a*pos[0]+d)*sin(b*pos[1]+e)*sin(c*pos[2]+f);
            solution_gradient[istate][1] = b*sin(a*pos[0]+d)*cos(b*pos[1]+e)*sin(c*pos[2]+f);
            solution_gradient[istate][2] = c*sin(a*pos[0]+d)*sin(b*pos[1]+e)*cos(c*pos[2]+f);
        }

        if (nstate > 1) {
            int istate = 1;
            if (dim==1) {
                solution_gradient[istate][0] = -a*sin(a*pos[0]+d);
            } else if (dim==2) {
                solution_gradient[istate][0] = -a*sin(a*pos[0]+d)*cos(b*pos[1]+e);
                solution_gradient[istate][1] = -b*cos(a*pos[0]+d)*sin(b*pos[1]+e);
            } else if (dim==3) {
                solution_gradient[istate][0] = -a*sin(a*pos[0]+d)*cos(b*pos[1]+e)*cos(c*pos[2]+f);
                solution_gradient[istate][1] = -b*cos(a*pos[0]+d)*sin(b*pos[1]+e)*cos(c*pos[2]+f);
                solution_gradient[istate][2] = -c*cos(a*pos[0]+d)*cos(b*pos[1]+e)*sin(c*pos[2]+f);
            }
        }
    }

    template <int dim, int nstate, typename real>
    double Physics<dim,nstate,real>
    ::integral_output (bool linear) const
    {
        using phys = Physics<dim,nstate,real>;
        const double a = phys::freq_x, b = phys::freq_y, c = phys::freq_z;
        const double d = phys::offs_x, e = phys::offs_y, f = phys::offs_z;

        // See integral_hypercube.m MATLAB file
        double integral = 0;
        if (dim==1) { 
            // Source from Wolfram Alpha
            // https://www.wolframalpha.com/input/?i=integrate+sin(a*x%2Bd)+dx+,+x+%3D0,1
            if(linear)  integral += (cos(d) - cos(a + d))/a;
            else        integral += (sin(2.0*d)/4.0 - sin(2.0*a + 2.0*d)/4.0)/a + 1.0/2.0;
        }
        if (dim==2) {
            // Source from Wolfram Alpha
            // https://www.wolframalpha.com/input/?i=integrate+sin(a*x%2Bd)*sin(b*y%2Be)+dx+dy,+x+%3D0,1,y%3D0,1
            if(linear)  integral += ((cos(d) - cos(a + d))*(cos(e) - cos(b + e)))/(a*b);
            else        integral += ((2.0*a + sin(2.0*d) - sin(2.0*a + 2.0*d)) *(2.0*b + sin(2.0*e) - sin(2.0*b + 2.0*e))) /(16.0*a*b);
        }
        if (dim==3) {
            // Source from Wolfram Alpha
            // https://www.wolframalpha.com/input/?i=integrate+sin(a*x%2Bd)*sin(b*y%2Be)*sin(c*z%2Bf)++dx+dy+dz,+x+%3D0,1,y%3D0,1,z%3D0,1
            if(linear)  integral += ( 4.0*(cos(f) - cos(c + f)) * sin(a/2.0)*sin(b/2.0)*sin(a/2.0 + d)*sin(b/2.0 + e) ) /(a*b*c);
            else        integral += ((2.0*a + sin(2.0*d) - sin(2.0*a + 2.0*d)) *(2.0*b + sin(2.0*e) - sin(2.0*b + 2.0*e)) *(2.0*c + sin(2.0*f) - sin(2.0*c + 2.0*f))) /(64.0*a*b*c);
        }

        //std::cout << "NSTATE   " << nstate << std::endl;
        //if (nstate > 1) {
        //    std::cout << "Adding 2nd state variable to integral output" << std::endl;
        //    if (dim==1) { 
        //        if(linear)  integral += (sin(a + d) - sin(d))/a;
        //        else        integral += 0.5 - (sin(2.0*d)/4 - sin(2.0*a + 2.0*d)/4.0)/a;
        //    }
        //    if (dim==2) {
        //        if(linear)  integral += ((sin(a + d) - sin(d))*(sin(b + e) - sin(e)))/(a*b);
        //        else        integral += ((2.0*a - sin(2.0*d) + sin(2.0*a + 2.0*d))*(2.0*b - sin(2.0*e) + sin(2.0*b + 2.0*e)))/(16.0*a*b);
        //    }
        //    if (dim==3) {
        //        if(linear)  integral += -((cos(c + f) - cos(f))*(sin(a + d) - sin(d))*(sin(b + e) - sin(e)))/(a*b*c);
        //        else        integral += ((2.0*a - sin(2.0*d) + sin(2.0*a + 2.0*d))*(2.0*b - sin(2.0*e) + sin(2.0*b + 2.0*e))*(2.0*c + sin(2.0*f) - sin(2.0*c + 2.0*f)))/(64.0*a*b*c);
        //    }
        //}
        return integral;
    }

    template <int dim, int nstate, typename real>
    void Physics<dim,nstate,real>
    ::boundary_face_values (
            const int /*boundary_type*/,
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

    //  template <int dim, int nstate, typename real>
    //  std::array<Tensor<1,dim,real>,nstate> LinearAdvection<dim,nstate,real>
    //  ::apply_diffusion_matrix(
    //          const std::array<real,nstate> &/*solution*/,
    //          const std::array<Tensor<1,dim,real>,nstate> &/*solution_grad*/) const
    //  {
	//  	std::array<Tensor<1,dim,real>,nstate> zero; // deal.II tensors are initialized with zeros
    //      return zero;
    //  }

    template <int dim, int nstate, typename real>
    void LinearAdvection<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &/*solution*/,
        const std::array<Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
        std::array<Tensor<1,dim,real>,nstate> &diss_flux) const
    {
        // No dissipation
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

        int istate = 0;
        if (dim==1) {
            const real x = pos[0];
            source[istate] = vel[0]*a*cos(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source[istate] = vel[0]*a*cos(a*x+d)*sin(b*y+e) +
                             vel[1]*b*sin(a*x+d)*cos(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];
            source[istate] =  vel[0]*a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                              vel[1]*b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                              vel[2]*c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f);
        }

        if (nstate > 1) {
            int istate = 1;
            if (dim==1) {
                const real x = pos[0];
                source[istate] = -vel[0]*a*sin(a*x+d);
            } else if (dim==2) {
                const real x = pos[0], y = pos[1];
                source[istate] = - vel[0]*a*sin(a*x+d)*cos(b*y+e)
                                 - vel[1]*b*cos(a*x+d)*sin(b*y+e);
            } else if (dim==3) {
                const real x = pos[0], y = pos[1], z = pos[2];
                source[istate] =  - vel[0]*a*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                                  - vel[1]*b*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                                  - vel[2]*c*cos(a*x+d)*cos(b*y+e)*sin(c*z+f);
            }
        }
    }

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

    //  template <int dim, int nstate, typename real>
    //  std::array<Tensor<1,dim,real>,nstate> Diffusion<dim,nstate,real>
    //  ::apply_diffusion_matrix(
    //          const std::array<real,nstate> &/*solution*/,
    //          const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const
    //  {
    //      // deal.II tensors are initialized with zeros
    //      std::array<Tensor<1,dim,real>,nstate> diffusion;
    //      for (int d=0; d<dim; d++) {
    //          diffusion[0][d] = 1.0*solution_grad[0][d];
    //      }
    //      return diffusion;
    //  }

    template <int dim, int nstate, typename real>
    void Diffusion<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &/*solution*/,
        const std::array<Tensor<1,dim,real>,nstate> &solution_gradient,
        std::array<Tensor<1,dim,real>,nstate> &diss_flux) const
    {
        const double diff_coeff = this->diff_coeff;

        using phys = Physics<dim,nstate,real>;
        //const double a11 = 10*phys::freq_x, a12 = phys::freq_y, a13 = phys::freq_z;
        //const double a21 = phys::offs_x, a22 = 10*phys::offs_y, a23 = phys::offs_z;
        //const double a31 = phys::velo_x, a32 = phys::velo_y, a33 = 10*phys::velo_z;

        const double a11 = phys::A11, a12 = phys::A12, a13 = phys::A13;
        const double a21 = phys::A11, a22 = phys::A22, a23 = phys::A23;
        const double a31 = phys::A11, a32 = phys::A32, a33 = phys::A33;
        for (int i=0; i<nstate; i++) {
            //diss_flux[i] = -diff_coeff*1.0*solution_gradient[i];
            if (dim==1) {
                diss_flux[i] = -diff_coeff*a11*solution_gradient[i];
            } else if (dim==2) {
                diss_flux[i][0] = -diff_coeff*a11*solution_gradient[i][0]
                                  -diff_coeff*a12*solution_gradient[i][1];
                diss_flux[i][1] = -diff_coeff*a21*solution_gradient[i][0]
                                  -diff_coeff*a22*solution_gradient[i][1];
            } else if (dim==3) {
                diss_flux[i][0] = -diff_coeff*a11*solution_gradient[i][0]
                                  -diff_coeff*a12*solution_gradient[i][1]
                                  -diff_coeff*a13*solution_gradient[i][2];
                diss_flux[i][1] = -diff_coeff*a21*solution_gradient[i][0]
                                  -diff_coeff*a22*solution_gradient[i][1]
                                  -diff_coeff*a23*solution_gradient[i][2];
                diss_flux[i][2] = -diff_coeff*a31*solution_gradient[i][0]
                                  -diff_coeff*a32*solution_gradient[i][1]
                                  -diff_coeff*a33*solution_gradient[i][2];
            }

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
        //const double a11 = 10*phys::freq_x, a12 = phys::freq_y, a13 = phys::freq_z;
        //const double a21 = phys::offs_x, a22 = 10*phys::offs_y, a23 = phys::offs_z;
        //const double a31 = phys::velo_x, a32 = phys::velo_y, a33 = 10*phys::velo_z;
        const double a11 = phys::A11, a12 = phys::A12, a13 = phys::A13;
        const double a21 = phys::A11, a22 = phys::A22, a23 = phys::A23;
        const double a31 = phys::A11, a32 = phys::A32, a33 = phys::A33;
        if (dim==1) {
            const real x = pos[0];
            source[ISTATE] = diff_coeff*a11*a*a*sin(a*x+d);
        } else if (dim==2) {
            const real x = pos[0], y = pos[1];
            source[ISTATE] =   diff_coeff*a11*a*a*sin(a*x+d)*sin(b*y+e)
                             - diff_coeff*a12*a*b*cos(a*x+d)*cos(b*y+e)
                             - diff_coeff*a21*b*a*cos(a*x+d)*cos(b*y+e)
                             + diff_coeff*a22*b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const real x = pos[0], y = pos[1], z = pos[2];

            source[ISTATE] =   diff_coeff*a11*a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f)
                             - diff_coeff*a12*a*b*cos(a*x+d)*cos(b*y+e)*sin(c*z+f)
                             - diff_coeff*a13*a*c*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                             - diff_coeff*a21*b*a*cos(a*x+d)*cos(b*y+e)*sin(c*z+f)
                             + diff_coeff*a22*b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f)
                             - diff_coeff*a23*b*c*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                             - diff_coeff*a31*c*a*cos(a*x+d)*sin(b*y+e)*cos(c*z+f)
                             - diff_coeff*a32*c*b*sin(a*x+d)*cos(b*y+e)*cos(c*z+f)
                             + diff_coeff*a33*c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
    }

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

    //  template <int dim, int nstate, typename real>
    //  std::array<Tensor<1,dim,real>,nstate> ConvectionDiffusion<dim,nstate,real>
    //  ::apply_diffusion_matrix(
    //          const std::array<real,nstate> &/*solution*/,
    //          const std::array<Tensor<1,dim,real>,nstate> &solution_grad) const
    //  {
    //      // deal.II tensors are initialized with zeros
    //      std::array<Tensor<1,dim,real>,nstate> diffusion;
    //      for (int d=0; d<dim; d++) {
    //          diffusion[0][d] = 1.0*solution_grad[0][d];
    //      }
    //      return diffusion;
    //  }

    template <int dim, int nstate, typename real>
    void ConvectionDiffusion<dim,nstate,real>
    ::dissipative_flux (
        const std::array<real,nstate> &/*solution*/,
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
    // Instantiate explicitly

    template class Physics < PHILIP_DIM, 1, double >;
    template class Physics < PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class Physics < PHILIP_DIM, 2, double >;
    template class Physics < PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
    template class Physics < PHILIP_DIM, 3, double >;
    template class Physics < PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
    template class Physics < PHILIP_DIM, 4, double >;
    template class Physics < PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
    template class Physics < PHILIP_DIM, 5, double >;
    template class Physics < PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

    template class LinearAdvection < PHILIP_DIM, 1, double >;
    template class LinearAdvection < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class LinearAdvection < PHILIP_DIM, 2, double >;
    template class LinearAdvection < PHILIP_DIM, 2, Sacado::Fad::DFad<double>  >;
    template class LinearAdvection < PHILIP_DIM, 3, double >;
    template class LinearAdvection < PHILIP_DIM, 3, Sacado::Fad::DFad<double>  >;
    template class LinearAdvection < PHILIP_DIM, 4, double >;
    template class LinearAdvection < PHILIP_DIM, 4, Sacado::Fad::DFad<double>  >;
    template class LinearAdvection < PHILIP_DIM, 5, double >;
    template class LinearAdvection < PHILIP_DIM, 5, Sacado::Fad::DFad<double>  >;

    template class PhysicsFactory<PHILIP_DIM, 1, double>;
    template class PhysicsFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 2, double>;
    template class PhysicsFactory<PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 3, double>;
    template class PhysicsFactory<PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 4, double>;
    template class PhysicsFactory<PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
    template class PhysicsFactory<PHILIP_DIM, 5, double>;
    template class PhysicsFactory<PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

    template class Diffusion < PHILIP_DIM, 1, double >;
    template class Diffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 2, double >;
    template class Diffusion < PHILIP_DIM, 2, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 3, double >;
    template class Diffusion < PHILIP_DIM, 3, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 4, double >;
    template class Diffusion < PHILIP_DIM, 4, Sacado::Fad::DFad<double>  >;
    template class Diffusion < PHILIP_DIM, 5, double >;
    template class Diffusion < PHILIP_DIM, 5, Sacado::Fad::DFad<double>  >;

    template class ConvectionDiffusion < PHILIP_DIM, 1, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 1, Sacado::Fad::DFad<double>  >;
    template class ConvectionDiffusion < PHILIP_DIM, 2, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 2, Sacado::Fad::DFad<double>  >;
    template class ConvectionDiffusion < PHILIP_DIM, 3, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 3, Sacado::Fad::DFad<double>  >;
    template class ConvectionDiffusion < PHILIP_DIM, 4, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 4, Sacado::Fad::DFad<double>  >;
    template class ConvectionDiffusion < PHILIP_DIM, 5, double >;
    template class ConvectionDiffusion < PHILIP_DIM, 5, Sacado::Fad::DFad<double>  >;


} // end of PHiLiP namespace

