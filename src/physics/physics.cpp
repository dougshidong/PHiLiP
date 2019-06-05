#include <assert.h>
#include <cmath>
#include <vector>

#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_math.h>
#include <deal.II/differentiation/ad/sacado_number_types.h>
#include <deal.II/differentiation/ad/sacado_product_types.h>

#include "physics.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>* // returns points to base class PhysicsBase
PhysicsFactory<dim,nstate,real>
::create_Physics(Parameters::AllParameters::PartialDifferentialEquation pde_type)
{
    using PDE_enum = Parameters::AllParameters::PartialDifferentialEquation;

    if (pde_type == PDE_enum::advection || pde_type == PDE_enum::advection_vector) {
        if constexpr (nstate<=2) return new LinearAdvection<dim,nstate,real>;
    } else if (pde_type == PDE_enum::diffusion) {
        if constexpr (nstate==1) return new Diffusion<dim,nstate,real>;
    } else if (pde_type == PDE_enum::convection_diffusion) {
        if constexpr (nstate==1) return new ConvectionDiffusion<dim,nstate,real>;
    } else if (pde_type == PDE_enum::euler) {
        if constexpr (nstate==dim+2) return new Euler<dim,nstate,real>;
    }
    std::cout << "Can't create PhysicsBase, invalid PDE type: " << pde_type << std::endl;
    assert(0==1 && "Can't create PhysicsBase, invalid PDE type");
    return nullptr;
}


template <int dim, int nstate, typename real>
PhysicsBase<dim,nstate,real>::~PhysicsBase() {}

// Common manufactured solution for advection, diffusion, convection-diffusion
template <int dim, int nstate, typename real>
std::array<real,nstate> PhysicsBase<dim,nstate,real>
::manufactured_solution (const dealii::Point<dim,double> &pos) const
//::manufactured_solution (const dealii::Point<dim,double> &pos, real *const solution) const
{
    std::array<real,nstate> solution;
    using phys = PhysicsBase<dim,nstate,real>;
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
    return solution;
}
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> PhysicsBase<dim,nstate,real>
::manufactured_gradient (const dealii::Point<dim,double> &pos) const
{
    std::array<dealii::Tensor<1,dim,real>,nstate> solution_gradient;
    using phys = PhysicsBase<dim,nstate,real>;
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
    return solution_gradient;
}

template <int dim, int nstate, typename real>
double PhysicsBase<dim,nstate,real>
::integral_output (bool linear) const
{
    using phys = PhysicsBase<dim,nstate,real>;
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
void PhysicsBase<dim,nstate,real>
::boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, double> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{
}
template <int dim, int nstate, typename real>
void PhysicsBase<dim,nstate,real>
::set_manufactured_dirichlet_boundary_condition (
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{}
template <int dim, int nstate, typename real>
void PhysicsBase<dim,nstate,real>
::set_manufactured_neumann_boundary_condition (
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const
{}


template class PhysicsBase < PHILIP_DIM, 1, double >;
template class PhysicsBase < PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 2, double >;
template class PhysicsBase < PHILIP_DIM, 2, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 3, double >;
template class PhysicsBase < PHILIP_DIM, 3, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 4, double >;
template class PhysicsBase < PHILIP_DIM, 4, Sacado::Fad::DFad<double> >;
template class PhysicsBase < PHILIP_DIM, 5, double >;
template class PhysicsBase < PHILIP_DIM, 5, Sacado::Fad::DFad<double> >;

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

} // Physics namespace
} // PHiLiP namespace

