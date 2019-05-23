#include <deal.II/base/tensor.h>

#include "parameters/parameters.h"
#include "physics/physics.h"
#include "numerical_flux/numerical_flux.h"

using PDEType  = Parameters::AllParameters::PartialDifferentialEquation;
using ConvType = Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType = Parameters::AllParameters::DissipativeNumericalFlux;

const double TOLERANCE = 1E-12;

template<int dim, int nstate>
void set_solution (
    std::array<double, nstate>               &soln_int, 
    std::array<double, nstate>               &soln_ext,
    std::array<dealii::Tensor<1,dim,double>, nstate> &soln_grad_int,
    std::array<dealii::Tensor<1,dim,double>, nstate> &soln_grad_ext,
    dealii::Tensor<1,dim,double>                     &normal_int)
{
    for (int d=0; d<dim; d++) {
        normal_int[d] = 1;
    }

    for (int s=0; s<nstate; s++) {
        soln_int[s] = exp(s+exp(1));
        soln_ext[s] = sin(s-atan(1));

        for (int d=0; d<dim; d++) {
            soln_grad_int[s][d] = d*exp(s+d*exp(1));
            soln_grad_ext[s][d] = sin(d*s-atan(1))/(d+0.1);
        }
    }
}

template<int dim, int nstate>
void compare_array ( const std::array<double, nstate> &array1, const std::array<double, nstate> &array2, double scale2)
{
    for (int s=0; s<nstate; s++) {
        const double diff = std::abs(array1[s] - scale2*array2[s]);
        std::cout
            << "State " << s << " out of " << nstate
            << std::endl
            << "Array 1 = " << array1[s]
            << std::endl
            << "Array 2 = " << array2[s]
            << std::endl
            << "Difference = " << diff
            << std::endl;
        assert(diff < TOLERANCE);
    }
    std::cout << std::endl
              << std::endl
              << std::endl;
}

template<int dim, int nstate>
int test_dissipative_numerical_flux_conservation (const PDEType pde_type, const DissType diss_type)
{
    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    std::cout << std::scientific;


    using namespace PHiLiP;
    Physics<dim, nstate, double> *pde_physics = PhysicsFactory<dim, nstate, double>::create_Physics(pde_type);

    Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    set_solution<dim, nstate> (soln_int, soln_ext, soln_grad_int, soln_grad_ext, normal_int);

    NumericalFluxDissipative<dim, nstate, double> *diss_num_flux = 
        NumericalFluxFactory<dim, nstate, double>
        ::create_dissipative_numerical_flux (diss_type, pde_physics);
    std::array<double, nstate> diss_num_flux_dot_n_1 = diss_num_flux->evaluate_solution_flux(soln_int, soln_ext, normal_int);
    std::array<double, nstate> diss_num_flux_dot_n_2 = diss_num_flux->evaluate_solution_flux(soln_ext, soln_int, -normal_int);

    double penalty = 100;
    std::array<double, nstate> diss_auxi_num_flux_dot_n_1 = diss_num_flux->evaluate_auxiliary_flux(
                 soln_int, soln_ext,
                 soln_grad_int, soln_grad_ext,
                 normal_int, penalty);

    std::array<double, nstate> diss_auxi_num_flux_dot_n_2 = diss_num_flux->evaluate_auxiliary_flux(
                 soln_ext, soln_int,
                 soln_grad_ext, soln_grad_int,
                 -normal_int, penalty);
 

    std::cout << "Dissipative solution flux conservation (should be equal since not dotted with normal)" << std::endl;
    compare_array<dim,nstate> (diss_num_flux_dot_n_1, diss_num_flux_dot_n_2, 1.0);

    std::cout << "Dissipative auxiliary flux conservation (should be equal and opposite since it is dotted with normal)" << std::endl;
    compare_array<dim,nstate> (diss_auxi_num_flux_dot_n_1, diss_auxi_num_flux_dot_n_2, -1.0);

    delete diss_num_flux;
    delete pde_physics;

    return 0;
}

template<int dim, int nstate>
int test_dissipative_numerical_flux_consistency (const PDEType pde_type, const DissType diss_type)
{
    using namespace PHiLiP;
    Physics<dim, nstate, double> *pde_physics = PhysicsFactory<dim, nstate, double>::create_Physics(pde_type);

    NumericalFluxDissipative<dim, nstate, double> *diss_num_flux = 
        NumericalFluxFactory<dim, nstate, double>
        ::create_dissipative_numerical_flux (diss_type, pde_physics);

    Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    set_solution<dim,nstate> (soln_int, soln_ext, soln_grad_int, soln_grad_ext, normal_int);

    // Copy state to ext cell
    soln_int = soln_ext;
    soln_grad_int = soln_grad_ext;

    // Evaluate numerical fluxes
    std::array<double, nstate> diss_soln_num_flux_dot_n = diss_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);
    double penalty = 100;
    std::array<double, nstate> diss_auxi_num_flux_dot_n = diss_num_flux->evaluate_auxiliary_flux(
                 soln_int, soln_ext,
                 soln_grad_int, soln_grad_ext,
                 normal_int, penalty);


    std::array<Tensor<1,dim,double>, nstate> diss_phys_flux_int;
    pde_physics->dissipative_flux (soln_int, diss_phys_flux_int);

    std::array<double, nstate> diss_phys_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        diss_phys_flux_dot_n[s] = diss_phys_flux_int[s]*normal_int;
    }

    std::cout << "Dissipative solution flux should be consistent (equal) with solution, when same values on each side of boundary" << std::endl;
    compare_array<dim,nstate> (diss_soln_num_flux_dot_n, soln_int, 1.0);

    std::cout << "Dissipative auxiliary flux should be consistent (equal) with gradient, when same values on each side of boundary" << std::endl;
    compare_array<dim,nstate> (diss_auxi_num_flux_dot_n, diss_phys_flux_int, 1.0);

    delete diss_num_flux;
    delete pde_physics;

    return 0;
}

template<int dim, int nstate>
int test_convective_numerical_flux_conservation (const PDEType pde_type, const ConvType conv_type)
{
    using namespace PHiLiP;
    Physics<dim, nstate, double> *pde_physics = PhysicsFactory<dim, nstate, double>::create_Physics(pde_type);

    NumericalFluxConvective<dim, nstate, double> *conv_num_flux = 
        NumericalFluxFactory<dim, nstate, double>
        ::create_convective_numerical_flux (conv_type, pde_physics);

    Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    set_solution<dim,nstate> (soln_int, soln_ext, soln_grad_int, soln_grad_ext, normal_int);

    std::array<double, nstate> conv_num_flux_dot_n_1 = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);
    std::array<double, nstate> conv_num_flux_dot_n_2 = conv_num_flux->evaluate_flux(soln_ext, soln_int, -normal_int);

    std::cout << "Convective numerical flux conservation (should be equal and opposite)" << std::endl;
    compare_array<dim,nstate> (conv_num_flux_dot_n_1, conv_num_flux_dot_n_2, -1.0);

    delete conv_num_flux;
    delete pde_physics;

    return 0;
}

template<int dim, int nstate>
int test_convective_numerical_flux_consistency (const PDEType pde_type, const ConvType conv_type)
{
    using namespace PHiLiP;
    Physics<dim, nstate, double> *pde_physics = PhysicsFactory<dim, nstate, double>::create_Physics(pde_type);

    NumericalFluxConvective<dim, nstate, double> *conv_num_flux = 
        NumericalFluxFactory<dim, nstate, double>
        ::create_convective_numerical_flux (conv_type, pde_physics);

    Tensor<1,dim,double> normal_int;
    std::array<double, nstate> soln_int, soln_ext;
    std::array<Tensor<1,dim,double>, nstate> soln_grad_int, soln_grad_ext;
    set_solution<dim,nstate> (soln_int, soln_ext, soln_grad_int, soln_grad_ext, normal_int);

    // Consistent numerical flux should be equal to physical flux when both states are equal
    // Therefore, f1 - f2 = 0
    soln_int = soln_ext;
    std::array<double, nstate> conv_num_flux_dot_n = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);

    std::array<Tensor<1,dim,double>, nstate> conv_phys_flux_int;
    pde_physics->convective_flux (soln_int, conv_phys_flux_int);

    std::array<double, nstate> conv_phys_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        conv_phys_flux_dot_n[s] = conv_phys_flux_int[s]*normal_int;
    }

    std::cout << "Convective numerical flux should be consistent with physical flux (equal), when same values on each side of boundary" << std::endl;
    compare_array<dim,nstate> (conv_num_flux_dot_n, conv_phys_flux_dot_n, 1.0);

    delete conv_num_flux;
    delete pde_physics;

    return 0;
}

int main (int /*argc*/, char * /*argv*/[])
{
    std::vector<PDEType> pde_type {
        PDEType::advection,
        PDEType::diffusion,
        PDEType::convection_diffusion,
        PDEType::advection_vector
    };

    std::vector<ConvType> conv_type {
        ConvType::lax_friedrichs
    };
    std::vector<DissType> diss_type {
        DissType::symm_internal_penalty
    };

    int success = 0;
    for (auto pde = pde_type.begin(); pde != pde_type.end() && success == 0; pde++) {
        for (auto conv = conv_type.begin(); conv != conv_type.end() && success == 0; conv++) {
            success = test_convective_numerical_flux_conservation<PHILIP_DIM,1> (*pde, *conv);
        }
        for (auto diss = diss_type.begin(); diss != diss_type.end() && success == 0; diss++) {
            success = test_dissipative_numerical_flux_conservation<PHILIP_DIM,1> (*pde, *diss);
        }
    }
    return success;
}

