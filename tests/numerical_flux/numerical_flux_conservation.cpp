#include "parameters.h"
#include "physics/physics.h"
#include "numerical_flux/numerical_flux.h"

using PDEType  = Parameters::AllParameters::PartialDifferentialEquation;
using ConvType = Parameters::AllParameters::ConvectiveNumericalFlux;
using DissType = Parameters::AllParameters::DissipativeNumericalFlux;

template<int dim, int nstate>
int test_numerical_flux_conservation (const PDEType pde_type, const ConvType conv_type)
{
    const double TOLERANCE = 1E-12;
    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    std::cout << std::scientific;


    using namespace PHiLiP;
    Physics<dim, nstate, double> *pde_physics = PhysicsFactory<dim, nstate, double>::create_Physics(pde_type);

    NumericalFluxConvective<dim, nstate, double> *conv_num_flux = 
        NumericalFluxFactory<dim, nstate, double>
        ::create_convective_numerical_flux (conv_type, pde_physics);

    Tensor<1,dim,double> normal_int;
    for (int d=0; d<dim; d++) {
        normal_int[d] = 1;
    }

    std::array<double, nstate> soln_int, soln_ext;
    for (int s=0; s<nstate; s++) {
        soln_int[s] = exp(s+exp(1));
        soln_ext[s] = sin(s-atan(1));
    }

    std::array<double, nstate> conv_num_flux_dot_n_1 = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);
    std::array<double, nstate> conv_num_flux_dot_n_2 = conv_num_flux->evaluate_flux(soln_ext, soln_int, -normal_int);

    std::cout << "PDE type " << pde_type << std::endl;
    for (int s=0; s<nstate; s++) {
        // Flux should be equal and opposite, therefore f1 + f2 = 0
        const double diff = std::abs(conv_num_flux_dot_n_1[s] + conv_num_flux_dot_n_2[s]);
        std::cout
            << "State " << s << " out of " << nstate
            << std::endl
            << "Num flux 1 = " << conv_num_flux_dot_n_1[s]
            << std::endl
            << "Num flux 2 = " << conv_num_flux_dot_n_2[s]
            << std::endl
            << "Difference = " << diff
            << std::endl;
        assert(diff < TOLERANCE);
    }

    delete conv_num_flux;
    delete pde_physics;

    return 0;
}

template<int dim, int nstate>
int test_numerical_flux_consistency (const PDEType pde_type, const ConvType conv_type)
{
    const double TOLERANCE = 1E-12;
    std::cout << std::setprecision(std::numeric_limits<double>::digits10 + 1);
    std::cout << std::scientific;


    using namespace PHiLiP;
    Physics<dim, nstate, double> *pde_physics = PhysicsFactory<dim, nstate, double>::create_Physics(pde_type);

    NumericalFluxConvective<dim, nstate, double> *conv_num_flux = 
        NumericalFluxFactory<dim, nstate, double>
        ::create_convective_numerical_flux (conv_type, pde_physics);

    Tensor<1,dim,double> normal_int;
    for (int d=0; d<dim; d++) {
        normal_int[d] = 1;
    }

    std::array<double, nstate> soln_int, soln_ext;
    for (int s=0; s<nstate; s++) {
        soln_int[s] = sin(s+exp(1));
        soln_ext[s] = soln_int[s];
    }

    std::array<double, nstate> conv_num_flux_dot_n = conv_num_flux->evaluate_flux(soln_int, soln_ext, normal_int);

    std::array<Tensor<1,dim,double>, nstate> conv_phys_flux_int;
    pde_physics->convective_flux (soln_int, conv_phys_flux_int);

    std::array<double, nstate> conv_phys_flux_dot_n;
    for (int s=0; s<nstate; s++) {
        conv_phys_flux_dot_n[s] = conv_phys_flux_int[s]*normal_int;
    }


    std::cout << "PDE type " << pde_type << std::endl;
    for (int s=0; s<nstate; s++) {
        // Consistent numerical flux should be equal to physical flux when both states are equal
        // Therefore, f1 - f2 = 0
        const double diff = std::abs(conv_num_flux_dot_n[s] - conv_phys_flux_dot_n[s]);
        std::cout
            << "State " << s << " out of " << nstate
            << std::endl
            << "Num flux 1 = " << conv_num_flux_dot_n[s]
            << std::endl
            << "Num flux 2 = " << conv_phys_flux_dot_n[s]
            << std::endl
            << "Difference = " << diff
            << std::endl;
        assert(diff < TOLERANCE);
    }
    std::cout << std::endl
              << std::endl
              << std::endl;

    delete conv_num_flux;
    delete pde_physics;

    return 0;
}

int main (int argc, char *argv[])
{
    const int nstate = 0;

    std::vector<PDEType> pde_type {
        PDEType::advection,
        PDEType::diffusion,
        PDEType::convection_diffusion
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
            success = test_numerical_flux_conservation<PHILIP_DIM,1> (*pde, *conv);
            success = test_numerical_flux_consistency<PHILIP_DIM,1> (*pde, *conv);
        }
    }
    return success;
}

