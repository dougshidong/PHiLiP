#include <Sacado.hpp>
#include <deal.II/differentiation/ad/sacado_product_types.h>
#include "numerical_flux.h"

namespace PHiLiP
{
    using namespace dealii;
    using AllParam = Parameters::AllParameters;

    template <int dim, int nstate, typename real>
    NumericalFluxConvective<dim,nstate,real>*
    NumericalFluxFactory<dim, nstate, real>
    ::create_convective_numerical_flux(
        AllParam::ConvectiveNumericalFlux conv_num_flux_type,
        Physics<dim, nstate, real> *physics_input)
    {
        if(conv_num_flux_type == AllParam::lax_friedrichs) {
            return new LaxFriedrichs<dim, nstate, real>(physics_input);
        }

        return nullptr;
    }
    template <int dim, int nstate, typename real>
    NumericalFluxDissipative<dim,nstate,real>*
    NumericalFluxFactory<dim, nstate, real>
    ::create_dissipative_numerical_flux(
        AllParam::DissipativeNumericalFlux diss_num_flux_type,
        Physics<dim, nstate, real> *physics_input)
    {
        if(diss_num_flux_type == AllParam::symm_internal_penalty) {
            return new SymmetricInternalPenalty<dim, nstate, real>(physics_input);
        }

        return nullptr;
    }

    template <int dim, int nstate, typename real>
    NumericalFluxConvective<dim,nstate,real>::~NumericalFluxConvective() {}

    template<int dim, int nstate, typename real>
    std::array<real, nstate> LaxFriedrichs<dim,nstate,real>
    ::evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const Tensor<1,dim,real> &n_int) const
    {
        using RealArrayVector = std::array<Tensor<1,dim,real>,nstate>;
        RealArrayVector conv_phys_flux_int;
        RealArrayVector conv_phys_flux_ext;

        pde_physics->convective_flux (soln_int, conv_phys_flux_int);
        pde_physics->convective_flux (soln_ext, conv_phys_flux_ext);
        
        RealArrayVector flux_avg;
        for (int s=0; s<nstate; s++) {
            flux_avg[s] = 0.5*(conv_phys_flux_int[s] + conv_phys_flux_ext[s]);
        }

        const std::array<real,nstate> conv_eig_int = pde_physics->convective_eigenvalues(soln_int, n_int);
        const std::array<real,nstate> conv_eig_ext = pde_physics->convective_eigenvalues(soln_ext, n_int); // using the same normal
        std::array<real,nstate> conv_eig_max;
        for (int s=0; s<nstate; s++) {
            conv_eig_max[s] = std::max(std::abs(conv_eig_int[s]), std::abs(conv_eig_ext[s]));
        }
        // Scalar dissipation
        std::array<real, nstate> numerical_flux_dot_n;
        for (int s=0; s<nstate; s++) {
            numerical_flux_dot_n[s] = flux_avg[s]*n_int - 0.5 * conv_eig_max[s] * (soln_ext[s]-soln_int[s]);
        }

        return numerical_flux_dot_n;
    }

    template <int dim, int nstate, typename real>
    NumericalFluxDissipative<dim,nstate,real>::~NumericalFluxDissipative() {}

    template<int dim, int nstate, typename real>
    void SymmetricInternalPenalty<dim,nstate,real>
    ::evaluate_flux (
        const std::array<real, nstate> &soln_int,
        const std::array<real, nstate> &soln_ext,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_int,
        const std::array<Tensor<1,dim,real>, nstate> &soln_grad_ext,
        const Tensor<1,dim,real> &normal1,
        std::array<real, nstate> &soln_flux,
        std::array<Tensor<1,dim,real>, nstate> &grad_flux) const
    {
        return;
    }



    // Instantiation
    template class NumericalFluxConvective<PHILIP_DIM, 1, double>;
    template class NumericalFluxConvective<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;

    template class LaxFriedrichs<PHILIP_DIM, 1, double>;
    template class LaxFriedrichs<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;

    template class NumericalFluxDissipative<PHILIP_DIM, 1, double>;
    template class NumericalFluxDissipative<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;

    template class SymmetricInternalPenalty<PHILIP_DIM, 1, double>;
    template class SymmetricInternalPenalty<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;

    template class NumericalFluxFactory<PHILIP_DIM, 1, double>;
    template class NumericalFluxFactory<PHILIP_DIM, 1, Sacado::Fad::DFad<double> >;
}
