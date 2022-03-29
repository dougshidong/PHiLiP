#include "ADTypes.hpp"

#include "burgers_rewienski.h"

namespace PHiLiP {
namespace Physics {

template <int dim, int nstate, typename real>
BurgersRewienski<dim,nstate,real>::BurgersRewienski(
        const double rewienski_a,
        const double rewienski_b,
        const bool rewienski_manufactured_solution,
        const bool convection,
        const bool diffusion,
        const dealii::Tensor<2, 3> input_diffusion_tensor,
        std::shared_ptr<ManufacturedSolutionFunction<dim, real>> manufactured_solution_function)
        : Burgers<dim, nstate, real>(convection,
                                     diffusion,
                                     input_diffusion_tensor,
                                     manufactured_solution_function)
        , rewienski_a(rewienski_a)
        , rewienski_b(rewienski_b)
        , rewienski_manufactured_solution(rewienski_manufactured_solution)
{
    static_assert(nstate==dim, "Physics::Burgers() should be created with nstate==dim");
}



template <int dim, int nstate, typename real>
void BurgersRewienski<dim,nstate,real>
::boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const
{
    std::array<real,nstate> boundary_values;
    std::array<dealii::Tensor<1,dim,real>,nstate> boundary_gradients;

    if(rewienski_manufactured_solution){
        for (int i=0; i<nstate; i++) {
            boundary_values[i] = this->manufactured_solution_function->value (pos, i);
        }
    }else{
        for (int i=0; i<nstate; i++) {
            boundary_values[i] = rewienski_a; // corresponds to 'a' in eq.(18) of reference Carlberg 2013
        }
    }

    for (int istate=0; istate<nstate; ++istate) {

        std::array<real,nstate> characteristic_dot_n = this->convective_eigenvalues(boundary_values, normal_int);
        const bool inflow = (characteristic_dot_n[istate] <= 0.);

        if (inflow || this->hasDiffusion) { // Dirichlet boundary condition
            soln_bc[istate] = boundary_values[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];

        } else { // Neumann boundary condition
            soln_bc[istate] = soln_int[istate];
            soln_grad_bc[istate] = soln_grad_int[istate];
        }
    }
}

template <int dim, int nstate, typename real>
std::array<real,nstate> BurgersRewienski<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &/*solution*/,
        const real /*current_time*/) const
{
    std::array<real,nstate> source;

    for (int istate=0; istate<nstate; istate++) {
        double b = rewienski_b; // corresponds to 'b' in eq.(18) of reference Carlberg 2013
        source[istate] = 0.02*exp(b*pos[0]);
    }
    if (rewienski_manufactured_solution){
        for (int istate=0; istate<nstate; istate++) {
            source[istate] = 0.0;
            dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, istate);
            for (int d=0;d<dim;++d) {
                real manufactured_solution = this->manufactured_solution_function->value (pos, d);
                source[istate] += 0.5*manufactured_solution*manufactured_gradient[d];
            }
        }
        for (int istate=0; istate<nstate; istate++) {
            real manufactured_solution = this->manufactured_solution_function->value (pos, istate);
            real divergence = 0.0;
            for (int d=0;d<dim;++d) {
                dealii::Tensor<1,dim,real> manufactured_gradient = this->manufactured_solution_function->gradient (pos, d);
                divergence += manufactured_gradient[d];
            }
            source[istate] += 0.5*manufactured_solution*divergence;
        }
    }
    return source;
}

template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, double >;
template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, FadType  >;
template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, RadType  >;
template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, FadFadType >;
template class BurgersRewienski < PHILIP_DIM, PHILIP_DIM, RadFadType >;

} // Physics namespace
} // PHiLiP namespace



