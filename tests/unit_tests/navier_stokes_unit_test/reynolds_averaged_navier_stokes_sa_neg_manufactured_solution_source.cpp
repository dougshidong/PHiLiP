#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>

#include "ADTypes.hpp"

#include <assert.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>

#include "physics/model.h"
#include "assert_compare_array.h"
#include "parameters/parameters.h"
#include "physics/reynolds_averaged_navier_stokes.h"
#include "physics/negative_spalart_allmaras_rans_model.h"
#include "physics/manufactured_solution.h"

const double TOLERANCE = 1E-5;

int main (int /*argc*/, char * /*argv*/[])
{
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = dim+3;
    
    //const double ref_length = 1.0, mach_inf=1.0, angle_of_attack = 0.0, side_slip_angle = 0.0, gamma_gas = 1.4;
    //const double prandtl_number = 0.72, reynolds_number_inf=1e-4;
    //const double turbulent_prandtl_number = 0.9;
    const double a = 1.0 , b = 0.0, c = 1.4, d=0.72, e=1.0e3, f=0.9; // Note: e==Re_inf --> set to small value so that viscous terms dominate
    PHiLiP::Physics::ReynoldsAveragedNavierStokes_SAneg<dim, nstate, double> rans_sa_neg_physics = PHiLiP::Physics::ReynoldsAveragedNavierStokes_SAneg<dim, nstate, double>(a,c,a,b,b,d,e,f);
    
    const double min = 0.0;
    const double max = 1.0;
    const int nx = 11;
    
    const dealii::types::global_dof_index cell_index=0;
    
    const double perturbation = 1e-5;
    
    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);
    
    // Fourth-order accurate, central finite difference
    std::array<double, nstate> soln_plus; // i+1
    std::array<double, nstate> soln_mins; // i-1
    std::array<double, nstate> soln_plus2; // i+2
    std::array<double, nstate> soln_mins2; // i-2
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins;
    std::array<dealii::Tensor<1,dim,double>,nstate> visc_flux_plus;
    std::array<dealii::Tensor<1,dim,double>,nstate> visc_flux_mins;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus2;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins2;
    std::array<dealii::Tensor<1,dim,double>,nstate> visc_flux_plus2;
    std::array<dealii::Tensor<1,dim,double>,nstate> visc_flux_mins2;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_gradient_plus;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_gradient_mins;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_gradient_plus2;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_gradient_mins2;
    
    // Create ManufacturedSolutionFunction
    std::shared_ptr< PHiLiP::ManufacturedSolutionFunction<dim,double> > 
        manufactured_solution_function = std::make_shared<PHiLiP::ManufacturedSolutionSine<dim,double>>(nstate);

    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
    
            const dealii::Point<dim,double> vertex = cell->vertex(v);
            std::array<double, nstate> convective_dissipative_source_term = rans_sa_neg_physics.convective_dissipative_source_term(vertex, soln_plus,cell_index);
    
            std::array<double, nstate> divergence_finite_differences;
            divergence_finite_differences.fill(0.0);
    
            std::vector<dealii::Tensor<1,dim,double>> manufactured_solution_gradient_dealii(nstate);
    
            for (int d=0; d<dim; d++) {
                dealii::Point<dim,double> vertex_plus = vertex;
                dealii::Point<dim,double> vertex_mins = vertex;
                vertex_plus[d] = vertex[d] + perturbation;
                vertex_mins[d] = vertex[d] - perturbation;
                dealii::Point<dim,double> vertex_plus2 = vertex;
                dealii::Point<dim,double> vertex_mins2 = vertex;
                vertex_plus2[d] = vertex[d] + 2.0*perturbation;
                vertex_mins2[d] = vertex[d] - 2.0*perturbation;
                
                // Get manufactured solution value
                for (int s=0; s<nstate; s++) {
                    soln_plus[s] = manufactured_solution_function->value(vertex_plus, s);
                    soln_mins[s] = manufactured_solution_function->value(vertex_mins, s);
                    soln_plus2[s] = manufactured_solution_function->value(vertex_plus2, s);
                    soln_mins2[s] = manufactured_solution_function->value(vertex_mins2, s);
                }
    
                // Get manufactured solution gradients
                // - compute gradient @ vertex_plus
                manufactured_solution_function->vector_gradient(vertex_plus, manufactured_solution_gradient_dealii);
                for (int d=0;d<dim;d++) {
                    for (int s=0; s<nstate; s++) {
                        soln_gradient_plus[s][d] = manufactured_solution_gradient_dealii[s][d];
                    }
                }
                // - compute gradient @ vertex_mins
                manufactured_solution_function->vector_gradient(vertex_mins, manufactured_solution_gradient_dealii);
                for (int d=0;d<dim;d++) {
                    for (int s=0; s<nstate; s++) {
                        soln_gradient_mins[s][d] = manufactured_solution_gradient_dealii[s][d];
                    }
                }
                // - compute gradient @ vertex_plus2
                manufactured_solution_function->vector_gradient(vertex_plus2, manufactured_solution_gradient_dealii);
                for (int d=0;d<dim;d++) {
                    for (int s=0; s<nstate; s++) {
                        soln_gradient_plus2[s][d] = manufactured_solution_gradient_dealii[s][d];
                    }
                }
                // - compute gradient @ vertex_mins2
                manufactured_solution_function->vector_gradient(vertex_mins2, manufactured_solution_gradient_dealii);
                for (int d=0;d<dim;d++) {
                    for (int s=0; s<nstate; s++) {
                        soln_gradient_mins2[s][d] = manufactured_solution_gradient_dealii[s][d];
                    }
                }
    
                conv_flux_plus  = rans_sa_neg_physics.convective_flux(soln_plus);
                conv_flux_mins  = rans_sa_neg_physics.convective_flux(soln_mins);
                visc_flux_plus  = rans_sa_neg_physics.dissipative_flux(soln_plus,soln_gradient_plus,cell_index);
                visc_flux_mins  = rans_sa_neg_physics.dissipative_flux(soln_mins,soln_gradient_mins,cell_index);
                conv_flux_plus2 = rans_sa_neg_physics.convective_flux(soln_plus2);
                conv_flux_mins2 = rans_sa_neg_physics.convective_flux(soln_mins2);
                visc_flux_plus2 = rans_sa_neg_physics.dissipative_flux(soln_plus2,soln_gradient_plus2,cell_index);
                visc_flux_mins2 = rans_sa_neg_physics.dissipative_flux(soln_mins2,soln_gradient_mins2,cell_index);
    
                for (int s=0; s<nstate; s++) {
                    divergence_finite_differences[s] += (-0.25*conv_flux_plus2[s][d] + 2.0*conv_flux_plus[s][d] - 2.0*conv_flux_mins[s][d] + 0.25*conv_flux_mins2[s][d]) / (3.0 * perturbation);
                    divergence_finite_differences[s] += (-0.25*visc_flux_plus2[s][d] + 2.0*visc_flux_plus[s][d] - 2.0*visc_flux_mins[s][d] + 0.25*visc_flux_mins2[s][d]) / (3.0 * perturbation);
                }
            }
    
            assert_compare_array<nstate> ( divergence_finite_differences, convective_dissipative_source_term, 1.0, TOLERANCE);
        }
    }
    return 0;
}
