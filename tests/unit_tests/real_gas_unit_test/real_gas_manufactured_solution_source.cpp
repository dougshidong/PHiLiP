#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>


#include <assert.h>
#include <deal.II/grid/grid_generator.h>

#include "assert_compare_array.h"
#include "parameters/parameters.h"
#include "physics/real_gas.h"

const double TOLERANCE = 1E-5;

int main (int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nspecies = PHILIP_SPECIES;
    const int nstate = dim+nspecies+1;

    //default parameters
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler); // default fills options
    PHiLiP::Parameters::AllParameters all_parameters;

    if (nspecies == 2)
        all_parameters.chemistry_input_file = "../../chemistry_files/O2_N2.kinetics";
    else if (nspecies == 3)
        all_parameters.chemistry_input_file = "../../chemistry_files/H2_O2_N2.kinetics";

    all_parameters.parse_parameters (parameter_handler);
    PHiLiP::Physics::RealGas<dim, nspecies, nstate, double> real_gas_physics = PHiLiP::Physics::RealGas<dim, nspecies, nstate, double>(&all_parameters);

    const double min = 0.0;
    const double max = 1.0;
    const int nx = 11;

    const double perturbation = 1e-5;

    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);

    std::array<double, dim+nspecies+1> soln_plus;
    std::array<double, dim+nspecies+1> soln_mins;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins;
    


    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {

            const dealii::Point<dim,double> vertex = cell->vertex(v);
            constexpr double dummy_time = 0;
            std::array<double, dim+nspecies+1> source_term = real_gas_physics.source_term(vertex, soln_plus, dummy_time);

            std::array<double, dim+nspecies+1> divergence_finite_differences;
            divergence_finite_differences.fill(0.0);

            for (int d=0; d<dim; d++) {
                dealii::Point<dim,double> vertex_plus = vertex;
                dealii::Point<dim,double> vertex_mins = vertex;
                vertex_plus[d] = vertex[d] + perturbation;
                vertex_mins[d] = vertex[d] - perturbation;
                for (int s=0; s<nstate; s++) {
                    soln_plus[s] = real_gas_physics.manufactured_solution_function->value(vertex_plus, s);
                    soln_mins[s] = real_gas_physics.manufactured_solution_function->value(vertex_mins, s);
                }
                conv_flux_plus = real_gas_physics.convective_flux(soln_plus);
                conv_flux_mins = real_gas_physics.convective_flux(soln_mins);

                for (int s=0; s<nstate; s++) {
                    divergence_finite_differences[s] += (conv_flux_plus[s][d] - conv_flux_mins[s][d]) / (2.0 * perturbation);
                }
            }

            assert_compare_array<nstate> ( divergence_finite_differences, source_term, 1.0, TOLERANCE);
        }
    }
    MPI_Finalize();
    return 0;
}

