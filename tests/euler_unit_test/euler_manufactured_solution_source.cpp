#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>


#include <assert.h>
#include <deal.II/grid/grid_generator.h>

#include "parameters/parameters.h"
#include "physics/physics.h"

const double TOLERANCE = 1E-7;

template<int dim, int nstate>
void assert_compare_array ( const std::array<double, nstate> &array1, const std::array<double, nstate> &array2, double scale2)
{
    for (int s=0; s<nstate; s++) {
        const double diff = std::abs(array1[s] - scale2*array2[s]);
        std::cout
            << "State " << s+1 << " out of " << nstate
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


int main (int /*argc*/, char * /*argv*/[])
{
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    PHiLiP::Physics::Euler<dim, nstate, double> euler_physics = PHiLiP::Physics::Euler<dim, nstate, double>();

    const double min = -10.0;
    const double max = 10.0;
    const int nx = 11;

    const double perturbation = 1e-6;

    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);

    std::array<double, dim+2> soln_plus;
    std::array<double, dim+2> soln_minus;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus;
    std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_minus;
    


    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {

            const dealii::Point<dim,double> vertex = cell->vertex(v);
            std::array<double, dim+2> source_term = euler_physics.source_term(vertex, soln_plus);

            std::array<double, dim+2> divergence_finite_differences;
            divergence_finite_differences.fill(0.0);

            for (int d=0; d<dim; d++) {
                dealii::Point<dim,double> vertex_perturb = cell->vertex(v);
                vertex_perturb[d] = vertex[d] + perturbation;
                soln_plus = euler_physics.manufactured_solution(vertex_perturb);
                conv_flux_plus = euler_physics.convective_flux(soln_plus);

                vertex_perturb[d] = vertex[d] - perturbation;
                soln_minus = euler_physics.manufactured_solution(vertex_perturb);
                conv_flux_minus = euler_physics.convective_flux(soln_minus);

                for (int s=0; s<nstate; s++) {
                    divergence_finite_differences[s] += (conv_flux_plus[s][d] - conv_flux_minus[s][d]) / (2.0 * perturbation);
                }
            }

            assert_compare_array<dim,nstate> ( divergence_finite_differences, source_term, 1.0);
        }
    }
    return 0;
}

