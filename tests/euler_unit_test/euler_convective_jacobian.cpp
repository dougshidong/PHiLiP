#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>

#include <deal.II/grid/grid_generator.h>

#include "assert_compare_array.h"
#include "parameters/parameters.h"
#include "physics/physics.h"

const double TOLERANCE = 1E-6;

int main (int /*argc*/, char * /*argv*/[])
{
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    PHiLiP::Physics::Euler<dim, nstate, double> euler_physics = PHiLiP::Physics::Euler<dim, nstate, double>();

    const double min = -10.0;
    const double max = 10.0;
    const int nx = 6;

    const double perturbation = 1e-5;

    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);

    std::array<double, dim+2> soln;
    
    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {

            const dealii::Point<dim,double> vertex = cell->vertex(v);

            for (int s=0; s<nstate; s++) {
                soln[s] = euler_physics.manufactured_solution_function.value(vertex, s);
            }
            for (int d=0; d<dim; d++) {

                dealii::Tensor<1,dim,double> normal;
                normal[d] = 1.0;

                dealii::Tensor<2,nstate,double> jacobian_an = euler_physics.convective_flux_directional_jacobian (soln, normal);

                dealii::Tensor<2,nstate,double> jacobian_fd;
                for (int col=0; col<nstate; col++) {
                    std::array<double, dim+2> soln_plus = soln;
                    std::array<double, dim+2> soln_mins = soln;
                    soln_plus[col] += perturbation;
                    soln_mins[col] -= perturbation;
                    const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus = euler_physics.convective_flux(soln_plus);
                    const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins = euler_physics.convective_flux(soln_mins);

                    for (int row=0; row<nstate; row++) {
                        jacobian_fd[row][col] = (conv_flux_plus[row][d] - conv_flux_mins[row][d]) / (2.0*perturbation);
                    }
                }

                for (int row=0; row<nstate; row++) {
                    printf("ROW %d\n", row);
                    std::array<double, nstate> fd_row, an_row;
                    for (int col=0; col<nstate; col++) {
                        fd_row[col] = jacobian_fd[row][col];
                        an_row[col] = jacobian_an[row][col];
                    }
                    assert_compare_array<nstate> ( fd_row, an_row, 1.0, TOLERANCE);
                }
            }
        }
    }
    return 0;
}

