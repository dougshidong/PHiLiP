#include <iomanip>
#include <cmath>
#include <limits>
#include <type_traits>

#include <deal.II/grid/grid_generator.h>

#include "assert_compare_array.h"
#include "parameters/parameters.h"
#include "physics/navier_stokes.h"

const double TOLERANCE = 1E-6;

int main (int /*argc*/, char * /*argv*/[])
{
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    //const double ref_length = 1.0, mach_inf=1.0, angle_of_attack = 0.0, side_slip_angle = 0.0, gamma_gas = 1.4;
    //const double prandtl_number = 0.72, reynolds_number_inf=50000.0;
    const double a = 1.0 , b = 0.0, c = 1.4, d=0.72, e=50000.0;
    PHiLiP::Physics::NavierStokes<dim, nstate, double> navier_stokes_physics = PHiLiP::Physics::NavierStokes<dim, nstate, double>(a,c,a,b,b,d,e);

    const double min = 0;
    const double max = 1.0;
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

    std::array<double, nstate> soln;
    
    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {

            const dealii::Point<dim,double> vertex = cell->vertex(v);

            for (int s=0; s<nstate; s++) {
                soln[s] = navier_stokes_physics.manufactured_solution_function->value(vertex, s);
            }
            for (int d=0; d<dim; d++) {

                dealii::Tensor<1,dim,double> normal;
                normal[d] = 1.0;

                dealii::Tensor<2,nstate,double> jacobian_an = navier_stokes_physics.convective_flux_directional_jacobian_via_dfad (soln, normal);

                dealii::Tensor<2,nstate,double> jacobian_fd;
                for (int col=0; col<nstate; col++) {
                    // const double dw = perturbation*soln[col];
                    // std::array<double, nstate> soln_plus = soln;
                    // std::array<double, nstate> soln_mins = soln;
                    // soln_plus[col] += dw;
                    // soln_mins[col] -= dw;
                    // const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus = euler_physics.convective_flux(soln_plus);
                    // const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins = euler_physics.convective_flux(soln_mins);

                    // for (int row=0; row<nstate; row++) {
                    //     jacobian_fd[row][col] = (conv_flux_plus[row][d] - conv_flux_mins[row][d]) / (2.0*dw);
                    // }

                    const double dw = perturbation*soln[col];
                    std::array<double, nstate> soln_plus2 = soln;
                    std::array<double, nstate> soln_plus = soln;
                    std::array<double, nstate> soln_mins = soln;
                    std::array<double, nstate> soln_mins2 = soln;
                    soln_plus2[col] += 2*dw;
                    soln_plus[col] += dw;
                    soln_mins[col] -= dw;
                    soln_mins2[col] -= 2*dw;

                    const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus2 = navier_stokes_physics.convective_flux(soln_plus2);
                    const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_plus = navier_stokes_physics.convective_flux(soln_plus);
                    const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins = navier_stokes_physics.convective_flux(soln_mins);
                    const std::array<dealii::Tensor<1,dim,double>,nstate> conv_flux_mins2 = navier_stokes_physics.convective_flux(soln_mins2);

                    for (int row=0; row<nstate; row++) {
                        jacobian_fd[row][col] = 
                                -     conv_flux_plus2[row][d] 
                                + 8 * conv_flux_plus[row][d]
                                - 8 * conv_flux_mins[row][d]
                                +     conv_flux_mins2[row][d];
                        jacobian_fd[row][col] /= 12*dw;
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

