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
    std::cout << "============= HERE =============\n";
    std::cout << std::setprecision(std::numeric_limits<long double>::digits10 + 1) << std::scientific;
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    //const double ref_length = 1.0, mach_inf=1.0, angle_of_attack = 0.0, side_slip_angle = 0.0, gamma_gas = 1.4;
    //const double prandtl_number = 0.72, reynolds_number_inf=50000.0;
    const double a = 1.0 , b = 0.0, c = 1.4, d=0.72, e=1.0;//e=10000000.0;
    PHiLiP::Physics::NavierStokes<dim, nstate, double> navier_stokes_physics = PHiLiP::Physics::NavierStokes<dim, nstate, double>(a,c,a,b,b,d,e);

    const double min = 0;
    const double max = 1.0;
    const int nx = 6;

    // const double perturbation = 1e-5;

    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);

    std::array<double, nstate> soln;
    std::array<double,nstate> primitive_soln;

    double temperature, scaled_viscosity_coefficient;
    const double temperature_ratio = 110.4/273.15;
    double dmudT_an, dmudT_ad;
    
    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {

            const dealii::Point<dim,double> vertex = cell->vertex(v);

            for (int s=0; s<nstate; s++) {
                soln[s] = navier_stokes_physics.manufactured_solution_function->value(vertex, s);
            }
            // Compute the analytical derivative of viscosity wrt to temperature
            primitive_soln = navier_stokes_physics.convert_conservative_to_primitive(soln); // from Euler
            temperature = navier_stokes_physics.compute_temperature(primitive_soln); // from Euler
            scaled_viscosity_coefficient = navier_stokes_physics.compute_scaled_viscosity_coefficient(primitive_soln);
            // -- Matasuka (2018), Eq.(4.14.17)
            dmudT_an = 0.5*(scaled_viscosity_coefficient/(temperature + temperature_ratio))*(1.0 + 3.0*temperature_ratio/temperature);
            
            // Compute derivative with automatic differentiation  
            dmudT_ad = navier_stokes_physics.compute_scaled_viscosity_coefficient_derivative_wrt_temperature_via_dfad(soln);

            const double diff = std::abs(dmudT_ad - dmudT_an);
            double max = std::max(std::abs(dmudT_ad), std::abs(dmudT_an));
            const double rel_diff = diff/max;
            std::cout
            << "Analytical = " << dmudT_an
            << std::endl
            << "Automatic Diff. = " << dmudT_ad
            << std::endl
            << "Relative difference = " << rel_diff
            << std::endl;
            std::cout << std::endl;
            if(rel_diff > TOLERANCE) {
                std::cout << "Difference too high. rel_diff=" << rel_diff << " and tolerance=" << TOLERANCE << std::endl;
                std::cout << "Failing test..." << std::endl;
                std::abort();
            }
        }
    }
    return 0;
}

