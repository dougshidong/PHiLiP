#include <assert.h>
#include <deal.II/grid/grid_generator.h>

#include "assert_compare_array.h"
#include "parameters/parameters.h"
#include "physics/euler.h"

const double TOLERANCE = 1E-12;


int main (int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    //const double ref_length = 1.0, mach_inf=1.0, angle_of_attack = 0.0, side_slip_angle = 0.0, gamma_gas = 1.4;
    const double a = 1.0 , b = 0.0, c = 1.4;
    //default parameters
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler); // default fills options
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);
    PHiLiP::Physics::Euler<dim, nstate, double> euler_physics = PHiLiP::Physics::Euler<dim, nstate, double>(&all_parameters,a,c,a,b,b);

    const double min = 0.0;
    const double max = 1.0;
    const int nx = 11;

    std::vector<unsigned int> repetitions(dim, nx);
    dealii::Point<dim,double> corner1, corner2;
    for (int d=0; d<dim; d++) { 
        corner1[d] = min;
        corner2[d] = max;
    }
    dealii::Triangulation<dim> grid;
    dealii::GridGenerator::subdivided_hyper_rectangle(grid, repetitions, corner1, corner2);

    std::array<double, dim+2> conservative_soln;
    std::array<double, dim+2> conservative_soln2;
    std::array<double, dim+2> primitive_soln;
    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
            const dealii::Point<dim,double> vertex = cell->vertex(v);
            for (int s=0; s<nstate; s++) {
                conservative_soln[s] = euler_physics.manufactured_solution_function->value(vertex, s);
            }
            primitive_soln = euler_physics.convert_conservative_to_primitive(conservative_soln);
            conservative_soln2 = euler_physics.convert_primitive_to_conservative(primitive_soln);

            // Flipping back and forth between conservative and primitive solution result
            // in the same solution
            assert_compare_array<nstate> ( conservative_soln, conservative_soln2, 1.0, TOLERANCE);
            // Manufactured solution gives positive density
            if(conservative_soln[0] < TOLERANCE) std::abort();
            // Manufactured solution gives positive energy
            if(conservative_soln[nstate-1] < TOLERANCE) std::abort();
            // Manufactured solution gives positive pressure
            if(primitive_soln[1+dim] < TOLERANCE) std::abort();

            if(euler_physics.compute_pressure(conservative_soln) < TOLERANCE) std::abort();
            if(euler_physics.compute_sound(conservative_soln) < TOLERANCE) std::abort();

        }
    }
    MPI_Finalize();
    return 0;
}

