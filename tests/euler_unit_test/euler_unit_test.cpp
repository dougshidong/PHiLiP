#include <assert.h>
#include <deal.II/grid/grid_generator.h>

#include "parameters/parameters.h"
#include "physics/physics.h"

const double TOLERANCE = 1E-12;

template<int dim, int nstate>
void assert_compare_array ( const std::array<double, nstate> &array1, const std::array<double, nstate> &array2, double scale2)
{
    for (int s=0; s<nstate; s++) {
        const double diff = std::abs(array1[s] - scale2*array2[s]);
        std::cout
            << "State " << s << " out of " << nstate
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
    const int dim = PHILIP_DIM;
    const int nstate = dim+2;

    PHiLiP::Physics::Euler<dim, nstate, double> euler_physics = PHiLiP::Physics::Euler<dim, nstate, double>();

    const double min = -10.0;
    const double max = 10.0;
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
            conservative_soln = euler_physics.manufactured_solution(vertex);
            primitive_soln = euler_physics.convert_conservative_to_primitive(conservative_soln);
            conservative_soln2 = euler_physics.convert_primitive_to_conservative(primitive_soln);

            // Flipping back and forth between conservative and primitive solution result
            // in the same solution
            assert_compare_array<dim,nstate> ( conservative_soln, conservative_soln2, 1.0);
            // Manufactured solution gives positive density
            assert(conservative_soln[0] > TOLERANCE);
            // Manufactured solution gives positive energy
            assert(conservative_soln[1+dim] > TOLERANCE);
            // Manufactured solution gives positive pressure
            assert(primitive_soln[1+dim] > TOLERANCE);
        }
    }
    return 0;
}

