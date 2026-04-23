#include <assert.h>
#include <deal.II/grid/grid_generator.h>

#include "assert_compare_array.h"
#include "parameters/parameters.h"
#include "physics/real_gas.h"

const double TOLERANCE = 1E-12;


int main (int argc, char * argv[])
{
    MPI_Init(&argc, &argv);
    const int dim = PHILIP_DIM;
    const int nspecies = PHILIP_SPECIES;
    const int nstate = dim+nspecies+1;

    //default parameters
    dealii::ParameterHandler parameter_handler;
    PHiLiP::Parameters::AllParameters::declare_parameters (parameter_handler); // default fills options
    PHiLiP::Parameters::AllParameters all_parameters;
    all_parameters.parse_parameters (parameter_handler);

    all_parameters.euler_param.mach_inf = 1.0; all_parameters.euler_param.gamma_gas = 1.4;
    if (nspecies == 2)
        all_parameters.chemistry_input_file = "../../chemistry_files/O2_N2.kinetics";
    else if (nspecies == 3)
        all_parameters.chemistry_input_file = "../../chemistry_files/H2_O2_N2.kinetics";

    using ManufacturedSolutionEnum = PHiLiP::Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
    all_parameters.manufactured_convergence_study_param.manufactured_solution_param.manufactured_solution_type = ManufacturedSolutionEnum::atan_solution;
    PHiLiP::Physics::RealGas<dim, nspecies, nstate, double> real_gas_physics = PHiLiP::Physics::RealGas<dim, nspecies, nstate, double>(&all_parameters);

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

    std::array<double, dim+nspecies+1> conservative_soln;
    std::array<double, dim+nspecies+1> conservative_soln2;
    std::array<double, dim+nspecies+1> primitive_soln;
    for (auto cell : grid.active_cell_iterators()) {
        for (unsigned int v=0; v < dealii::GeometryInfo<dim>::vertices_per_cell; ++v) {
            const dealii::Point<dim,double> vertex = cell->vertex(v);
            for (int s=0; s<nstate; s++) {
                conservative_soln[s] = real_gas_physics.manufactured_solution_function->value(vertex, s);
            }
            primitive_soln = real_gas_physics.convert_conservative_to_primitive(conservative_soln);
            conservative_soln2 = real_gas_physics.convert_primitive_to_conservative(primitive_soln);

            // Flipping back and forth between conservative and primitive solution result
            // in the same solution
            assert_compare_array<nstate> ( conservative_soln, conservative_soln2, 1.0, TOLERANCE);
            // Manufactured solution gives positive density
            if(conservative_soln[0] < TOLERANCE) std::abort();
            // Manufactured solution gives positive energy
            if(conservative_soln[dim+1] < TOLERANCE) std::abort();
            // Manufactured solution gives positive pressure
            if(primitive_soln[dim+1] < TOLERANCE) std::abort();
            // Manufactured solution gives positive species densities
            for(int ispecies = 0; ispecies < nspecies - 1; ++ispecies)
                if(conservative_soln[dim+2+ispecies] < TOLERANCE) std::abort();

            if(real_gas_physics.compute_mixture_pressure(conservative_soln) < TOLERANCE) std::abort();
            if(real_gas_physics.compute_sound(conservative_soln) < TOLERANCE) std::abort();

        }
    }
    MPI_Finalize();
    return 0;
}

