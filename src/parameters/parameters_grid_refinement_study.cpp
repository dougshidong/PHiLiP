#include <array>

#include "parameters_grid_refinement_study.h"

#include "parameters/parameters_manufactured_convergence_study.h"

namespace PHiLiP {

namespace Parameters {

GridRefinementStudyParam::GridRefinementStudyParam() : 
    functional_param(FunctionalParam()),
    manufactured_solution_param(ManufacturedSolutionParam()) {}

void GridRefinementStudyParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("grid refinement study");
    {
        Parameters::FunctionalParam::declare_parameters(prm);
        Parameters::ManufacturedSolutionParam::declare_parameters(prm);

        prm.enter_subsection("grid refinement");
        {
            Parameters::GridRefinementParam::declare_parameters(prm);
        }
        prm.leave_subsection();

        // looping over the elements of the array to declare the different refinement
        for(unsigned int i = 0; i < MAX_REFINEMENTS; ++i)
        {
            prm.enter_subsection("grid refinement [" + dealii::Utilities::int_to_string(i,1) + "]");
            {
                Parameters::GridRefinementParam::declare_parameters(prm);
            }
            prm.leave_subsection();
        }
    
        prm.declare_entry("poly_degree", "1",
                          dealii::Patterns::Integer(),
                          "Polynomial order of starting mesh.");

        prm.declare_entry("poly_degree_max", "5",
                          dealii::Patterns::Integer(),
                          "Maximum polynomial order.");

        prm.declare_entry("poly_degree_grid", "2",
                          dealii::Patterns::Integer(),
                          "Polynomial degree of the grid.");

        prm.declare_entry("num_refinements", "0",
                          dealii::Patterns::Integer(0, MAX_REFINEMENTS),
                          "Number of different refinements to be performed.");

        prm.declare_entry("grid_type", "hypercube",
                          dealii::Patterns::Selection("hypercube|sinehypercube|read_grid"),
                          "Enum of generated grid. "
                          "If read_grid, must have grids xxxx.msh."
                          "Choices are <hypercube|read_grid>.");

        prm.declare_entry("input_grid", "xxxx",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Name of Gmsh grid xxxx.msh used in the grid refinement study if read_grid is chosen as the grid_type.");

        prm.declare_entry("grid_left", "0.0",
                          dealii::Patterns::Double(),
                          "for grid_type hypercube, left bound of domain.");

        prm.declare_entry("grid_right", "1.0",
                          dealii::Patterns::Double(),
                          "for grid_type hypercube, right bound of domain.");

        prm.declare_entry("grid_size", "4",
                          dealii::Patterns::Integer(),
                          "Initial grid size (number of elements per side).");

        prm.declare_entry("use_interpolation", "false",
                          dealii::Patterns::Bool(),
                          "Indicates whether to interpolate the problem instead of solving with DG.");

        prm.declare_entry("approximate_functional", "false",
                          dealii::Patterns::Bool(),
                          "Indicates whether function is to be approximated from manufactured solution"
                          "or exact value read from functional_value parameter.");

        prm.declare_entry("functional_value", "0.0",
                          dealii::Patterns::Double(),
                          "Exact value of functional for goal-oriented convergence.");

        prm.declare_entry("output_vtk", "true",
                          dealii::Patterns::Bool(),
                          "Output flag for grid_refinement vtk files.");

        prm.declare_entry("output_adjoint_vtk", "false",
                          dealii::Patterns::Bool(),
                          "output flag for adjoint vtk files.");

        prm.declare_entry("output_solution_error", "true",
                          dealii::Patterns::Bool(),
                          "ouput the convergence table for the solution error.");

        prm.declare_entry("output_functional_error", "false",
                          dealii::Patterns::Bool(),
                          "ouput the convergence table for the functional error.");

        prm.declare_entry("output_gnuplot_solution", "true",
                          dealii::Patterns::Bool(),
                          "Output flag for gnuplot solution error figure.");

        prm.declare_entry("output_gnuplot_functional", "false",
                          dealii::Patterns::Bool(),
                          "Output flag for gnuplot functional error figure.");

        prm.declare_entry("refresh_gnuplot", "true",
                          dealii::Patterns::Bool(),
                          "Indicates whetherto output a new gnuplot figure at every iteration."
                          "Requires output_gnuplot == true.");

        prm.declare_entry("output_solution_time", "false",
                          dealii::Patterns::Bool(),
                          "Output flag for wall clock solution timing.");

        prm.declare_entry("output_adjoint_time", "false",
                          dealii::Patterns::Bool(),
                          "Output flag for wall clock adjoint timing.");
    }
    prm.leave_subsection();
}

void GridRefinementStudyParam::parse_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("grid refinement study");
    {
        functional_param.parse_parameters(prm);
        manufactured_solution_param.parse_parameters(prm);

        // default section gets written into vector pos 1 by default
        prm.enter_subsection("grid refinement");
        {
            grid_refinement_param_vector[0].parse_parameters(prm);
        }
        prm.leave_subsection();

        num_refinements = prm.get_integer("num_refinements");

        // overwrite if num_refinements > 0
        for(unsigned int i = 0; i < num_refinements; ++i)
        {
            prm.enter_subsection("grid refinement [" + dealii::Utilities::int_to_string(i,1) + "]");
            {
                grid_refinement_param_vector[i].parse_parameters(prm);
            }
            prm.leave_subsection();
        }

        poly_degree      = prm.get_integer("poly_degree");
        poly_degree_max  = prm.get_integer("poly_degree_max");
        poly_degree_grid = prm.get_integer("poly_degree_grid");

        const std::string grid_string = prm.get("grid_type");
        using GridEnum = Parameters::ManufacturedConvergenceStudyParam::GridEnum;
        if(grid_string == "hypercube")      {grid_type = GridEnum::hypercube;}
        else if(grid_string == "read_grid") {grid_type = GridEnum::read_grid;
                                             input_grid = prm.get("input_grid");}

        grid_left  = prm.get_double("grid_left");
        grid_right = prm.get_double("grid_right");

        grid_size  = prm.get_integer("grid_size");

        use_interpolation = prm.get_bool("use_interpolation");

        approximate_functional = prm.get_bool("approximate_functional");
        functional_value       = prm.get_double("functional_value");

        output_vtk         = prm.get_bool("output_vtk");
        output_adjoint_vtk = prm.get_bool("output_adjoint_vtk");

        output_solution_error   = prm.get_bool("output_solution_error");
        output_functional_error = prm.get_bool("output_functional_error");

        output_gnuplot_solution   = prm.get_bool("output_gnuplot_solution");
        output_gnuplot_functional = prm.get_bool("output_gnuplot_functional");
        refresh_gnuplot           = prm.get_bool("refresh_gnuplot");

        output_solution_time = prm.get_bool("output_solution_time");
        output_adjoint_time  = prm.get_bool("output_adjoint_time");
    }
    prm.leave_subsection();
}

} // namespace Parameters

} // namespace PHiLiP