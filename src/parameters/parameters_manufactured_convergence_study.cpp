#include "parameters_manufactured_convergence_study.h"

namespace PHiLiP {
namespace Parameters {

// Manufactured Solution inputs
ManufacturedConvergenceStudyParam::ManufacturedConvergenceStudyParam () {}

void ManufacturedConvergenceStudyParam::declare_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("manufactured solution convergence study");
    {
        prm.declare_entry("use_manufactured_source_term", "false",
                          dealii::Patterns::Bool(),
                          "Uses non-zero source term based on the manufactured solution and the PDE.");

        prm.declare_entry("manufactured_solution_type","sine_solution",
                          dealii::Patterns::Selection(
                          " sine_solution | "
                          " cosine_solution | "
                          " additive_solution | "
                          " exp_solution | "
                          " poly_solution"
                          " even_poly_solution"
                          ),
                          "The manufactured solution we want to use (if use_manufactured_source_term==true). "
                          "Choices are "
                          " <sine_solution | "
                          "  cosine_solution | "
                          "  additive_solution | "
                          "  exp_solution | "
                          "  poly_solution"
                          "  even_poly_solution>.");

        prm.declare_entry("grid_type", "hypercube",
                          dealii::Patterns::Selection("hypercube|sinehypercube|read_grid"),
                          "Enum of generated grid. "
                          "If read_grid, must have grids xxxx#.msh, where # is the grid numbering from 0 to number_of_grids-1."
                          "Choices are <hypercube|sinehypercube|read_grid>.");

        prm.declare_entry("input_grids", "xxxx",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Prefix of Gmsh grids xxxx#.msh used in the grid convergence if read_grid is chosen as the grid_type. ");

        prm.declare_entry("output_meshes", "false",
                          dealii::Patterns::Bool(),
                          "Writes out meshes used for the simulation."
                          "Output will be Gmsh grids named grid-#.msh");

        prm.declare_entry("random_distortion", "0.0",
                          dealii::Patterns::Double(0.0, 0.5),
                          "Randomly disturb grid."
                          "Displaces node by percentage of longest associated edge.");

        prm.declare_entry("initial_grid_size", "2",
                          dealii::Patterns::Integer(),
                          "Initial grid of size (initial_grid_size)^dim");
        prm.declare_entry("number_of_grids", "4",
                          dealii::Patterns::Integer(),
                          "Number of grids in grid study");
        prm.declare_entry("grid_progression", "1.5",
                          dealii::Patterns::Double(),
                          "Multiplier on grid size. "
                          "ith-grid will be of size (initial_grid*(i*grid_progression)+(i*grid_progression_add))^dim");
        prm.declare_entry("grid_progression_add", "0",
                          dealii::Patterns::Integer(),
                          "Adds number of cell to 1D grid. "
                          "ith-grid will be of size (initial_grid*(i*grid_progression)+(i*grid_progression_add))^dim");

        prm.declare_entry("slope_deficit_tolerance", "0.1",
                          dealii::Patterns::Double(),
                          "Tolerance within which the convergence orders are considered to be optimal. ");

        prm.declare_entry("degree_start", "0",
                          dealii::Patterns::Integer(),
                          "Starting degree for convergence study");
        prm.declare_entry("degree_end", "3",
                          dealii::Patterns::Integer(),
                          "Last degree used for convergence study");
    }
    prm.leave_subsection();
}

void ManufacturedConvergenceStudyParam ::parse_parameters (dealii::ParameterHandler &prm)
{
    prm.enter_subsection("manufactured solution convergence study");
    {
        use_manufactured_source_term = prm.get_bool("use_manufactured_source_term");
        
        const std::string manufactured_solution_string = prm.get("manufactured_solution_type");
        if(manufactured_solution_string == "sine_solution")           {manufactured_solution_type = sine_solution;} 
        else if(manufactured_solution_string == "cosine_solution")    {manufactured_solution_type = cosine_solution;} 
        else if(manufactured_solution_string == "additive_solution")  {manufactured_solution_type = additive_solution;} 
        else if(manufactured_solution_string == "exp_solution")       {manufactured_solution_type = exp_solution;} 
        else if(manufactured_solution_string == "poly_solution")      {manufactured_solution_type = poly_solution;} 
        else if(manufactured_solution_string == "even_poly_solution") {manufactured_solution_type = even_poly_solution;} 

        const std::string grid_string = prm.get("grid_type");
        if (grid_string == "hypercube") grid_type = GridEnum::hypercube;
        if (grid_string == "sinehypercube") grid_type = GridEnum::sinehypercube;
        if (grid_string == "read_grid") {
            grid_type = GridEnum::read_grid;
            input_grids = prm.get("input_grids");
        }

        random_distortion           = prm.get_double("random_distortion");
        output_meshes               = prm.get_bool("output_meshes");

        degree_start                = prm.get_integer("degree_start");
        degree_end                  = prm.get_integer("degree_end");

        initial_grid_size           = prm.get_integer("initial_grid_size");
        number_of_grids             = prm.get_integer("number_of_grids");
        grid_progression            = prm.get_double("grid_progression");
        grid_progression_add        = prm.get_integer("grid_progression_add");

        slope_deficit_tolerance     = prm.get_double("slope_deficit_tolerance");
    }
    prm.leave_subsection();
}

} // Parameters namespace
} // PHiLiP namespace
