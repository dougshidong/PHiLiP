#include "parameters_manufactured_solution.h"

namespace PHiLiP {

namespace Parameters {

// Manufactured Solution inputs
ManufacturedSolutionParam::ManufacturedSolutionParam() {}

void ManufacturedSolutionParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.declare_entry("use_manufactured_source_term", "false",
                      dealii::Patterns::Bool(),
                      "Uses non-zero source term based on the manufactured solution and the PDE.");

    prm.declare_entry("manufactured_solution_type","exp_solution",
                      dealii::Patterns::Selection(
                      " sine_solution | "
                      " cosine_solution | "
                      " additive_solution | "
                      " exp_solution | "
                      " poly_solution | "
                      " even_poly_solution | "
                      " atan_solution | "
                      " boundary_layer_solution | "
                      " s_shock_solution | "
                      " quadratic_solution"
                      ),
                      "The manufactured solution we want to use (if use_manufactured_source_term==true). "
                      "Choices are "
                      " <sine_solution | "
                      "  cosine_solution | "
                      "  additive_solution | "
                      "  exp_solution | "
                      "  poly_solution | "
                      "  even_poly_solution | "
                      "  atan_solution | "
                      "  boundary_layer_solution | "
                      "  s_shock_solution | "
                      "  quadratic_solution>.");
}

void ManufacturedSolutionParam::parse_parameters(dealii::ParameterHandler &prm)
{
    use_manufactured_source_term = prm.get_bool("use_manufactured_source_term");
    
    const std::string manufactured_solution_string = prm.get("manufactured_solution_type");
    if(manufactured_solution_string == "sine_solution")               {manufactured_solution_type = sine_solution;} 
    else if(manufactured_solution_string == "cosine_solution")        {manufactured_solution_type = cosine_solution;} 
    else if(manufactured_solution_string == "additive_solution")      {manufactured_solution_type = additive_solution;} 
    else if(manufactured_solution_string == "exp_solution")           {manufactured_solution_type = exp_solution;} 
    else if(manufactured_solution_string == "poly_solution")          {manufactured_solution_type = poly_solution;} 
    else if(manufactured_solution_string == "even_poly_solution")     {manufactured_solution_type = even_poly_solution;} 
    else if(manufactured_solution_string == "atan_solution")          {manufactured_solution_type = atan_solution;}
    else if(manufactured_solution_string == "boundary_layer_solution"){manufactured_solution_type = boundary_layer_solution;}
    else if(manufactured_solution_string == "s_shock_solution")       {manufactured_solution_type = s_shock_solution;}
    else if(manufactured_solution_string == "quadratic_solution")     {manufactured_solution_type = quadratic_solution;}
}

} // Parameters namespace

} // PHiLiP namespace
