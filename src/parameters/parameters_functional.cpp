#include <algorithm>
#include <string>

#include <deal.II/base/utilities.h>

#include "parameters_functional.h"

namespace PHiLiP {

namespace Parameters {

FunctionalParam::FunctionalParam(){}

void FunctionalParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("functional");
    {
        prm.declare_entry("functional_type", "normLp_volume",
                          dealii::Patterns::Selection(
                          " normLp_volume | "
                          " normLp_boundary | "
                          " weighted_integral_volume | "
                          " weighted_integral_boundary | "
                          " error_normLp_volume | "
                          " error_normLp_boundary"
                          ),
                          "Functional that we want to use. "
                          "Choice are "
                          " <normLp_volume | "
                          "  normLp_boundary | "
                          "  weighted_integral_volume | "
                          "  weighted_integral_boundary | "
                          "  error_normLp_volume | "
                          "  error_normLp_boundary>.");

        prm.declare_entry("normLp", "2.0",
                          dealii::Patterns::Double(1.0,dealii::Patterns::Double::max_double_value),
                          "Lp norm strength (may not be used depending on the functional choice).");

        prm.declare_entry("weight_function_type","exp_solution",
                          dealii::Patterns::Selection(
                          " sine_solution | "
                          " cosine_solution | "
                          " additive_solution | "
                          " exp_solution | "
                          " poly_solution | "
                          " even_poly_solution | "
                          " atan_solution"
                          ),
                          "The weight function we want to use (may not be used depending on the functional choice). "
                          "Choices are "
                          " <sine_solution | "
                          "  cosine_solution | "
                          "  additive_solution | "
                          "  exp_solution | "
                          "  poly_solution | "
                          "  even_poly_solution | "
                          "  atan_solution>.");

        prm.declare_entry("use_weight_function_laplacian", "false",
                          dealii::Patterns::Bool(),
                          "Indicates whether to use weight function value or laplacian in the functional.");

        prm.declare_entry("boundary_vector", "[0, 1]",
                          dealii::Patterns::Anything(),
                          "Stores a vector with a list of the (unsigned) integer boundary ID. "
                          "Formatted in square brackets and seperated by commas, eg. \"[0,2]\"");

        prm.declare_entry("use_all_boundaries", "true",
                          dealii::Patterns::Bool(),
                          "Indicates whether all boundaries should be evaluated (for boundary functionals). "
                          "If set to true (default), overrides the boundary_vector list.");
    }
    prm.leave_subsection();
}

void FunctionalParam::parse_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("functional");
    {
        const std::string functional_string = prm.get("functional_type");
        if(functional_string == "normLp_volume")                   {functional_type = FunctionalType::normLp_volume;}
        else if(functional_string == "normLp_boundary")            {functional_type = FunctionalType::normLp_boundary;}
        else if(functional_string == "weighted_integral_volume")   {functional_type = FunctionalType::weighted_integral_volume;}
        else if(functional_string == "weighted_integral_boundary") {functional_type = FunctionalType::weighted_integral_boundary;}
        else if(functional_string == "error_normLp_volume")        {functional_type = FunctionalType::error_normLp_volume;}
        else if(functional_string == "error_normLp_boundary")      {functional_type = FunctionalType::error_normLp_boundary;}

        normLp = prm.get_double("normLp");

        using ManufacturedSolutionEnum = Parameters::ManufacturedSolutionParam::ManufacturedSolutionType;
        const std::string weight_string = prm.get("weight_function_type");
        if(weight_string == "sine_solution")           {weight_function_type = ManufacturedSolutionEnum::sine_solution;} 
        else if(weight_string == "cosine_solution")    {weight_function_type = ManufacturedSolutionEnum::cosine_solution;} 
        else if(weight_string == "additive_solution")  {weight_function_type = ManufacturedSolutionEnum::additive_solution;} 
        else if(weight_string == "exp_solution")       {weight_function_type = ManufacturedSolutionEnum::exp_solution;} 
        else if(weight_string == "poly_solution")      {weight_function_type = ManufacturedSolutionEnum::poly_solution;} 
        else if(weight_string == "even_poly_solution") {weight_function_type = ManufacturedSolutionEnum::even_poly_solution;} 
        else if(weight_string == "atan_solution")      {weight_function_type = ManufacturedSolutionEnum::atan_solution;}

        use_weight_function_laplacian = prm.get_bool("use_weight_function_laplacian");

        std::string boundary_string = prm.get("boundary_vector");

        // removing formatting
        std::string removeChars = "[]";
        for(unsigned int i = 0; i < removeChars.length(); ++i)
            boundary_string.erase(std::remove(boundary_string.begin(), boundary_string.end(), removeChars.at(i)), boundary_string.end());
        std::vector<std::string> boundary_string_vector = dealii::Utilities::split_string_list(boundary_string);

        // pushing into a vector
        for(auto entry : boundary_string_vector)
            boundary_vector.push_back(dealii::Utilities::string_to_int(entry));

        use_all_boundaries = prm.get_bool("use_all_boundaries");
    }
    prm.leave_subsection();
}

} // namespace Parameters

} // namespace PHiLiP