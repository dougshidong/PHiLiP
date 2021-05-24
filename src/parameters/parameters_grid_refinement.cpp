#include <algorithm>
#include <string>

#include <deal.II/base/utilities.h>

#include "parameters_grid_refinement.h"

namespace PHiLiP {

namespace Parameters {

// Manufactured Solution inputs
GridRefinementParam::GridRefinementParam(){}

void GridRefinementParam::declare_parameters(dealii::ParameterHandler &prm)
{
    // prm.enter_subsection("grid refinement");
    // {
        prm.declare_entry("refinement_steps", "3",
                          dealii::Patterns::Integer(),
                          "Number of iterations to be performed");

        prm.declare_entry("refinement_method", "uniform",
                           dealii::Patterns::Selection(
                          " uniform | "
                          " fixed_fraction | " 
                          " continuous"),
                          "Enum of refinement methods."
                          "Choices are "
                          " <uniform | "
                          "  fixed_fraction | "
                          "  continuous>.");

        prm.declare_entry("refinement_type", "h",
                          dealii::Patterns::Selection(
                          " h | "
                          " p | "
                          " hp"),
                          "Enum of refinement types."
                          "Choices are "
                          " <h | "
                          "  p | "
                          "  hp>.");

        prm.declare_entry("anisotropic", "false",
                          dealii::Patterns::Bool(),
                          "Inidcates whether the refinement should be done anisotropically.");

        prm.declare_entry("anisotropic_ratio_max", "1.0e+12",
                          dealii::Patterns::Double(1.0, dealii::Patterns::Double::max_double_value),
                          "maximum anisotropic ratio for continuous size field targets.");

        prm.declare_entry("anisotropic_ratio_min", "1.0e-12",
                          dealii::Patterns::Double(dealii::Patterns::Double::min_double_value, 1.0),
                          "miniumum anistropic ratio for continuous size field targets.");

        prm.declare_entry("anisotropic_threshold_ratio", "3.0",
                          dealii::Patterns::Double(1.0, dealii::Patterns::Double::max_double_value),
                          "Threshold for flagging cells with anisotropic refinement.");

        prm.declare_entry("anisotropic_indicator", "jump_based",
                          dealii::Patterns::Selection(
                          " jump_based | "
                          " reconstruction_based"),
                          "Enum of anisotropic indicators (unused for isotropic refinement)."
                          "Choices are "
                          " <jump_based | "
                          "  reconstruction_based>.");

        prm.declare_entry("error_indicator", "error_based",
                          dealii::Patterns::Selection(
                          " error_based | "
                          " hessian_based | "
                          " residual_based | "
                          " adjoint_based"),
                          "Enum of error indicators (unused for uniform refinement)."
                          "Choices are "
                          " <error_based | "
                          "  hessian_based | "
                          "  residual_based | "
                          "  adjoint_based>.");

        prm.declare_entry("output_type", "msh_out",
                          dealii::Patterns::Selection(
                          " gmsh_out | "
                          " msh_out"),
                          "Enum of output data types (for interface with mesh generators)."
                          "Choices are "
                          " <gmsh_out | "
                          "  msh_out>.");

        prm.declare_entry("output_data_type", "size_field",
                          dealii::Patterns::Selection(
                          " size_field | "
                          " frame_field | "
                          " metric_field"),
                          "Enum of output data types of refinement indicator (used for msh_out only currently)."
                          "Choices are "
                          " <size_field | "
                          "  frame_field | "
                          "  metric_field>.");

        prm.declare_entry("norm_Lq", "2.0",
                          dealii::Patterns::Double(1.0, dealii::Patterns::Double::max_double_value),
                          "Degree of q for use in the Lq norm of some indicators.");

        prm.declare_entry("refinement_fraction", "0.3",
                          dealii::Patterns::Double(0.0, 1.0),
                          "Fraction of elements to undergo refinement for fixed_fraction method.");

        prm.declare_entry("coarsening_fraction", "0.03",
                          dealii::Patterns::Double(0.0, 1.0),
                          "Fraction of elements to undergo coarsening for fixed_fraction method.");
    
        prm.declare_entry("r_max", "20",
                          dealii::Patterns::Double(1.0, dealii::Patterns::Double::max_double_value),
                          "Maximum refinement factor for adjoint-based size-field (from log DWR).");

        prm.declare_entry("c_max", "4",
                          dealii::Patterns::Double(1.0, dealii::Patterns::Double::max_double_value),
                          "Maximum coarsening factor for adjoint-based size-field (from log DWR).");

        prm.declare_entry("complexity_scale", "2.0",
                          dealii::Patterns::Double(),
                          "Scaling factor multiplying previous complexity.");

        prm.declare_entry("complexity_add", "0.0",
                          dealii::Patterns::Double(),
                          "Constant added to the complexity at each step.");

        prm.declare_entry("complexity_vector", "[]",
                          dealii::Patterns::Anything(),
                          "Stores an initial vector of values for complexity targets. "
                          "Will iterate over and then switch to scaling and adding. "
                          "Formatted in square brackets and seperated by commas, eg. \"[1000,2000]\"");

        prm.declare_entry("exit_after_refine", "false",
                          dealii::Patterns::Bool(),
                          "Option to exit after call to the grid refinement (for debugging mesh write).");

    // }
    // prm.leave_subsection();
}

void GridRefinementParam::parse_parameters(dealii::ParameterHandler &prm)
{
    // prm.enter_subsection("grid refinement");
    {
        refinement_steps = prm.get_integer("refinement_steps");

        const std::string refinement_method_string = prm.get("refinement_method");
        if(refinement_method_string == "uniform")            {refinement_method = RefinementMethod::uniform;}
        else if(refinement_method_string == "fixed_fraction"){refinement_method = RefinementMethod::fixed_fraction;}
        else if(refinement_method_string == "continuous")    {refinement_method = RefinementMethod::continuous;}

        const std::string refinement_type_string = prm.get("refinement_type");
        if(refinement_type_string == "h")      {refinement_type = RefinementType::h;}
        else if(refinement_type_string == "p") {refinement_type = RefinementType::p;}
        else if(refinement_type_string == "hp"){refinement_type = RefinementType::hp;}

        anisotropic = prm.get_bool("anisotropic");

        anisotropic_ratio_max = prm.get_double("anisotropic_ratio_max");
        anisotropic_ratio_min = prm.get_double("anisotropic_ratio_min");

        anisotropic_threshold_ratio = prm.get_double("anisotropic_threshold_ratio");

        const std::string anisotropic_indicator_string = prm.get("anisotropic_indicator");
        if(anisotropic_indicator_string == "jump_based")               {anisotropic_indicator = AnisoIndicator::jump_based;}
        else if(anisotropic_indicator_string == "reconstruction_based"){anisotropic_indicator = AnisoIndicator::reconstruction_based;}

        const std::string error_indicator_string = prm.get("error_indicator");
        if(error_indicator_string == "error_based")        {error_indicator = ErrorIndicator::error_based;}
        else if(error_indicator_string == "hessian_based") {error_indicator = ErrorIndicator::hessian_based;}
        else if(error_indicator_string == "residual_based"){error_indicator = ErrorIndicator::residual_based;}
        else if(error_indicator_string == "adjoint_based") {error_indicator = ErrorIndicator::adjoint_based;}
        
        const std::string output_type_string = prm.get("output_type");
        if(output_type_string == "gmsh_out")     {output_type = OutputType::gmsh_out;}
        else if(output_type_string == "msh_out") {output_type = OutputType::msh_out;}

        const std::string output_data_type_string = prm.get("output_data_type");
        if(output_data_type_string == "size_field")        {output_data_type = OutputDataType::size_field;}
        else if(output_data_type_string == "frame_field")  {output_data_type = OutputDataType::frame_field;}
        else if(output_data_type_string == "metric_field") {output_data_type = OutputDataType::metric_field;}

        norm_Lq             = prm.get_double("norm_Lq");
        refinement_fraction = prm.get_double("refinement_fraction");
        coarsening_fraction = prm.get_double("coarsening_fraction");

        r_max = prm.get_double("r_max");
        c_max = prm.get_double("c_max");

        complexity_scale = prm.get_double("complexity_scale");
        complexity_add   = prm.get_double("complexity_add");
    
        std::string complexity_string = prm.get("complexity_vector");
        
        // remove formatting
        std::string removeChars = "[]";
        for(unsigned int i = 0; i < removeChars.length(); ++i)
            complexity_string.erase(std::remove(complexity_string.begin(), complexity_string.end(), removeChars.at(i)), complexity_string.end());
        std::vector<std::string> complexity_string_vector = dealii::Utilities::split_string_list(complexity_string);
    
        // pushing into a vector
        for(auto entry : complexity_string_vector)
            complexity_vector.push_back(dealii::Utilities::string_to_int(entry));
        
        exit_after_refine = prm.get_bool("exit_after_refine");
    }
    // prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
