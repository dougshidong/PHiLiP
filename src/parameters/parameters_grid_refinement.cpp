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

        prm.declare_entry("isotropic", "true",
                          dealii::Patterns::Bool(),
                          "Inidcates whether the refinement should be done isotropically.");


        prm.declare_entry("error_indicator", "error_based",
                          dealii::Patterns::Selection(
                          " error_based | "
                          " hessian_based | "
                          " residual_based | "
                          " adjoint_based"
                          ),
                          "Enum of error indicators (unused for uniform refinement."
                          "Choices are "
                          " <error_based | "
                          "  hessian_based | "
                          "  residual_based | "
                          "  adjoint_based>.");

        prm.declare_entry("norm_Lq", "2.0",
                          dealii::Patterns::Double(1.0, dealii::Patterns::Double::max_double_value),
                          "Degree of q for use in the Lq norm of some indicators.");

        prm.declare_entry("refinement_fraction", "0.3",
                          dealii::Patterns::Double(0.0, 1.0),
                          "Fraction of elements to undergo refinement for fixed_fraction method.");

        prm.declare_entry("coarsening_fraction", "0.03",
                          dealii::Patterns::Double(0.0, 1.0),
                          "Fraction of elements to undergo coarsening for fixed_fraction method.");
    
        prm.declare_entry("complexity_scale", "2.0",
                          dealii::Patterns::Double(),
                          "Scaling factor multiplying previous complexity.");

        prm.declare_entry("complexity_add", "0.0",
                          dealii::Patterns::Double(),
                          "Constant added to the complexity at each step.");
    // }
    // prm.leave_subsection();
}

void GridRefinementParam::parse_parameters(dealii::ParameterHandler &prm)
{
    // prm.enter_subsection("grid refinement");
    // {
        refinement_steps = prm.get_integer("refinement_steps");

        const std::string refinement_method_string = prm.get("refinement_method");
        if(refinement_method_string == "uniform")            {refinement_method = RefinementMethod::uniform;}
        else if(refinement_method_string == "fixed_fraction"){refinement_method = RefinementMethod::fixed_fraction;}
        else if(refinement_method_string == "continuous")    {refinement_method = RefinementMethod::continuous;}

        const std::string refinement_type_string = prm.get("refinement_type");
        if(refinement_type_string == "h")      {refinement_type = RefinementType::h;}
        else if(refinement_type_string == "p") {refinement_type = RefinementType::p;}
        else if(refinement_type_string == "hp"){refinement_type = RefinementType::hp;}

        isotropic = prm.get_bool("isotropic");

        const std::string error_indicator_string = prm.get("error_indicator");
        if(error_indicator_string == "error_based")        {error_indicator = ErrorIndicator::error_based;}
        else if(error_indicator_string == "hessian_based") {error_indicator = ErrorIndicator::hessian_based;}
        else if(error_indicator_string == "residual_based"){error_indicator = ErrorIndicator::residual_based;}
        else if(error_indicator_string == "adjoint_based") {error_indicator = ErrorIndicator::adjoint_based;}
        
        norm_Lq             = prm.get_double("norm_Lq");
        refinement_fraction = prm.get_double("refinement_fraction");
        coarsening_fraction = prm.get_double("coarsening_fraction");

        complexity_scale = prm.get_double("complexity_scale");
        complexity_add   = prm.get_double("complexity_add");
    // }
    // prm.leave_subsection();
}

} // Parameters namespace

} // PHiLiP namespace
