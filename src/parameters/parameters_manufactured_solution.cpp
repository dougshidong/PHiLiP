#include "parameters_manufactured_solution.h"

#include <deal.II/base/tensor.h>
#include <string>

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
                      " quadratic_solution | "
                      " Alex_solution | "
                      " navah_solution_1 | "
                      " navah_solution_2 | "
                      " navah_solution_3 | "
                      " navah_solution_4 | "
                      " navah_solution_5"
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
                      "  quadratic_solution | "
                      "  navah_solution_1 | "
                      "  navah_solution_2 | "
                      "  navah_solution_3 | "
                      "  navah_solution_4 | "
                      "  navah_solution_5>.");

    // diffusion tensor, get default from function and convert entries to string
    const dealii::Tensor<2,3,double> default_diffusion_tensor = get_default_diffusion_tensor();
    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 3; ++j)
        {
            const std::string str_i = std::to_string(i);
            const std::string str_j = std::to_string(j);
            const std::string name = 
                "diffusion_" + str_i + str_j;
            const std::string description = 
                "[" + str_i + "," + str_j + "] term of diffusion tensor.";
            prm.declare_entry(name, std::to_string(default_diffusion_tensor[i][j]),
                              dealii::Patterns::Double(),
                              description);
        }
    }

    // advection vector, get default from function and convert entries to string
    const dealii::Tensor<1,3,double> default_advection_vector = get_default_advection_vector();
    for(int i = 0; i < 3; ++i)
    {
        const std::string str_i = std::to_string(i);
        const std::string name = 
            "advection_" + str_i;
        const std::string description = 
            "[" + str_i + "] term of advection vector.";
        prm.declare_entry(name, std::to_string(default_advection_vector[i]),
                          dealii::Patterns::Double(),
                          description);
    }

    // diffusion coefficient
    prm.declare_entry("diffusion_coefficient", std::to_string(get_default_diffusion_coefficient()),
                      dealii::Patterns::Double(),
                      "Set the diffusion matrix coefficient.");

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
    else if(manufactured_solution_string == "Alex_solution")           {manufactured_solution_type = Alex_solution;} 
    else if(manufactured_solution_string == "navah_solution_1")         {manufactured_solution_type = navah_solution_1;}
    else if(manufactured_solution_string == "navah_solution_2")         {manufactured_solution_type = navah_solution_2;}
    else if(manufactured_solution_string == "navah_solution_3")         {manufactured_solution_type = navah_solution_3;}
    else if(manufactured_solution_string == "navah_solution_4")         {manufactured_solution_type = navah_solution_4;}
    else if(manufactured_solution_string == "navah_solution_5")         {manufactured_solution_type = navah_solution_5;}

 
    diffusion_tensor[0][0] = prm.get_double("diffusion_00");
    diffusion_tensor[0][1] = prm.get_double("diffusion_01");
    diffusion_tensor[1][0] = prm.get_double("diffusion_10");
    diffusion_tensor[1][1] = prm.get_double("diffusion_11");
    diffusion_tensor[0][2] = prm.get_double("diffusion_02");
    diffusion_tensor[2][0] = prm.get_double("diffusion_20");
    diffusion_tensor[2][1] = prm.get_double("diffusion_21");
    diffusion_tensor[1][2] = prm.get_double("diffusion_12");
    diffusion_tensor[2][2] = prm.get_double("diffusion_22");

    advection_vector[0] = prm.get_double("advection_0");
    advection_vector[1] = prm.get_double("advection_1");
    advection_vector[2] = prm.get_double("advection_2");

    diffusion_coefficient = prm.get_double("diffusion_coefficient");
}

dealii::Tensor<2,3,double> ManufacturedSolutionParam::get_default_diffusion_tensor()
{
    dealii::Tensor<2,3,double> default_diffusion_tensor;

    /* Default before merge */
    // default_diffusion_tensor[0][0] = 12;
    // default_diffusion_tensor[0][1] = -2;
    // default_diffusion_tensor[1][0] = 3;
    // default_diffusion_tensor[1][1] = 20;
    // default_diffusion_tensor[0][2] = -6;
    // default_diffusion_tensor[2][0] = -2;
    // default_diffusion_tensor[2][1] = 0.5;
    // default_diffusion_tensor[1][2] = -4;
    // default_diffusion_tensor[2][2] = 8;

    /* For after merge, current as of 2021-04-25 */
    default_diffusion_tensor[0][0] = 12;
    default_diffusion_tensor[0][1] = 3;
    default_diffusion_tensor[1][0] = 3;
    default_diffusion_tensor[1][1] = 20;
    default_diffusion_tensor[0][2] = -2;
    default_diffusion_tensor[2][0] = 2;
    default_diffusion_tensor[2][1] = 5;
    default_diffusion_tensor[1][2] = -5;
    default_diffusion_tensor[2][2] = 18;

    return default_diffusion_tensor;
}

dealii::Tensor<1,3,double> ManufacturedSolutionParam::get_default_advection_vector()
{
    dealii::Tensor<1,3,double> default_advection_vector;

    const double pi = atan(1)*4.0;
    const double ee = exp(1);

    default_advection_vector[0] = 1.1;
    default_advection_vector[1] = -pi/ee;
    default_advection_vector[2] = ee/pi;

    return default_advection_vector;
}

double ManufacturedSolutionParam::get_default_diffusion_coefficient()
{
    const double pi = atan(1)*4.0;
    const double ee = exp(1);
    return 0.1 * pi/ee;
}

} // Parameters namespace

} // PHiLiP namespace
