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
                      " navah_solution "
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
                      "  navah_solution>.");

    prm.declare_entry("navah_manufactured_solution_number","1",
                      dealii::Patterns::Selection(
                      " 1 | "
                      " 2 | "
                      " 3 | "
                      " 4 | "
                      " 5 | "
                      ),
                      "Which Navah manufactured solution we want to use (if manufactured_solution_type==navah_solution). "
                      "Choices are "
                      " <1 | "
                      "  2 | "
                      "  3 | "
                      "  4 | "
                      "  5>.");

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
    else if(manufactured_solution_string == "navah_solution")         {manufactured_solution_type = navah_solution;}

    if(manufactured_solution_string == "navah_solution") {
        const int navah_manufactured_solution_number = std::stoi(prm.get("navah_manufactured_solution_number"));
        NavahCoefficientMatrix = get_navah_coefficient_matrix(navah_manufactured_solution_number);
    }
    
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

std::array<dealii::Tensor<1,7,double>,5> ManufacturedSolutionParam::get_navah_coefficient_matrix(const int navah_manufactured_solution_number)
{
    // Matrix with all coefficients of the various manufactured solutions given in Navah's paper.
    // Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018

    std::array<dealii::Tensor<1,7,double>,5> ncm; // navah coefficient matrix (ncm)
    if(navah_manufactured_solution_number == 1) {
        /* MS-1 */
        ncm[0][0]= 1.0; ncm[0][1]=0.3; ncm[0][2]=-0.2; ncm[0][3]=0.3; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 1.0; ncm[1][1]=0.3; ncm[1][2]= 0.3; ncm[1][3]=0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 1.0; ncm[2][1]=0.3; ncm[2][2]= 0.3; ncm[2][3]=0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=18.0; ncm[3][1]=5.0; ncm[3][2]= 5.0; ncm[3][3]=0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        for(int j=0; j<7; j++) {
            ncm[4][j] = 0.0;
        }
    } else if(navah_manufactured_solution_number == 2) {
        /* MS-2 */
        ncm[0][0]=2.7; ncm[0][1]=0.9; ncm[0][2]=-0.9; ncm[0][3]=1.0; ncm[0][4]=1.5; ncm[0][5]=1.5; ncm[0][6]=1.5;
        ncm[1][0]=2.0; ncm[1][1]=0.7; ncm[1][2]= 0.7; ncm[1][3]=0.4; ncm[1][4]=1.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]=2.0; ncm[2][1]=0.4; ncm[2][2]= 0.4; ncm[2][3]=0.4; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=2.0; ncm[3][1]=1.0; ncm[3][2]= 1.0; ncm[3][3]=0.5; ncm[3][4]=1.0; ncm[3][5]=1.0; ncm[3][6]=1.5;
        for(int j=0; j<7; j++) {
            ncm[4][j] = 0.0;
        } 
    } else if(navah_manufactured_solution_number == 3) {
        /* MS-3 */
        ncm[0][0]= 1.0; ncm[0][1]=0.1; ncm[0][2]=-0.2; ncm[0][3]=0.1; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 2.0; ncm[1][1]=0.3; ncm[1][2]= 0.3; ncm[1][3]=0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 2.0; ncm[2][1]=0.3; ncm[2][2]= 0.3; ncm[2][3]=0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=10.0; ncm[3][1]=1.0; ncm[3][2]= 1.0; ncm[3][3]=0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        for(int j=0; j<7; j++) {
            ncm[4][j] = 0.0;
        } 
    } else if(navah_manufactured_solution_number == 4) {
        /* MS-4 */
        ncm[0][0]= 1.0; ncm[0][1]=  0.1; ncm[0][2]= -0.2; ncm[0][3]= 0.1; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 2.0; ncm[1][1]=  0.3; ncm[1][2]=  0.3; ncm[1][3]= 0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 2.0; ncm[2][1]=  0.3; ncm[2][2]=  0.3; ncm[2][3]= 0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=10.0; ncm[3][1]=  1.0; ncm[3][2]=  1.0; ncm[3][3]= 0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        ncm[4][0]= 0.6; ncm[4][1]=-0.03; ncm[4][2]=-0.02; ncm[4][3]=0.02; ncm[4][4]=2.0; ncm[4][5]=1.0; ncm[4][6]=3.0;
    } else if(navah_manufactured_solution_number == 5) {
        /* MS-5 */
        ncm[0][0]= 1.0; ncm[0][1]= 0.1; ncm[0][2]=-0.2; ncm[0][3]=0.1; ncm[0][4]=1.0; ncm[0][5]=1.0; ncm[0][6]=1.0;
        ncm[1][0]= 2.0; ncm[1][1]= 0.3; ncm[1][2]= 0.3; ncm[1][3]=0.3; ncm[1][4]=3.0; ncm[1][5]=1.0; ncm[1][6]=1.0;
        ncm[2][0]= 2.0; ncm[2][1]= 0.3; ncm[2][2]= 0.3; ncm[2][3]=0.3; ncm[2][4]=1.0; ncm[2][5]=1.0; ncm[2][6]=1.0;
        ncm[3][0]=10.0; ncm[3][1]= 1.0; ncm[3][2]= 1.0; ncm[3][3]=0.5; ncm[3][4]=2.0; ncm[3][5]=1.0; ncm[3][6]=1.0;
        ncm[4][0]=-6.0; ncm[4][1]=-0.3; ncm[4][2]=-0.2; ncm[4][3]=0.2; ncm[4][4]=2.0; ncm[4][5]=1.0; ncm[4][6]=3.0;
    }
    return ncm;
}

} // Parameters namespace

} // PHiLiP namespace
