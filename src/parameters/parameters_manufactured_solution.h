#ifndef __PARAMETERS_MANUFACTURED_SOLUTION_H__
#define __PARAMETERS_MANUFACTURED_SOLUTION_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

#include <deal.II/base/tensor.h>

namespace PHiLiP {

namespace Parameters {

/// Parameters related to the manufactured convergence study
class ManufacturedSolutionParam
{
public:
    ManufacturedSolutionParam(); ///< Constructor

    /// Uses non-zero source term based on the manufactured solution and the PDE.
    bool use_manufactured_source_term;
    
    /// Selects the manufactured solution to be used if use_manufactured_source_term=true
    enum ManufacturedSolutionType{
        sine_solution,
        cosine_solution,
        additive_solution,
        exp_solution,
        poly_solution,
        even_poly_solution,
        atan_solution,
        boundary_layer_solution,
        s_shock_solution,
        quadratic_solution,
        navah_solution
        };
    ManufacturedSolutionType manufactured_solution_type; ///< Selected ManufacturedSolutionType from the input file

    /// Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
    /// Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
    std::array<dealii::Tensor<1,7,double>,5> NavahCoefficientMatrix;

    /// Diffusion tensor
    dealii::Tensor<2,3,double> diffusion_tensor;

    /// Advection velocity
    dealii::Tensor<1,3,double> advection_vector;

    /// Diffusion coefficient
    double diffusion_coefficient;

    /// Declares the possible variables and sets the defaults.
    static void declare_parameters (dealii::ParameterHandler &prm);
    /// Parses input file and sets the variables.
    void parse_parameters (dealii::ParameterHandler &prm);

    /// gets the default diffusion tensor
    static dealii::Tensor<2,3,double> get_default_diffusion_tensor();
    /// gets the default advection vector
    static dealii::Tensor<1,3,double> get_default_advection_vector();
    /// gets the default diffusion coefficient;
    static double get_default_diffusion_coefficient();

    /// Gets the NavahCoefficientMatrix for the specified navah_solution. 
    /// Matrix with all coefficients of the various manufactured solutions given in Navah's paper. 
    /// Reference: Navah F. and Nadarajah S., A comprehensive high-order solver verification methodology for free fluid flows, 2018
    static std::array<dealii::Tensor<1,7,double>,5> get_navah_coefficient_matrix(int navah_manufactured_solution_number);
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_MANUFACTURED_SOLUTION_H__

