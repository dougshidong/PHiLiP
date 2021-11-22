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
        Alex_solution,
        navah_solution_1,
        navah_solution_2,
        navah_solution_3,
        navah_solution_4,
        navah_solution_5
        };
    ManufacturedSolutionType manufactured_solution_type; ///< Selected ManufacturedSolutionType from the input file

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
};

} // Parameters namespace

} // PHiLiP namespace

#endif // __PARAMETERS_MANUFACTURED_SOLUTION_H__

