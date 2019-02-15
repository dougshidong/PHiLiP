#include <deal.II/base/parameter_handler.h>
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__
namespace Parameters
{
    using namespace dealii;

    // Prints usage message in case the user does not provide
    // an input file, or an incorrectly formatted input file
    void print_usage_message (ParameterHandler &prm);

    // Parses command line for input line and reads parameters
    // into the ParameterHandler object
    void parse_command_line ( const int argc, char *const *argv,
                              ParameterHandler &parameter_handler);
    class ODE
    {
    public:
        ODE ();
        enum SolverType { explicit_solver, implicit_solver };
        SolverType solver_type;

        unsigned int nonlinear_max_iterations;
        double nonlinear_residual;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

    class ManufacturedConvergenceStudy
    {
    public:
        ManufacturedConvergenceStudy ();
        enum PartialDifferentialEquation { advection, convection_diffusion };
        PartialDifferentialEquation pde_type;

        unsigned int nonlinear_max_iterations;
        double nonlinear_residual;

        unsigned int dimension;
        unsigned int degree_start;
        unsigned int degree_end;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

    class AllParameters
        : public ManufacturedConvergenceStudy
        //,public Output
    {
    public:

        AllParameters();
        //FunctionParser<dim> initial_conditions;
        //BoundaryConditions  boundary_conditions[max_n_boundaries];
        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);

        //Parameters::Refinement::declare_parameters (prm);
        //Parameters::Flux::declare_parameters (prm);
        //Parameters::Output::declare_parameters (prm);
    };  
}

#endif
