#include <deal.II/base/parameter_handler.h>
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__
namespace Parameters
{
    using namespace dealii;

    // Be careful about multiple inheritance
    // AllParameters will have acess to all the various varibles
    // declared in the other classes.
    // There might be a conflict if there are variables with the same name

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
        unsigned int print_iteration_modulo;
        double nonlinear_steady_residual_tolerance;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

    class ManufacturedConvergenceStudy
    {
    public:
        ManufacturedConvergenceStudy ();

        unsigned int degree_start;
        unsigned int degree_end;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

    class AllParameters
        : public ManufacturedConvergenceStudy
        , public ODE
        //,public Output
    {
    public:
        unsigned int dimension;
        enum PartialDifferentialEquation { advection, convection_diffusion };
        PartialDifferentialEquation pde_type;

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
