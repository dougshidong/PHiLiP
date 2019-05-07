#include <deal.II/base/parameter_handler.h>
#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__
namespace Parameters
{
    using namespace dealii;

    enum OutputType { quiet, verbose };

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

        OutputType ode_output;
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

        enum GridType { hypercube, sinehypercube, read_grid };

        GridType grid_type;
        std::string input_grids;

        double random_distortion;
        bool output_meshes;

        unsigned int degree_start;
        unsigned int degree_end;
        unsigned int initial_grid_size;
        unsigned int number_of_grids;
        double grid_progression;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

    class LinearSolver
    {
    public:
        LinearSolver ();
        enum LinearSolverType { direct, gmres };

        OutputType linear_solver_output;
        LinearSolverType linear_solver_type;

        // GMRES options
        double ilut_drop, ilut_rtol, ilut_atol;
        int ilut_fill;

        double linear_residual;
        int max_iterations;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

    class AllParameters
        : public ManufacturedConvergenceStudy
        , public ODE
        , public LinearSolver
        //,public Output
    {
    public:
        unsigned int dimension;
        enum PartialDifferentialEquation { advection, diffusion, convection_diffusion };
        PartialDifferentialEquation pde_type;

        enum ConvectiveNumericalFlux { lax_friedrichs };
        ConvectiveNumericalFlux conv_num_flux_type;

        enum DissipativeNumericalFlux { symm_internal_penalty };
        DissipativeNumericalFlux diss_num_flux_type;

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
