#include <deal.II/base/parameter_handler.h>

#include "parameters.h"

namespace Parameters
{
    using namespace dealii;

    void print_usage_message (ParameterHandler &prm);

    void parse_command_line (const int argc, char *const *argv,
                             ParameterHandler &parameter_handler)
    {
        if (argc < 2) {
            print_usage_message (parameter_handler);
            exit (1);
        }
        std::list<std::string> args;
        for (int i=1; i<argc; ++i) {
            args.emplace_back(argv[i]);
        }
        while (args.size()) {

            if (args.front() == std::string("-i")) {

                if (args.size() == 1) {
                    std::cerr << "Error: flag '-i' must be followed by the "
                              << "name of a parameter file."
                              << std::endl;
                    print_usage_message (parameter_handler);
                    exit (1);
                }
                args.pop_front ();
                const std::string input_file_name = args.front ();
                args.pop_front ();
                try {
                    parameter_handler.parse_input(input_file_name);
                }
                catch (...) {
                    std::cerr << "Error: unable to parse parameter file named "
                              << input_file_name
                              << std::endl;
                    print_usage_message (parameter_handler);
                    exit (1);
                }
            //} else if (args.front() == std::string("-o")) {

            //    if (args.size() == 1) {
            //        std::cerr << "Error: flag '-o' must be followed by the "
            //                  << "name of an output file."
            //                  << std::endl;
            //        print_usage_message (parameter_handler);
            //        exit (1);
            //    }
            //    args.pop_front ();
            //    output_file = args.front();
            //    args.pop_front ();
            } else {
                std::cerr << "Error: unknown flag '"
                         << args.front()
                         << "'"
                         << std::endl;
                print_usage_message (parameter_handler);
                exit (1);
            }

        }

    }

    ODE::ODE () {}

    void ODE::declare_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("ODE solver");
        {
            prm.declare_entry("ode_output", "quiet",
                              Patterns::Selection("quiet|verbose"),
                              "State whether output from ODE solver should be printed. "
                              "Choices are <quiet|verbose>.");

            prm.declare_entry("solver_type", "implicit",
                              Patterns::Selection("explicit|implicit"),
                              "Explicit or implicit solver"
                              "Choices are <explicit|implicit>.");

            prm.declare_entry("nonlinear_max_iterations", "500000",
                              Patterns::Integer(1,Patterns::Integer::max_int_value),
                              "Maximum nonlinear solver iterations");
            prm.declare_entry("nonlinear_steady_residual_tolerance", "1e-13",
                              Patterns::Double(1e-16,Patterns::Double::max_double_value),
                              "Nonlinear solver residual tolerance");

            prm.declare_entry("print_iteration_modulo", "1",
                              Patterns::Integer(0,Patterns::Integer::max_int_value),
                              "Print every print_iteration_modulo iterations of "
                              "the nonlinear solver");
        }
        prm.leave_subsection();
    }

    void ODE::parse_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("ODE solver");
        {
            const std::string output_string = prm.get("ode_output");
            if (output_string == "quiet")   ode_output = OutputType::quiet;
            if (output_string == "verbose") ode_output = OutputType::verbose;

            const std::string solver_string = prm.get("solver_type");
            if (solver_string == "explicit") solver_type = SolverType::explicit_solver;
            if (solver_string == "implicit") solver_type = SolverType::implicit_solver;

            nonlinear_steady_residual_tolerance  = prm.get_double("nonlinear_steady_residual_tolerance");
            nonlinear_max_iterations = prm.get_integer("nonlinear_max_iterations");

            print_iteration_modulo = prm.get_integer("print_iteration_modulo");
        }
        prm.leave_subsection();
    }



    // Manufactured Solution inputs
    ManufacturedConvergenceStudy::ManufacturedConvergenceStudy () {}

    void ManufacturedConvergenceStudy::declare_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("manufactured solution convergence study");
        {
            //prm.declare_entry("output", "quiet",
            //                  Patterns::Selection("quiet|verbose"),
            //                  "State whether output from solver runs should be printed. "
            //                  "Choices are <quiet|verbose>.");
            prm.declare_entry("initial_grid_size", "2",
                              Patterns::Integer(),
                              "Initial grid of size (initial_grid_size)^dim");
            prm.declare_entry("number_of_grids", "4",
                              Patterns::Integer(),
                              "Number of grids in grid study");
            prm.declare_entry("grid_progression", "1.5",
                              Patterns::Double(),
                              "Multiplier on grid size. "
                              "nth-grid will be of size (initial_grid^grid_progression)^dim");

            prm.declare_entry("degree_start", "0",
                              Patterns::Integer(),
                              "Starting degree for convergence study");
            prm.declare_entry("degree_end", "3",
                              Patterns::Integer(),
                              "Last degree used for convergence study");
        }
        prm.leave_subsection();
    }

    void ManufacturedConvergenceStudy ::parse_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("manufactured solution convergence study");
        {
            //const std::string output_string = prm.get("output");
            //if (output_string == "verbose") output = verbose;
            //if (output_string == "quiet") output = quiet;

            degree_start                = prm.get_integer("degree_start");
            degree_end                  = prm.get_integer("degree_end");

            initial_grid_size           = prm.get_integer("initial_grid_size");
            number_of_grids             = prm.get_integer("number_of_grids");
            grid_progression            = prm.get_double("grid_progression");
        }
        prm.leave_subsection();
    }

    // Linear solver inputs
    LinearSolver::LinearSolver () {}

    void LinearSolver::declare_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("linear solver");
        {
            prm.declare_entry("linear_solver_output", "quiet",
                              Patterns::Selection("quiet|verbose"),
                              "State whether output from linear solver should be printed. "
                              "Choices are <quiet|verbose>.");

            prm.declare_entry("linear_solver_type", "gmres",
                              Patterns::Selection("direct|gmres"),
                              "Type of linear solver"
                              "Choices are <direct|gmres>.");

            prm.enter_subsection("gmres options");
            {
                prm.declare_entry("linear_residual_tolerance", "1e-4",
                                  Patterns::Double(),
                                  "Linear residual tolerance for convergence of the linear system");
                prm.declare_entry("max_iterations", "1000",
                                  Patterns::Integer(),
                                  "Maximum number of iterations for linear solver");

                // ILU with threshold parameters
                prm.declare_entry("ilut_fill", "2",
                                  Patterns::Integer(),
                                  "Amount of additional fill-in elements besides the sparse matrix structure");
                prm.declare_entry("ilut_drop", "1e-10",
                                  Patterns::Double(),
                                  "relative size of elements which should be dropped when forming an incomplete lu decomposition with threshold");
                prm.declare_entry("ilut_rtol", "1.1",
                                  Patterns::Double(),
                                  "Amount of an absolute perturbation that will be added to the diagonal of the matrix, "
                                  "which sometimes can help to get better preconditioners");
                prm.declare_entry("ilut_atol", "1e-9",
                                  Patterns::Double(),
                                  "Factor by which the diagonal of the matrix will be scaled, "
                                  "which sometimes can help to get better preconditioners");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
    }

    void LinearSolver ::parse_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("linear solver");
        {
            const std::string output_string = prm.get("linear_solver_output");
            if (output_string == "verbose") linear_solver_output = verbose;
            if (output_string == "quiet") linear_solver_output = quiet;

            const std::string solver_string = prm.get("linear_solver_type");
            if (solver_string == "direct") linear_solver_type = LinearSolverType::direct;

            if (solver_string == "gmres") 
            {
                linear_solver_type = LinearSolverType::gmres;
                prm.enter_subsection("gmres options");
                {
                    max_iterations  = prm.get_integer("max_iterations");
                    linear_residual = prm.get_double("linear_residual_tolerance");

                    ilut_fill = prm.get_integer("ilut_fill");
                    ilut_drop = prm.get_double("ilut_drop");
                    ilut_rtol = prm.get_double("ilut_rtol");
                    ilut_atol = prm.get_double("ilut_atol");
                }
                prm.leave_subsection();
            }
        }
        prm.leave_subsection();
    }

    AllParameters::AllParameters () {}
    void AllParameters::declare_parameters (ParameterHandler &prm)
    {
        prm.declare_entry("dimension", "1",
                          Patterns::Integer(),
                          "Number of dimensions");
        prm.declare_entry("pde_type", "advection",
                          Patterns::Selection("advection|diffusion|convection_diffusion"),
                          "The PDE we want to solve. "
                          "Choices are <advection|diffusion|convection_diffusion>.");
        prm.declare_entry("conv_num_flux", "lax_friedrichs",
                          Patterns::Selection("lax_friedrichs"),
                          "Convective numerical flux. "
                          "Choices are <lax_friedrichs>.");
        prm.declare_entry("diss_num_flux", "symm_internal_penalty",
                          Patterns::Selection("symm_internal_penalty"),
                          "Dissipative numerical flux. "
                          "Choices are <symm_internal_penalty>.");

        Parameters::LinearSolver::declare_parameters (prm);
        Parameters::ManufacturedConvergenceStudy::declare_parameters (prm);
        Parameters::ODE::declare_parameters (prm);
    }
    void AllParameters::parse_parameters (ParameterHandler &prm)
    {
        dimension                   = prm.get_integer("dimension");

        const std::string pde_string = prm.get("pde_type");
        if (pde_string == "advection") pde_type = advection;
        if (pde_string == "diffusion") pde_type = diffusion;
        if (pde_string == "convection_diffusion") pde_type = convection_diffusion;

        const std::string conv_num_flux_string = prm.get("conv_num_flux");
        if (conv_num_flux_string == "lax_friedrichs") conv_num_flux_type = lax_friedrichs;

        const std::string diss_num_flux_string = prm.get("diss_num_flux");
        if (diss_num_flux_string == "symm_internal_penalty") diss_num_flux_type = symm_internal_penalty;


        Parameters::LinearSolver::parse_parameters (prm);
        Parameters::ManufacturedConvergenceStudy::parse_parameters (prm);
        Parameters::ODE::parse_parameters (prm);
    }

    void print_usage_message (ParameterHandler &prm)
    {
        static const char *message
          =
            "\n"
            "deal.II intermediate format to other graphics formats.\n"
            "\n"
            "Usage:\n"
            "    ./PHiLiP [-i input_file_name] input_file_name \n"
            //"              [-x output_format] [-o output_file]\n"
            "\n"
            "Parameter sequences in brackets can be omitted if a parameter file is\n"
            "specified on the command line and if it provides values for these\n"
            "missing parameters.\n"
            "\n"
            "The parameter file has the following format and allows the following\n"
            "values (you can cut and paste this and use it for your own parameter\n"
            "file):\n"
            "\n";
        std::cout << message;
        prm.print_parameters (std::cout, ParameterHandler::Text);
    }
}
