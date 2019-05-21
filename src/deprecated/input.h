#ifndef __INPUT_H__
#define __INPUT_H__

#include <deal.II/base/parameter_handler.h>
namespace Parameters
{
    using namespace dealii;

    void print_usage_message (ParameterHandler &prm)
    {
      static const char *message
        =
          "\n"
          "deal.II intermediate format to other graphics formats.\n"
          "\n"
          "Usage:\n"
          "    ./PHiLiP [-p input_file] input_file \n"
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
    void parse_command_line (
        const int argc, 
        char *const *argv,
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

            if (args.front() == std::string("-p")) {

                if (args.size() == 1) {
                    std::cerr << "Error: flag '-p' must be followed by the "
                              << "name of a parameter file."
                              << std::endl;
                    print_usage_message (parameter_handler);
                    exit (1);
                }
                args.pop_front ();
                const std::string input_file = args.front ();
                args.pop_front ();
                try {
                    parameter_handler.parse_input(input_file);
                }
                catch (...) {
                    std::cerr << "Error: unable to process parameter file "
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

    class ManufacturedConvergenceStudy
    {
    public:
        ManufacturedConvergenceStudy ();
        enum PartialDifferentialEquation { advection, convection_diffusion };
        PartialDifferentialEquation pde_type;
        unsigned int dimension;
        unsigned int degree_start;
        unsigned int degree_end;

        unsigned int nonlinear_max_iterations;
        double nonlinear_residual;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };
    ManufacturedConvergenceStudy::ManufacturedConvergenceStudy () {}
    void ManufacturedConvergenceStudy::declare_parameters (ParameterHandler &prm)
    {
        prm.enter_subsection("manufactured solution convergence study");
        {
            //prm.declare_entry("output", "quiet",
            //                  Patterns::Selection("quiet|verbose"),
            //                  "State whether output from solver runs should be printed. "
            //                  "Choices are <quiet|verbose>.");
            prm.declare_entry("pde_type", "advection",
                              Patterns::Selection("advection|convection_diffusion"),
                              "The kind of solver for the linear system. "
                              "Choices are <advection|convection_diffusion>.");
            prm.declare_entry("nonlinear_max_iterations", "500000",
                              Patterns::Integer(),
                              "Maximum nonlinear solver iterations");
            prm.declare_entry("nonlinear_residual", "1e-13",
                              Patterns::Double(),
                              "Nonlinear solver residual tolerance");
            prm.declare_entry("dimension", "1",
                              Patterns::Integer(),
                              "Number of dimensions");

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

            const std::string pde_string = prm.get("method");
            if (pde_string == "advection") pde_type = advection;
            if (pde_string == "convection_diffusion") pde_type = convection_diffusion;

            nonlinear_residual          = prm.get_double("nonlinear_residual");

            nonlinear_max_iterations    = prm.get_integer("nonlinear_max_iterations");
            dimension                   = prm.get_integer("dimension");
            degree_start                = prm.get_integer("degree_start");
            degree_end                  = prm.get_integer("degree_end");
        }
        prm.leave_subsection();
    }

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
    AllParameters::AllParameters () {}
    void AllParameters::declare_parameters (ParameterHandler &prm)
    {
        Parameters::ManufacturedConvergenceStudy::declare_parameters (prm);
    }
    void AllParameters::parse_parameters (ParameterHandler &prm)
    {
        Parameters::ManufacturedConvergenceStudy::parse_parameters (prm);
    }
}
#endif
