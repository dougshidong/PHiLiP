#include <deal.II/base/mpi.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>
#include "parameters.h"

namespace PHiLiP {
namespace Parameters {

void print_usage_message (dealii::ParameterHandler &prm);

void parse_command_line (const int argc, char *const *argv,
                         std::vector<dealii::ParameterHandler> &parameter_handler)
{
    if (argc < 2) {
        print_usage_message (parameter_handler[0]);
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
                print_usage_message (parameter_handler[0]);
                exit (1);
            }
            args.pop_front ();
            const int number_of_parameter_file = args.size();
            for (int i=0;i<number_of_parameter_file;++i){
                const std::string input_file_name = args.front ();
                args.pop_front ();
                try {
                    parameter_handler[i].parse_input(input_file_name);
                }
                catch (std::exception &exc)
                {
                    std::cerr << std::endl << std::endl
                              << "----------------------------------------------------"
                              << std::endl;
                    std::cerr << "Error: unable to parse parameter file named "
                              << input_file_name
                              << std::endl;
                    std::cerr << "Exception on processing: " << std::endl
                              << exc.what() << std::endl
                              << "Aborting!" << std::endl
                              << "----------------------------------------------------"
                              << std::endl;
                    print_usage_message (parameter_handler[i]);
                    exit (1);
                }
            }

        } else {
            std::cerr << "Error: unknown flag '"
                     << args.front()
                     << "'"
                     << std::endl;
            print_usage_message (parameter_handler[0]);
            exit (1);
        }

    }

}


void print_usage_message (dealii::ParameterHandler &prm)
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
    dealii::ConditionalOStream pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
    pcout << message;
    if (pcout.is_active()) prm.print_parameters (pcout.get_stream(), dealii::ParameterHandler::Text);
}

unsigned int get_number_of_values_in_string(const std::string string_of_values)
{
    std::string line = string_of_values;
    unsigned int count = 0;
    std::string::size_type sz1;
    while(line!="" && line!=" ") {
        count += 1;
        std::stod(line,&sz1);
        line = line.substr(sz1);
        sz1 = 0;
    }
    return count;
}

} // Parameters namespace
} // PHiLiP namespace
