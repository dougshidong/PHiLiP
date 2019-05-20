#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <deal.II/base/parameter_handler.h>

namespace Parameters
{
    using namespace dealii;

    enum OutputEnum { quiet, verbose };

    /// Prints usage message in case the user does not provide
    /// an input file, or an incorrectly formatted input file
    void print_usage_message (ParameterHandler &prm);

    /// Parses command line for input line and reads parameters
    /// into the ParameterHandler object
    void parse_command_line ( const int argc, char *const *argv,
                              ParameterHandler &parameter_handler);
}

#endif
