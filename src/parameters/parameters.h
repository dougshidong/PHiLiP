#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
namespace Parameters {

enum OutputEnum { quiet, verbose };

/// Prints usage message in case the user does not provide
/// an input file, or an incorrectly formatted input file
void print_usage_message (dealii::ParameterHandler &prm);

/// Parses command line for input line and reads parameters
/// into the dealii::ParameterHandler object
void parse_command_line ( const int argc, char *const *argv,
                          std::vector<dealii::ParameterHandler> &parameter_handler);

} // Parameters namespace
} // PHiLiP namespace

#endif
