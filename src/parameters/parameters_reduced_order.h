#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
    namespace Parameters {
/// Parameters related to basis creation for reduced-order model
        class ReducedOrderModelParam
        {
        public:
            double mach_number;

            ReducedOrderModelParam (); ///< Constructor

            /// Declares the possible variables and sets the defaults.
            static void declare_parameters (dealii::ParameterHandler &prm);
            /// Parses input file and sets the variables.
            void parse_parameters (dealii::ParameterHandler &prm);
        };

    } // Parameters namespace
} // PHiLiP namespace
#endif
