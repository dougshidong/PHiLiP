#ifndef __PARAMETERS_REDUCED_ORDER_H__
#define __PARAMETERS_REDUCED_ORDER_H__

#include <deal.II/base/parameter_handler.h>

namespace PHiLiP {
    namespace Parameters {
/// Parameters related to reduced-order model
        class ReducedOrderModelParam
        {
        public:
            /// Parameter a for eq.(18) in Carlberg 2011
            double rewienski_a;

            /// Parameter b for eq.(18) in Carlberg 2011
            double rewienski_b;

            /// Final solution time for PDE
            double final_time;

            ReducedOrderModelParam (); ///< Constructor

            /// Declares the possible variables and sets the defaults.
            static void declare_parameters (dealii::ParameterHandler &prm);
            /// Parses input file and sets the variables.
            void parse_parameters (dealii::ParameterHandler &prm);
        };

    } // Parameters namespace
} // PHiLiP namespace
#endif
