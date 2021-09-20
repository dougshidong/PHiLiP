#include "parameters/parameters_reduced_order.h"
namespace PHiLiP {
    namespace Parameters {

// NavierStokes inputs
        ReducedOrderModelParam::ReducedOrderModelParam () {}

        void ROMParam::declare_parameters (dealii::ParameterHandler &prm)
        {
            prm.enter_subsection("reduced order");
            {
                prm.declare_entry("mach_number", "0.70",
                                  dealii::Patterns::Double(0, 1.5),
                                  "Mach number");
            }
            prm.leave_subsection();
        }

        void ReducedOrderModelParam::parse_parameters (dealii::ParameterHandler &prm)
        {
            prm.enter_subsection("reduced order");
            {
                mach_number = prm.get_double("mach_number");
            }
            prm.leave_subsection();
        }

    } // Parameters namespace
} // PHiLiP namespace
