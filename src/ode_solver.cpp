
#include "ode_solver.h"

int ODESolver::steady_state ()
{
    allocate_system ();

    dg->assemble_system ();
    residual_norm = dg->right_hand_side.l2_norm();

    while (    residual_norm     > parameters->nonlinear_steady_residual_tolerance 
            && current_iteration < parameters->nonlinear_max_iterations )
    {
        ++current_iteration;

        if ( (current_iteration%parameters->print_iteration_modulo) == 0 )
        std::cout << " Iteration: " << current_iteration 
                  << " Residual norm: " << residual_norm
                  << std::endl;

        get_solution_update (solution_update);
        solution += solution_update;

        dg->assemble_system ();
        residual_norm = dg->right_hand_side.l2_norm();
    }
    return 1;
};

void Explicit_ODESolver::get_solution_update ()
{
    solution += (dg->right_hand_side*=dt);
}
