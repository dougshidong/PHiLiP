#include "halton_sampling.h"
#include <iostream>
#include <filesystem>
#include "functional/functional.h"
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/vector_operation.h>
#include "reduced_order_solution.h"
#include "flow_solver/flow_solver.h"
#include "flow_solver/flow_solver_factory.h"
#include <cmath>
#include "rbf_interpolation.h"
#include "ROL_Algorithm.hpp"
#include "ROL_LineSearchStep.hpp"
#include "ROL_StatusTest.hpp"
#include "ROL_Stream.hpp"
#include "ROL_Bounds.hpp"
#include "halton.h"
#include "min_max_scaler.h"
#include <deal.II/base/timer.h>

namespace PHiLiP {

template<int dim, int nstate>
HaltonSampling<dim, nstate>::HaltonSampling(const PHiLiP::Parameters::AllParameters *const parameters_input,
                                                const dealii::ParameterHandler &parameter_handler_input)
        : AdaptiveSamplingBase<dim, nstate>(parameters_input, parameter_handler_input)
{}

template <int dim, int nstate>
int HaltonSampling<dim, nstate>::run_sampling() const
{
    this->pcout << "Starting Halton sampling process" << std::endl;
    auto stream = this->pcout;
    dealii::TimerOutput timer(stream,dealii::TimerOutput::summary,dealii::TimerOutput::wall_times);
    timer.enter_subsection ("Solve FOMs and Assemble POD");
    this->placeInitialSnapshots();
    this->current_pod->computeBasis();
    
    timer.leave_subsection();

    this->outputIterationData("final");

    return 0;
}

#if PHILIP_DIM==1
        template class HaltonSampling<PHILIP_DIM, PHILIP_DIM>;
#endif

#if PHILIP_DIM!=1
        template class HaltonSampling<PHILIP_DIM, PHILIP_DIM+2>;
#endif

}