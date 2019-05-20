#ifndef __PARAMETERS_MANUFACTURED_CONVERGENCE_STUDY_H__
#define __PARAMETERS_MANUFACTURED_CONVERGENCE_STUDY_H__

#include <deal.II/base/parameter_handler.h>
#include "parameters/parameters.h"

namespace Parameters
{
    using namespace dealii;

    class ManufacturedConvergenceStudyParam
    {
    public:
        ManufacturedConvergenceStudyParam ();

        enum GridEnum { hypercube, sinehypercube, read_grid };

        GridEnum grid_type;
        std::string input_grids;

        double random_distortion;
        bool output_meshes;

        unsigned int degree_start;
        unsigned int degree_end;
        unsigned int initial_grid_size;
        unsigned int number_of_grids;
        double grid_progression;

        static void declare_parameters (ParameterHandler &prm);
        void parse_parameters (ParameterHandler &prm);
    };

}

#endif

