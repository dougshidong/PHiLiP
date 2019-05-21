#ifndef __FLOWSOLVER_H__
#define __FLOWSOLVER_H__

#include <deal.II/lac/vector.h>

#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include "parameters.h"


namespace PHiLiP
{
    using namespace dealii;

    class FlowSolver
    {
    public:
        FlowSolver();

        int solve();

        Vector<real> solution;
        Vector<real> residual;

    private:

        Parameters::AllParameters *parameters;

        // Mesh
        Triangulation<dim>        triangulation;
        ODESolver                 odesolver;
        SpaceDiscretization<dim>  spacediscretization;

    }; // end of FlowSolver class

} // end of PHiLiP namespace

#endif


