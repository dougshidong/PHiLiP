#ifndef __MESH_H__
#define __MESH_H__

#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>

#include "parameters.h"


namespace PHiLiP
{
    using namespace dealii;

    class Mesh
    {
    public:
        Mesh(Parameters::AllParameters *parameters_input, const unsigned int degree)

    private:


    }; // end of Mesh class
    template<int dim>
    class Mesh_dim
    {
    public:

    private:
        Triangulation<dim>  triangulation_input;
    }; // end of Mesh class

#endif

