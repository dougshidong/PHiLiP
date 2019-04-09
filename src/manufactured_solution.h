#ifndef __MANUFACTURED_SOLUTION_H__
#define __MANUFACTURED_SOLUTION_H__

//#include "manufactured_advection.h"
//#include "manufactured_diffusion.h"
//#include "manufactured_convection_diffusion.h"
#include <deal.II/base/point.h>
namespace PHiLiP
{
    using namespace dealii;

    template<int dim>
    double manufactured_solution (const Point<dim> point);

    template<int dim>
    double manufactured_advection_source (const Point<dim> point);

    template<int dim>
    double manufactured_diffusion_source (const Point<dim> point);

    template<int dim>
    double manufactured_convection_diffusion_source (const Point<dim> point);
}

#endif
