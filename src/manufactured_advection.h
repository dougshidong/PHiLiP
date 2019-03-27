#ifndef __MANUFACTURED_ADVECTION_H__
#define __MANUFACTURED_ADVECTION_H__

#include <deal.II/base/point.h>
namespace PHiLiP
{
    using namespace dealii;

    template<int dim>
    double manufactured_advection_solution (const Point<dim> point);

    template<int dim>
    double manufactured_advection_source (const Point<dim> point);
}
#endif

