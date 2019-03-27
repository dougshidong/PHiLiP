#ifndef __MANUFACTURED_CONVECTION_DIFFUSION_H__
#define __MANUFACTURED_CONVECTION_DIFFUSION_H__

#include <deal.II/base/point.h>
namespace PHiLiP
{
    using namespace dealii;

    template<int dim>
    double manufactured_convection_diffusion_solution (const Point<dim> point);

    template<int dim>
    double manufactured_convection_diffusion_source (const Point<dim> point);
}
#endif

