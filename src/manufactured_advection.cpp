#include "manufactured_advection.h"
#include <deal.II/base/point.h>

namespace PHiLiP
{
    using namespace dealii;

    template<int dim>
    double manufactured_advection_solution (const Point<dim> point)
    {
        double uexact;

        const double a = 1*1.19/dim;
        const double b = 2*1.19/dim;
        const double c = 3*1.19/dim;
        if (dim==1) uexact = sin(a*point(0));
        if (dim==2) uexact = sin(a*point(0))*sin(b*point(1));
        if (dim==3) uexact = sin(a*point(0))*sin(b*point(1))*sin(c*point(2));
        return uexact;
    }
    template double manufactured_advection_solution<PHILIP_DIM> (const Point<PHILIP_DIM>);

    template<int dim>
    double manufactured_advection_source (const Point<dim> point)
    {
        double source;

        const double a = 1*1.19/dim;
        const double b = 2*1.19/dim;
        const double c = 3*1.19/dim;
        if (dim==1) {
            const double x = point(0);
            source = a*cos(a*x);
        } else if (dim==2) {
            const double x = point(0), y = point(1);
            source = a*cos(a*x)*sin(b*y) + b*sin(a*x)*cos(b*y);
        } else if (dim==3) {
            const double x = point(0), y = point(1), z = point(2);
            source =   a*cos(a*x)*sin(b*y)*sin(c*z)
                     + b*sin(a*x)*cos(b*y)*sin(c*z)
                     + c*sin(a*x)*sin(b*y)*cos(c*z);
        }
        return source;
    }
    template double manufactured_advection_source<PHILIP_DIM> (const Point<PHILIP_DIM>);
}
