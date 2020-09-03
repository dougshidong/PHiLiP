#include "manufactured_solution.h"
#include <deal.II/base/point.h>

namespace PHiLiP {
    using namespace dealii;

    template<int dim>
    double manufactured_solution (const Point<dim> point)
    {
        double uexact;

        const double a = 1*1.59/dim;
        const double b = 2*1.81/dim;
        const double c = 3*1.76/dim;
        const double d = 1, e = 1.2, f = 1.5;
        if (dim==1) uexact = sin(a*point(0)+d);
        if (dim==2) uexact = sin(a*point(0)+d)*sin(b*point(1)+e);
        if (dim==3) uexact = sin(a*point(0)+d)*sin(b*point(1)+e)*sin(c*point(2)+f);
        return uexact;
    }
    template double manufactured_solution<PHILIP_DIM> (const Point<PHILIP_DIM>);

    template<int dim>
    double manufactured_advection_source (const Point<dim> point)
    {
        double source;

        const double a = 1*1.59/dim;
        const double b = 2*1.81/dim;
        const double c = 3*1.76/dim;
        const double d = 1, e = 1.2, f = 1.5;
        if (dim==1) {
            const double x = point(0);
            source = a*cos(a*x+d);
        } else if (dim==2) {
            const double x = point(0), y = point(1);
            source = a*cos(a*x+d)*sin(b*y+e) +
                     b*sin(a*x+d)*cos(b*y+e);
        } else if (dim==3) {
            const double x = point(0), y = point(1), z = point(2);
            source =   a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                       c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f);
        }
        return source;
    }
    template double manufactured_advection_source<PHILIP_DIM> (const Point<PHILIP_DIM>);

    template<int dim>
    double manufactured_diffusion_source (const Point<dim> point)
    {
        double source;

        const double a = 1*1.59/dim;
        const double b = 2*1.81/dim;
        const double c = 3*1.76/dim;
        const double d = 1, e = 1.2, f = 1.5;
        if (dim==1) {
            const double x = point(0);
            source = a*a*sin(a*x+d);
        } else if (dim==2) {
            const double x = point(0), y = point(1);
            source = a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const double x = point(0), y = point(1), z = point(2);
            source =  a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                      c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
        return source;
    }
    template double manufactured_diffusion_source<PHILIP_DIM> (const Point<PHILIP_DIM>);

    template<int dim>
    double manufactured_convection_diffusion_source (const Point<dim> point)
    {
        double source;

        const double a = 1*1.59/dim;
        const double b = 2*1.81/dim;
        const double c = 3*1.76/dim;
        const double d = 1, e = 1.2, f = 1.5;
        if (dim==1) {
            const double x = point(0);
            source = a*cos(a*x+d) +
                     a*a*sin(a*x+d);
        } else if (dim==2) {
            const double x = point(0), y = point(1);
            source = a*cos(a*x+d)*sin(b*y+e) +
                     b*sin(a*x+d)*cos(b*y+e) +
                     a*a*sin(a*x+d)*sin(b*y+e) +
                     b*b*sin(a*x+d)*sin(b*y+e);
        } else if (dim==3) {
            const double x = point(0), y = point(1), z = point(2);
            source =   a*cos(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       b*sin(a*x+d)*cos(b*y+e)*sin(c*z+f) +
                       c*sin(a*x+d)*sin(b*y+e)*cos(c*z+f) +
                       a*a*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       b*b*sin(a*x+d)*sin(b*y+e)*sin(c*z+f) +
                       c*c*sin(a*x+d)*sin(b*y+e)*sin(c*z+f);
        }
        return source;
    }
    template double manufactured_convection_diffusion_source<PHILIP_DIM> (const Point<PHILIP_DIM>);
}


