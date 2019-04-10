#ifndef __BOUNDARY_H__
#define __BOUNDARY_H__

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include "manufactured_solution.h"
namespace PHiLiP
{
    using namespace dealii;

    template <int dim>
    class Boundary : public Function<dim>
    {
    public:
        Boundary() = default;
        virtual void value_list(const std::vector<Point<dim>> &points,
                                std::vector<double> &          values,
                                const unsigned int /*component = 0*/) const override;
    };

    template <int dim>
    void Boundary<dim>::value_list(const std::vector<Point<dim> > &points,
                                            std::vector<double> &values,
                                            const unsigned int /*component*/) const
    {
        Assert(values.size()==points.size(),
               ExcDimensionMismatch(values.size(),points.size()));
        for (unsigned int i=0; i<values.size(); ++i) {
            values[i] = manufactured_solution (points[i]);
        }
    }
} // end of PHiLiP namespace

#endif
