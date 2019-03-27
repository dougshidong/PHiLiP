#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include "manufactured_solution.h"
namespace PHiLiP
{
    using namespace dealii;

    template <int dim>
    class AdvectionBoundary : public Function<dim>
    {
    public:
        AdvectionBoundary() = default;
        virtual void value_list(const std::vector<Point<dim>> &points,
                                std::vector<double> &          values,
                                const unsigned int /*component = 0*/) const override;
    };

    template <int dim>
    void AdvectionBoundary<dim>::value_list(const std::vector<Point<dim> > &points,
                                            std::vector<double> &values,
                                            const unsigned int /*component*/) const
    {
        Assert(values.size()==points.size(),
               ExcDimensionMismatch(values.size(),points.size()));
        for (unsigned int i=0; i<values.size(); ++i) {
            values[i] = manufactured_advection_solution (points[i]);
            //values[i] = 1.;
            //for (unsigned int idim=0; idim<dim; ++idim) {
            //    const double loc = points[i](idim);
            //    values[i] *= sin(3.19/dim*loc);
            //}
        }
    }
} // end of PHiLiP namespace
