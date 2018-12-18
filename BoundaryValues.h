#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/mapping_q1.h> // Might need mapping_q

#include <deal.II/fe/fe_dgq.h>

// We are going to use the simplest possible solver, called Richardson
// iteration, that represents a simple defect correction. This, in combination
// with a block SSOR preconditioner (defined in precondition_block.h), that
// uses the special block matrix structure of system matrices arising from DG
// discretizations.
#include <deal.II/lac/solver_richardson.h>
#include <deal.II/lac/precondition_block.h>
// We are going to use gradients as refinement indicator.
#include <deal.II/numerics/derivative_approximation.h>

// Here come the new include files for using the MeshWorker framework. The first
// contains the class MeshWorker::DoFInfo, which provides local integrators with
// a mapping between local and global degrees of freedom. It stores the results
// of local integrals as well in its base class MeshWorker::LocalResults.
// In the second of these files, we find an object of type
// MeshWorker::IntegrationInfo, which is mostly a wrapper around a group of
// FEValues objects. The file <tt>meshworker/simple.h</tt> contains classes
// assembling locally integrated data into a global system containing only a
// single matrix. Finally, we will need the file that runs the loop over all
// mesh cells and faces.
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/simple.h>
#include <deal.II/meshworker/loop.h>

// Like in all programs, we finish this section by including the needed C++
// headers and declaring we want to use objects in the dealii namespace without
// prefix.
#include <iostream>
#include <fstream>
namespace PHiLiP
{
  using namespace dealii;

  // @sect3{Equation data}
  //
  // First, we define a class describing the inhomogeneous boundary data. Since
  // only its values are used, we implement value_list(), but leave all other
  // functions of Function undefined.
  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    BoundaryValues() = default;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &          values,
                            const unsigned int component = 0) const override;
  };

  // Given the flow direction, the inflow boundary of the unit square $[0,1]^2$
  // are the right and the lower boundaries. We prescribe discontinuous boundary
  // values 1 and 0 on the x-axis and value 0 on the right boundary. The values
  // of this function on the outflow boundaries will not be used within the DG
  // scheme.
  template <int dim>
  void BoundaryValues<dim>::value_list(const std::vector<Point<dim>> &points,
                                       std::vector<double> &          values,
                                       const unsigned int component) const
  {
    (void)component;
    AssertIndexRange(component, 1);
    Assert(values.size() == points.size(),
           ExcDimensionMismatch(values.size(), points.size()));

    for (unsigned int i = 0; i < values.size(); ++i)
      {
        if (points[i](0) < 0.5)
          values[i] = 1.;
        else
          values[i] = 0.;
      }
  }


  // Finally, a function that computes and returns the wind field
  // $\beta=\beta(\mathbf x)$. As explained in the introduction, we will use a
  // rotational field around the origin in 2d. In 3d, we simply leave the
  // $z$-component unset (i.e., at zero), whereas the function can not be used
  // in 1d in its current implementation:
  template <int dim>
  Tensor<1, dim> beta(const Point<dim> &p)
  {
    Assert(dim >= 2, ExcNotImplemented());

    Point<dim> wind_field;
    wind_field(0) = -p(1);
    wind_field(1) = p(0);
    wind_field /= wind_field.norm();

    return wind_field;
  }
}

