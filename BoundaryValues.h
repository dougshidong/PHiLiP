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

