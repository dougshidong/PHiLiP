#include "design_parameterization_base.hpp"

namespace PHiLiP {

template<int dim>
DesignParameterizationBase<dim> :: DesignParameterizationBase (
    std::shared_ptr<HighOrderGrid<dim,double>> _high_order_grid)
    : high_order_grid(_high_order_grid)
{}

template<int dim>
DesignParameterizationBase<dim> :: ~DesignParameterizationBase()
{
    // Does nothing for now. Overriden in derived classes.
}

template<int dim>
void DesignParameterizationBase<dim> :: output_design_variables(unsigned int /*iteration_no*/) const
{
    // Does nothing by default. Overriden in derived classes.
}

template class DesignParameterizationBase<PHILIP_DIM>;
} // PHiLiP namespace
