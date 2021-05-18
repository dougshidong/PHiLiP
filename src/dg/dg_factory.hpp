#ifndef __DG_FACTORY_H__
#define __DG_FACTORY_H__

#include "dg.h"

namespace PHiLiP {

/// This class creates a new DGBase object
/** This allows the DGBase to not be templated on the number of state variables
  * while allowing DG to be template on the number of state variables */
#if PHILIP_DIM==1 // dealii::parallel::distributed::Triangulation<dim> does not work for 1D
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DGFactory
{
    /** Triangulation to store the grid.
     *  In 1D, dealii::Triangulation<dim> is used.
     *  In 2D, 3D, dealii::parallel::distributed::Triangulation<dim> is used.
     */
    using Triangulation = MeshType;
public:
    /// Creates a derived object DG, but returns it as DGBase.
    /** That way, the caller is agnostic to the number of state variables */
    static std::shared_ptr< DGBase<dim,real,MeshType> >
        create_discontinuous_galerkin(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const unsigned int grid_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// calls the above dg factory with grid_degree_input = degree + 1
    static std::shared_ptr< DGBase<dim,real,MeshType> >
        create_discontinuous_galerkin(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const unsigned int max_degree_input,
        const std::shared_ptr<Triangulation> triangulation_input);

    /// calls the above dg factory with max_degree_input = degree
    static std::shared_ptr< DGBase<dim,real,MeshType> >
        create_discontinuous_galerkin(
        const Parameters::AllParameters *const parameters_input,
        const unsigned int degree,
        const std::shared_ptr<Triangulation> triangulation_input);
};

} // PHiLiP namespace

#endif
