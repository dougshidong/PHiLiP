#ifndef __RK_TABLEAU_BASE__
#define __RK_TABLEAU_BASE__

#include "dg/dg.h"
#include <deal.II/base/conditional_ostream.h>

namespace PHiLiP {
namespace ODE {

/// Base class for storing the RK method 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RKTableauBase
{
public:
    /// Default constructor that will set the constants.
    RKTableauBase(); 

    /// Destructor
    ~RKTableauBase() {};

    /// Returns Butcher tableau "a" coefficient at position [i][j]
    virtual double a(int i, int j);

    /// Returns Butcher tableau "b" coefficient at position [i]
    virtual double b(int i);

protected:

    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0

};

} // ODE namespace
} // PHiLiP namespace

#endif
