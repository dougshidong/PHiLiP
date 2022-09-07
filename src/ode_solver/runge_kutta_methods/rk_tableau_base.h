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
protected:
    /// Default constructor that will set the constants.
    RKTableauBase(); 

public:
    /// Destructor
    ~RKTableauBase() {};

    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double a(const int i, const int j) const;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double b(const int i) const;

protected:

    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;

};

} // ODE namespace
} // PHiLiP namespace

#endif
