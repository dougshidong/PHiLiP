#ifndef __RK_TABLEAU_BASE__
#define __RK_TABLEAU_BASE__

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

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
    RKTableauBase(const int n_rk_stages, const std::string rk_method_string_input); 

    /// Calls setters for butcher tableau
    virtual void set_tableau()=0;

    /// Store number of stages
    const int n_rk_stages;
    
    /// Returns Butcher tableau "b" coefficient at position [i]
    /** This is in the base class because RRK must access it for both LSRK and 
     *  standard Butcher RK
     **/
    double get_b(const int i) const;

    
protected:

    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
    /// String identifying the RK method
    const std::string rk_method_string;
    
    /// Butcher tableau "b"
    /** This is in the base class because RRK must access it for both LSRK and 
     *  standard Butcher RK
     **/
    dealii::Table<1,double> butcher_tableau_b;
    

};

} // ODE namespace
} // PHiLiP namespace

#endif
