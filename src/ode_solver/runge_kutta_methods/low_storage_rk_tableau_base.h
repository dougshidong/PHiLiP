#ifndef __LOW_STORAGE_RK_TABLEAU_BASE__
#define __LOW_STORAGE_RK_TABLEAU_BASE__

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
class LowStorageRKTableauBase
{
public:
    /// Default constructor that will set the constants.
    LowStorageRKTableauBase(const std::string rk_method_string_input); 

    /// Destructor
    virtual ~LowStorageRKTableauBase() = default;

protected:
    /// String identifying the RK method
    const std::string rk_method_string;
    
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
public:
    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double get_gamma(const int i, const int j) const;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double get_beta(const int i) const;

    /// Returns Butcher tableau "c" coefficient at position [i]
    double get_delta(const int i) const;

    /// Calls setters for butcher tableau
    void set_tableau();

    
protected:

    

    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_gamma;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_beta;
    
    /// Butcher tableau "c"
    dealii::Table<1,double> butcher_tableau_delta;
    
    /// Setter for butcher_tableau_a
    // virtual void set_a() = 0;
    void set_gamma();

    /// Setter for butcher_tableau_b
    // virtual void set_b() = 0;
    void set_beta();
    
    /// Setter for butcher_tableau_c
    // virtual void set_c() = 0;
    void set_delta();


};

} // ODE namespace
} // PHiLiP namespace

#endif
