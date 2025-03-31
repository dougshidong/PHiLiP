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
    LowStorageRKTableauBase(const int n_rk_stages, const int num_delta, const std::string rk_method_string_input); 

    /// Destructor
    virtual ~LowStorageRKTableauBase() = default;

protected:
    /// String identifying the RK method
    const std::string rk_method_string;
    
    dealii::ConditionalOStream pcout; ///< Parallel std::cout that only outputs on mpi_rank==0
    
public:
    /// Returns Butcher tableau "gamma" coefficient at position [i][j]
    double get_gamma(const int i, const int j) const;

    /// Returns Butcher tableau "beta" coefficient at position [i]
    double get_beta(const int i) const;

    /// Returns Butcher tableau "delta" coefficient at position [i]
    double get_delta(const int i) const;

    /// Returns Butcher tableau "b hat" coefficient at position [i]
    double get_b_hat(const int i) const;

    /// Calls setters for butcher tableau
    void set_tableau();

    
protected:

    /// Butcher tableau "gamma"
    dealii::Table<2,double> butcher_tableau_gamma;

    /// Butcher tableau "beta"
    dealii::Table<1,double> butcher_tableau_beta;
    
    /// Butcher tableau "delta"
    dealii::Table<1,double> butcher_tableau_delta;

    /// Butcher tableau "b hat"
    dealii::Table<1,double> butcher_tableau_b_hat;
    
    /// Setter for gamma
    virtual void set_gamma() = 0;

    /// Setter for beta
    virtual void set_beta() = 0;
    
    /// Setter for delta
    virtual void set_delta() = 0;

    /// Setter for b hat
    virtual void set_b_hat() = 0;

};

} // ODE namespace
} // PHiLiP namespace

#endif
