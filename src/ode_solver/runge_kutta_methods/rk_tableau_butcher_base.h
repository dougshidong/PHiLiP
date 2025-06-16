#ifndef __RK_TABLEAU_BUTCHER_BASE__
#define __RK_TABLEAU_BUTCHER_BASE__

#include <deal.II/base/conditional_ostream.h>

#include <deal.II/grid/tria.h>
#include <deal.II/distributed/shared_tria.h>
#include <deal.II/distributed/tria.h>

#include "rk_tableau_base.h"

namespace PHiLiP {
namespace ODE {

/// Base class for storing the RK method 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RKTableauButcherBase: public RKTableauBase<dim,real,MeshType>
{
public:
    /// Default constructor that will set the constants.
    RKTableauButcherBase(const int n_rk_stages, const std::string rk_method_string_input); 

    /// Returns Butcher tableau "c" coefficient at position [i]
    double get_c(const int i) const;

    /// Calls setters for butcher tableau
    void set_tableau() override;

protected:


    /// Butcher tableau "c"
    dealii::Table<1,double> butcher_tableau_c;
    
    /// Setter for butcher_tableau_a
    virtual void set_a() = 0;

    /// Setter for butcher_tableau_b
    virtual void set_b() = 0;
    
    /// Setter for butcher_tableau_c
    virtual void set_c() = 0;


};

} // ODE namespace
} // PHiLiP namespace

#endif
