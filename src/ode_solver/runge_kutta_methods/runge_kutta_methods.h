#ifndef __RUNGE_KUTTA_METHODS__
#define __RUNGE_KUTTA_METHODS__

#include "rk_tableau_base.h"
#include "dg/dg.h"

namespace PHiLiP {
namespace ODE {

/// Third-order strong stability preserving explicit RK
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class SSPRK3Explicit: public RKTableauBase <dim, real, MeshType>
{
public:
    SSPRK3Explicit(int n_rk_stages); ///< Constructor.

    /// Destructor
    ~SSPRK3Explicit() {};
    
    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double a(int i, int j) override;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double b(int i) override;

protected:
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;


};

/// Classical fourth-order explicit RK
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class RK4Explicit: public RKTableauBase <dim, real, MeshType>
{
public:
    RK4Explicit(int n_rk_stages); ///< Constructor.

    /// Destructor
    ~RK4Explicit() {};
    
    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double a(int i, int j) override;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double b(int i) override;

protected:
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;
};

/// Forward Euler (explicit) 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EulerExplicit: public RKTableauBase <dim, real, MeshType>
{
public:
    EulerExplicit(int n_rk_stages); ///< Constructor.

    /// Destructor
    ~EulerExplicit() {};
    
    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double a(int i, int j) override;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double b(int i) override;

protected:
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;
};


/// Implicit Euler 
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class EulerImplicit: public RKTableauBase <dim, real, MeshType>
{
public:
    EulerImplicit(int n_rk_stages); ///< Constructor.

    /// Destructor
    ~EulerImplicit() {};
    
    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double a(int i, int j) override;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double b(int i) override;

protected:
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;
};


/// Second-order diagonally-implicit RK
#if PHILIP_DIM==1
template <int dim, typename real, typename MeshType = dealii::Triangulation<dim>>
#else
template <int dim, typename real, typename MeshType = dealii::parallel::distributed::Triangulation<dim>>
#endif
class DIRK2Implicit: public RKTableauBase <dim, real, MeshType>
{
public:
    DIRK2Implicit(int n_rk_stages); ///< Constructor.

    /// Destructor
    ~DIRK2Implicit() {};
    
    /// Returns Butcher tableau "a" coefficient at position [i][j]
    double a(int i, int j) override;

    /// Returns Butcher tableau "b" coefficient at position [i]
    double b(int i) override;

protected:
    
    /// Butcher tableau "a"
    dealii::Table<2,double> butcher_tableau_a;

    /// Butcher tableau "b"
    dealii::Table<1,double> butcher_tableau_b;
};

} // ODE namespace
} // PHiLiP namespace

#endif
