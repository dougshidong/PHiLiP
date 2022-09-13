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
    SSPRK3Explicit(int n_rk_stages) : RKTableauBase<dim,real,MeshType>(n_rk_stages) { set_tableau();} ///< Constructor.

    /// Destructor
    ~SSPRK3Explicit() {};

protected:

    /// Setter for butcher tableau
    void set_tableau() override;
    
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
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
    RK4Explicit(int n_rk_stages) : RKTableauBase<dim,real,MeshType>(n_rk_stages) { set_tableau();} ///< Constructor.

    /// Destructor
    ~RK4Explicit() {};

protected:

    /// Setter for butcher tableau
    void set_tableau() override;
    
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
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
    EulerExplicit(int n_rk_stages) : RKTableauBase<dim,real,MeshType>(n_rk_stages) { set_tableau();} ///< Constructor.

    /// Destructor
    ~EulerExplicit() {};
    
protected:

    /// Setter for butcher tableau
    void set_tableau() override;
    
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
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
    EulerImplicit(int n_rk_stages) : RKTableauBase<dim,real,MeshType>(n_rk_stages) { set_tableau();} ///< Constructor.

    /// Destructor
    ~EulerImplicit() {};
    
protected:

    /// Setter for butcher tableau
    void set_tableau() override;
    
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
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
    DIRK2Implicit(int n_rk_stages) : RKTableauBase<dim,real,MeshType>(n_rk_stages) { set_tableau();} ///< Constructor.

    /// Destructor
    ~DIRK2Implicit() {};
    
protected:

    /// Setter for butcher tableau
    void set_tableau() override;
    
    /// Setter for butcher_tableau_a
    void set_a() override;

    /// Setter for butcher_tableau_b
    void set_b() override;

    /// Setter for butcher_tableau_c
    void set_c() override;
};

} // ODE namespace
} // PHiLiP namespace

#endif
