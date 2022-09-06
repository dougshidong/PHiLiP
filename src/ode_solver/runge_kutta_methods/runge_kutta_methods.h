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
    
};

} // ODE namespace
} // PHiLiP namespace

#endif
